from __future__ import annotations

import collections
import hashlib
import inspect
import pathlib
import types
import typing
import warnings

import numpy as np


DCNUM_PPID_GENERATION = "13"
"""The dcnum pipeline generation.
Increment this string if there are breaking changes that make
previous pipelines unreproducible.
"""


class ClassWithPPIDCapabilities(typing.Protocol):
    def get_ppid(self) -> str:
        """full pipeline identifier for the class (instance method)"""
        return ""

    def get_ppid_code(self) -> str:
        """string representing the class in the pipeline (classmethod)"""
        return ""

    def get_ppid_from_ppkw(self) -> str:
        """pipeline identifier from specific pipeline keywords (classmethod)"""
        return ""

    def get_ppkw_from_ppid(self) -> dict:
        """class keywords from full pipeline identifier (staticmethod)"""
        return {}


def compute_pipeline_hash(*, bg_id, seg_id, feat_id, gate_id,
                          dat_id="unknown", gen_id=DCNUM_PPID_GENERATION):
    hasher = hashlib.md5()
    hasher.update("|".join([
        gen_id, dat_id, bg_id, seg_id, feat_id, gate_id]).encode())
    pph = hasher.hexdigest()
    return pph


def convert_to_dtype(value, dtype):
    """Convert an object to the correct dtype

    If `dtype` is a Union of types, or a list of types, the first
    non-NoneType type is used for conversion.
    """
    if dtype is bool:
        if isinstance(value, str):
            if value.lower() in ["true", "yes"]:
                value = True
            elif value.lower() in ["false", "no"]:
                value = False
        value = bool(float(value))
    elif dtype in [pathlib.Path, pathlib.Path | str]:
        value = str(value)
    else:
        if typing.get_origin(dtype) in {typing.Union, types.UnionType}:
            dtype = dtype.__args__
        if isinstance(dtype, (list, tuple)):
            for dtarg in dtype:
                if dtarg is type(None) or dtarg is None:
                    continue
                else:
                    value = dtarg(value)
                    break
            else:
                raise ValueError(
                    f"Could not convert {value=} of type {type(value)} "
                    f"to type {dtype}")
        else:
            value = dtype(value)
    return value


def get_class_method_info(class_obj: ClassWithPPIDCapabilities,
                          static_kw_methods: list | None = None,
                          static_kw_defaults: dict | None = None,
                          ) -> dict[str, typing.Any]:
    """Return dictionary of class info with static keyword methods docs

    Parameters
    ----------
    class_obj: object
        Class to inspect, must implement the `key` method.
    static_kw_methods: list of callable
        The methods to inspect; all kwargs-only keyword arguments
        are extracted.
    static_kw_defaults: dict
        If a key in this dictionary matches an item in `static_kw_methods`,
        then these are the default values returned in the "defaults"
        dictionary. This is used in cases where a base class does
        implement some annotations, but the subclass does not actually
        use them, because e.g. they are taken from a property such as is
        the case for the mask postprocessing of segmenter classes.
    """
    if static_kw_defaults is None:
        static_kw_defaults = {}
    doc = class_obj.__doc__ or class_obj.__init__.__doc__ or ""
    info: dict[str, typing.Any] = {
        "code": class_obj.get_ppid_code(),
        "doc": doc,
        "title": doc.split("\n")[0],
        }
    if static_kw_methods:
        defau = collections.OrderedDict()
        annot = collections.OrderedDict()
        for mm in static_kw_methods:
            meth = getattr(class_obj, mm)
            spec = inspect.getfullargspec(meth)
            if mm_defaults := static_kw_defaults.get(mm):
                defau[mm] = mm_defaults
            else:
                defau[mm] = spec.kwonlydefaults or {}
            annot[mm] = spec.annotations
            for k, v in annot[mm].items():
                annot[mm][k] = simple_type_eval(v)
        info["defaults"] = defau
        info["annotations"] = annot
    return info


def kwargs_to_ppid(cls: ClassWithPPIDCapabilities,
                   method: str,
                   kwargs: dict,
                   allow_invalid_keys: bool = True):
    info = get_class_method_info(cls, [method, "__init__"])

    concat_strings = []
    if info["defaults"][method]:
        kwdefaults = info["defaults"][method]
        kwdefaults_init = info["defaults"]["__init__"]
        kw_false = (set(kwargs.keys())
                    - set(kwdefaults.keys())
                    - set(kwdefaults_init.keys()))
        if kw_false:
            # This should not have happened.
            msg = (f"Invalid kwargs {kw_false} specified for method "
                   f"'{method}'! Valid kwargs are"
                   f"{sorted(kwdefaults.keys())}. If you wrote this "
                   f"segmenter and had to implement `__init__`, make sure "
                   f"that it accepts all kwonly-arguments its super class "
                   f"accepts. If this is not the case, you are probably "
                   f"passing invalid kwargs to the segmenter."
                   )
            if allow_invalid_keys:
                warnings.warn(msg, UserWarning)
            else:
                raise KeyError(msg)
        kwannot = info["annotations"][method]
        kws = list(kwdefaults.keys())
        kws_abrv = get_unique_prefix(kws)
        for kw, abr in zip(kws, kws_abrv):
            val = kwargs.get(kw, kwdefaults[kw])
            if kwannot[kw] in [pathlib.Path, str | pathlib.Path]:
                # If we have paths as arguments, only use the filename
                path = pathlib.Path(val)
                if path.exists():
                    val = path.name
            if isinstance(val, (bool, np.bool_)):
                val = int(val)  # do not print e.g. "True"
            elif isinstance(val, (float, np.floating)):
                if val == int(val):
                    val = int(val)  # omit the ".0" at the end
            concat_strings.append(f"{abr}={val}")
    return "^".join(concat_strings)


def ppid_to_kwargs(cls, method, ppid):
    """Convert pipeline method id to method keyword arguments

    Notes
    -----
    Keep in mind that when a `method` is changed in a later
    version, new keyword arguments should always be put
    AT THE VERY END of the keyword list. Otherwise, might will
    be ambiguities regarding the abbreviated keys!
    """
    info = get_class_method_info(cls, [method])
    items = ppid.split("^")
    kwargs = {}

    if info["defaults"][method] and items:
        # assemble the individual entries
        entries = []
        for abr, val in [it.split("=") for it in items]:
            entries.append((abr, val))
        # sort the entries according to their length
        # (This is not really necessary, but increases robustness.)
        entries = sorted(entries, key=lambda x: -len(x[0]))

        # populate default keyword arguments
        kwargs.update(info["defaults"][method])
        # keep the keys in their original order, such that we are
        # backwards-compatible with shorter pipeline identifiers
        keys = list(kwargs.keys())

        # determine the correct values by iterating through the info
        used_keys = []
        for abr_key, val in entries:
            for full_key in keys:
                if full_key not in used_keys and full_key.startswith(abr_key):
                    annot = info["annotations"][method][full_key]
                    kwargs[full_key] = convert_to_dtype(val, annot)
                    used_keys.append(full_key)
                    break
            else:
                raise ValueError(f"Unknown abbreviated key '{abr_key}'!")
    return kwargs


def simple_type_eval(type_string: str | type) -> str | type | list:
    """Return the type encoded by a string, e.g. "bool" -> bool

    If `type_string` is already a type, it is passed through.
    If there is no rule to convert `type_string` to a type,
    `type_string` is returned as-is.
    If `type_string` represents a union of types (using '|'), then
    a list of types is returned.
    """
    if isinstance(type_string, str):
        type_string = type_string.strip()
        if type_string.count("|"):
            try:
                return [simple_type_eval(ts) for ts in type_string.split("|")]
            except BaseException:
                pass
        if type_string in ["bool", "dict", "float", "int", "str"]:
            return eval(type_string)
        elif type_string == "None":
            return type(None)

    return type_string


class AbrvStr:
    def __init__(self, string):
        self.string = string
        self.abrv_lengths = [1]  # initialize with minimum length 1

    def __getitem__(self, item):
        return self.string.__getitem__(item)

    @property
    def abrv(self):
        return self.string[:max(self.abrv_lengths)]

    def meet(self, other):
        assert self.string != other.string
        if len(self.string) >= len(other.string):
            a, b = other, self
        else:
            a, b = self, other

        al = 1
        bl = 1

        while b[:bl].startswith(a[:al]):
            if bl == len(a.string):
                bl += 1
                break
            else:
                al += 1
                bl += 1

        a.abrv_lengths.append(al)
        b.abrv_lengths.append(bl)


def get_unique_prefix(str_list):
    """Find unique prefix for a list of strings

    Parameters
    ----------
    str_list: list of str
        List of strings to abbreviate
    """
    size = len(str_list)
    abrv_str_list = [AbrvStr(a) for a in str_list]
    for ii in range(size):
        for jj in range(size):
            if ii != jj:
                abrv_str_list[ii].meet(abrv_str_list[jj])
    return [a.abrv for a in abrv_str_list]
