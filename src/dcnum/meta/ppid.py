from __future__ import annotations

import collections
import hashlib
import inspect
import pathlib
from typing import Dict, List, Protocol
import warnings


#: Increment this string if there are breaking changes that make
#: previous pipelines unreproducible.
DCNUM_PPID_GENERATION = "7"


class ClassWithPPIDCapabilities(Protocol):
    def get_ppid(self) -> str:
        """full pipeline identifier for the class (instance method)"""
        pass

    def get_ppid_code(self) -> str:
        """string representing the class in the pipeline (classmethod)"""
        pass

    def get_ppid_from_ppkw(self) -> str:
        """pipeline identifier from specific pipeline keywords (classmethod)"""
        pass

    def get_ppkw_from_ppid(self) -> Dict:
        """class keywords from full pipeline identifier (staticmethod)"""
        pass


def compute_pipeline_hash(*, bg_id, seg_id, feat_id, gate_id,
                          dat_id="unknown", gen_id=DCNUM_PPID_GENERATION):
    hasher = hashlib.md5()
    hasher.update("|".join([
        gen_id, dat_id, bg_id, seg_id, feat_id, gate_id]).encode())
    pph = hasher.hexdigest()
    return pph


def convert_to_dtype(value, dtype):
    if isinstance(dtype, str):
        raise ValueError("Annotations are strings, pleace make sure to not "
                         "import __annotations__ from future!")
    elif dtype is bool:
        if isinstance(value, str):
            if value.lower() in ["true", "yes"]:
                value = True
            elif value.lower() in ["false", "no"]:
                value = False
        value = bool(float(value))
    elif dtype in [pathlib.Path, pathlib.Path | str]:
        value = str(value)
    else:
        value = dtype(value)
    return value


def get_class_method_info(class_obj: ClassWithPPIDCapabilities,
                          static_kw_methods: List = None):
    """Return dictionary of class info with static keyword methods docs

    Parameters
    ----------
    class_obj: object
        Class to inspect, must implement the `key` method.
    static_kw_methods: list of callable
        The methods to inspect; all kwargs-only keyword arguments
        are extracted.
    """
    doc = class_obj.__doc__ or class_obj.__init__.__doc__
    info = {
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
            defau[mm] = spec.kwonlydefaults or {}
            annot[mm] = spec.annotations
        info["defaults"] = defau
        info["annotations"] = annot
    return info


def kwargs_to_ppid(cls: ClassWithPPIDCapabilities,
                   method: str,
                   kwargs: Dict,
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
            if isinstance(val, bool):
                val = int(val)  # do not print e.g. "True"
            elif isinstance(val, float):
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
