"""Pipeline job definition"""
from __future__ import annotations

import collections
import copy
import inspect
import logging
import pathlib
from typing import Any, Literal

from ..common import cpu_count
from ..feat import QueueEventExtractor
from ..feat.feat_background.base import get_available_background_methods
from ..feat.gate import Gate
from ..meta.ppid import compute_pipeline_hash, DCNUM_PPID_GENERATION
from ..read import HDF5Data
from ..segm import get_segmenters
from ..write.writer import get_compression_kwargs


class DCNumPipelineJob:
    def __init__(self,
                 path_in: pathlib.Path | str,
                 path_out: pathlib.Path | str | None = None,
                 data_code: str = "hdf",
                 data_kwargs: dict | None = None,
                 background_code: str = "sparsemed",
                 background_kwargs: dict | None = None,
                 segmenter_code: str = "thresh",
                 segmenter_kwargs: dict | None = None,
                 feature_code: str = "legacy",
                 feature_kwargs: dict | None = None,
                 gate_code: str = "norm",
                 gate_kwargs: dict | None = None,
                 basin_strategy: Literal["drain", "tap"] = "drain",
                 compression: str = "zstd-5",
                 num_procs: int | None = None,
                 log_level: int = logging.INFO,
                 debug: bool = False,
                 ):
        """Pipeline job recipe

        Parameters
        ----------
        path_in: pathlib.Path | str
            input data path
        path_out: pathlib.Path | str
            output data path; defaults to ``<input stem>_dcn.rtdc``
            in the directory of ``path_in``
        data_code: str
            identification code of input data reader to use
        data_kwargs: dict
            keyword arguments for data reader; If "pixel_size" is not
            specified, it is read from the input file.
        background_code: str
            identification code of background data computation method
        background_kwargs: dict
            keyword arguments for background data computation method
        segmenter_code: str
            identification code of segmenter to use
        segmenter_kwargs: dict
            keyword arguments for segmenter; mask post-processing
            keyword arguments are passed via the "kwargs_mask" key
        feature_code: str
            identification code of feature extractor
        feature_kwargs: dict
            keyword arguments for feature extractor
        gate_code: str
            identification code for gating/event filtering class
        gate_kwargs: dict
            keyword arguments for gating/event filtering class
        basin_strategy: str
            strategy on how to handle event data; In principle, not all
            events have to be stored in the output file if basins are
            defined, linking back to the original file.

            - You can "drain" all basins which means that the output file
              will contain all features, but will also be very big.
            - You can "tap" the basins, including the input file, which means
              that the output file will be comparatively small.
        compression: str
            compression algorithm to use; Set this to "none" to disable
            compression. Currently, only the Zstandard compression
            algorithm may be used, with the least compression "zstd-1"
            and the best compression "zstd-9". The default "zstd-5" is
            a trade-off. Set the compression to a higher number if the
            bottleneck is disk-IO. Set the compression to a lower number
            if the bottleneck is the CPU. Note that "zstd-5" is the
            accepted minimum compression setting for long-term data
            storage in the DC universe (enforced e.g. by DCOR-Aid).
        num_procs: int
            Number of processes to use; defaults to
            :func:`dcnum.common.cpu_count`
        log_level: int
            Logging level to use.
        debug: bool
            Whether to set logging level to "DEBUG" (overrides
            ``log_level``) and use threads instead of processes

        Notes
        -----
        The keyword argument dictionaries (e.g. ``data_kwargs``) are
        copied, so the dictionaries passed by the caller are not
        modified.
        """
        self.kwargs = {}
        """keyword arguments of this job (with defaults filled in)"""

        # Evaluate the annotations (they are strings due to
        # `from __future__ import annotations`).
        sig = inspect.signature(DCNumPipelineJob.__init__, eval_str=True)
        locs = locals()
        for arg, param in sig.parameters.items():
            if arg == "self":
                continue
            value = locs[arg]
            if param.annotation == dict | None:
                # Do not modify the dictionaries passed by the caller.
                value = {} if value is None else copy.deepcopy(value)
            self.kwargs[arg] = value
        # Set default pixel size for this job
        if "pixel_size" not in self.kwargs["data_kwargs"]:
            # Extract from input file
            with HDF5Data(path_in) as hd:
                self.kwargs["data_kwargs"]["pixel_size"] = hd.pixel_size
        # Set default output path
        if path_out is None:
            pin = pathlib.Path(path_in)
            path_out = pin.with_name(pin.stem + "_dcn.rtdc")
        # Set logging level to DEBUG in debugging mode
        if self.kwargs["debug"]:
            self.kwargs["log_level"] = logging.DEBUG
        self.kwargs["path_out"] = pathlib.Path(path_out)
        # Set default mask kwargs for segmenter
        self.kwargs["segmenter_kwargs"].setdefault("kwargs_mask", {})
        # Set default number of processes
        if num_procs is None:
            self.kwargs["num_procs"] = cpu_count()

    def __getitem__(self, item):
        """Return a copy of the keyword argument ``item``

        Since a copy is returned, modifying the returned value (e.g.
        ``job["data_kwargs"]["pixel_size"] = 0.3``) does not modify the
        job. Use :attr:`kwargs` for that.
        """
        return copy.deepcopy(self.kwargs[item])

    def __getstate__(self):
        state = copy.deepcopy(self.kwargs)
        return state

    def __setstate__(self, state):
        if not hasattr(self, "kwargs"):
            self.kwargs = {}
        self.kwargs.clear()
        self.kwargs.update(copy.deepcopy(state))

    def assert_pp_codes(self):
        """Sanity check of ``self.kwargs``"""
        # PPID classes with only one option
        for cls, key in [
            (HDF5Data, "data_code"),
            (Gate, "gate_code"),
            (QueueEventExtractor, "feature_code"),
        ]:
            code_act = self.kwargs[key]
            code_exp = cls.get_ppid_code()
            if code_act != code_exp:
                raise ValueError(f"Invalid code '{code_act}' for '{key}', "
                                 f"expected '{code_exp}'!")
        # PPID classes with multiple options
        for options, key in [
            (get_available_background_methods(), "background_code"),
            (get_segmenters(), "segmenter_code"),
        ]:
            code_act = self.kwargs[key]
            if code_act not in options:
                raise ValueError(f"Invalid code '{code_act}' for '{key}', "
                                 f"expected one of '{options}'!")

    def get_hdf5_dataset_kwargs(self) -> dict:
        """Validate and return output HDF5 Dataset keyword arguments
        """
        return get_compression_kwargs(self.kwargs["compression"])

    def get_ppid(self, ret_hash=False, ret_dict=False):
        """Return the pipeline identifier of this job

        The pipeline identifier (PPID) is composed of the generation
        identifier and the PPIDs of the data reader, background
        method, segmenter, feature extractor, and gate, joined by
        ``"|"``.

        Parameters
        ----------
        ret_hash: bool
            whether to also return the pipeline hash
        ret_dict: bool
            whether to also return the dictionary of the individual
            identifiers

        Returns
        -------
        ppid: str
            pipeline identifier
        pp_hash: str
            pipeline hash computed with
            :func:`~dcnum.meta.ppid.compute_pipeline_hash`
            (only if ``ret_hash`` is True)
        pp_dict: collections.OrderedDict
            individual identifiers with the keys "gen_id", "dat_id",
            "bg_id", "seg_id", "feat_id", and "gate_id"
            (only if ``ret_dict`` is True)

        Notes
        -----
        If neither ``ret_hash`` nor ``ret_dict`` is set, only the
        pipeline identifier string is returned, otherwise a list.
        """
        self.assert_pp_codes()
        pp_hash_kw = collections.OrderedDict()
        pp_hash_kw["gen_id"] = DCNUM_PPID_GENERATION
        for pp_kw, cls, cls_kw in [
            ("dat_id", HDF5Data, "data_kwargs"),
            ("bg_id",
             get_available_background_methods()[
                 self.kwargs["background_code"]],
             "background_kwargs"),
            ("seg_id",
             get_segmenters()[self.kwargs["segmenter_code"]],
             "segmenter_kwargs"),
            ("feat_id", QueueEventExtractor, "feature_kwargs"),
            ("gate_id", Gate, "gate_kwargs"),
        ]:
            pp_hash_kw[pp_kw] = cls.get_ppid_from_ppkw(self.kwargs[cls_kw])

        ppid = "|".join(pp_hash_kw.values())

        ret: list[Any] = [ppid]
        if ret_hash:
            pp_hash = compute_pipeline_hash(**pp_hash_kw)
            ret.append(pp_hash)
        if ret_dict:
            ret.append(pp_hash_kw)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def get_segmenter_class(self):
        """Return the class of the segmenter associated with this job"""
        return get_segmenters()[self.kwargs["segmenter_code"]]

    def validate(self):
        """Make sure the pipeline will run given the job kwargs

        Returns
        -------
        bool
            Always True (for testing convenience)

        Raises
        ------
        dcnum.segm.SegmenterNotApplicableError
            the segmenter is incompatible with the input path
        """
        # Check segmenter applicability
        seg_cls = self.get_segmenter_class()
        with HDF5Data(self.kwargs["path_in"]) as hd:
            seg_cls.validate_applicability(
                segmenter_kwargs=self.kwargs["segmenter_kwargs"],
                logs=hd.logs,
                meta=hd.meta)
        return True
