import collections
import copy
import hdf5plugin
import inspect
import logging
import pathlib
import re
from typing import Dict, Literal

from ..common import cpu_count
from ..feat import QueueEventExtractor
from ..feat.feat_background.base import get_available_background_methods
from ..feat.gate import Gate
from ..meta.ppid import compute_pipeline_hash, DCNUM_PPID_GENERATION
from ..read import HDF5Data
from ..segm import get_available_segmenters


class DCNumPipelineJob:
    def __init__(self,
                 path_in: pathlib.Path | str,
                 path_out: pathlib.Path | str = None,
                 data_code: str = "hdf",
                 data_kwargs: Dict = None,
                 background_code: str = "sparsemed",
                 background_kwargs: Dict = None,
                 segmenter_code: str = "thresh",
                 segmenter_kwargs: Dict = None,
                 feature_code: str = "legacy",
                 feature_kwargs: Dict = None,
                 gate_code: str = "norm",
                 gate_kwargs: Dict = None,
                 basin_strategy: Literal["drain", "tap"] = "drain",
                 compression: str = "zstd-5",
                 num_procs: int = None,
                 log_level: int = logging.INFO,
                 debug: bool = False,
                 ):
        """Pipeline job recipe

        Parameters
        ----------
        path_in: pathlib.Path | str
            input data path
        path_out: pathlib.Path | str
            output data path
        data_code: str
            identification code of input data reader to use
        data_kwargs: dict
            keyword arguments for data reader
        background_code: str
            identification code of background data computation method
        background_kwargs: dict
            keyword arguments for background data computation method
        segmenter_code: str
            identification code of segmenter to use
        segmenter_kwargs: dict
            keyword arguments for segmenter
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
            Number of processes to use
        log_level: int
            Logging level to use.
        debug: bool
            Whether to set logging level to "DEBUG" and
            use threads instead of processes
        """
        self.kwargs = {}
        """initialize keyword arguments for this job"""

        spec = inspect.getfullargspec(DCNumPipelineJob.__init__)
        locs = locals()
        for arg in spec.args:
            if arg == "self":
                continue
            value = locs[arg]
            if value is None and spec.annotations[arg] is Dict:
                value = {}
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
        """Sanity check of `self.kwargs`"""
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
            (get_available_segmenters(), "segmenter_code"),
        ]:
            code_act = self.kwargs[key]
            if code_act not in options:
                raise ValueError(f"Invalid code '{code_act}' for '{key}', "
                                 f"expected one of '{options}'!")

    def get_hdf5_dataset_kwargs(self) -> dict:
        """Validate and return output HDF5 Dataset keyword arguments
        """
        cp = str(self.kwargs["compression"]).lower().strip()
        ds_kw = {"fletcher32": True}
        if cp == "none":
            # No compression
            ds_kw["compression"] = None
            ds_kw["compression_opts"] = None
        elif re.match("^zstd-[1-9]$", cp):
            # Zstandard compression
            clevel = int(cp[-1])
            for key, val in dict(hdf5plugin.Zstd(clevel=clevel)).items():
                ds_kw[key] = val
        else:
            raise ValueError(f"Unsupported compression setting '{cp}'")

        return ds_kw

    def get_ppid(self, ret_hash=False, ret_dict=False):
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
             get_available_segmenters()[self.kwargs["segmenter_code"]],
             "segmenter_kwargs"),
            ("feat_id", QueueEventExtractor, "feature_kwargs"),
            ("gate_id", Gate, "gate_kwargs"),
        ]:
            pp_hash_kw[pp_kw] = cls.get_ppid_from_ppkw(self.kwargs[cls_kw])

        ppid = "|".join(pp_hash_kw.values())

        ret = [ppid]
        if ret_hash:
            pp_hash = compute_pipeline_hash(**pp_hash_kw)
            ret.append(pp_hash)
        if ret_dict:
            ret.append(pp_hash_kw)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def validate(self):
        """Make sure the pipeline will run given the job kwargs

        Returns
        -------
        True:
            for testing convenience

        Raises
        ------
        dcnum.segm.SegmenterNotApplicableError:
            the segmenter is incompatible with the input path
        """
        # Check segmenter applicability applicability
        seg_cls = get_available_segmenters()[self.kwargs["segmenter_code"]]
        with HDF5Data(self.kwargs["path_in"]) as hd:
            seg_cls.validate_applicability(
                segmenter_kwargs=self.kwargs["segmenter_kwargs"],
                logs=hd.logs,
                meta=hd.meta)
        return True
