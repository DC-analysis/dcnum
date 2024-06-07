import functools
import pathlib
import re
from typing import Dict

from ...meta import paths

from ..segmenter import Segmenter, SegmenterNotApplicableError

from .torch_model import load_model


class TorchSegmenterBase(Segmenter):
    """Torch segmenters that use a pretrained model for segmentation"""
    requires_background_correction = False
    mask_postprocessing = True
    mask_default_kwargs = {
        "clear_border": True,
        "fill_holes": True,
        "closing_disk": 0,
    }

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs, kwargs_mask=None):
        kwargs_new = kwargs.copy()
        # Make sure that the `model_file` kwarg is actually just a filename
        # so that the pipeline identifier only contains the name, but not
        # the full path.
        if "model_file" in kwargs:
            model_file = kwargs["model_file"]
            mpath = pathlib.Path(model_file)
            if mpath.exists():
                # register the location of the file in the search path
                # registry so other threads/processes will find it.
                paths.register_search_path("torch_model_files", mpath.parent)
                kwargs_new["model_file"] = mpath.name
        return super(TorchSegmenterBase, cls).get_ppid_from_ppkw(kwargs_new,
                                                                 kwargs_mask)

    @classmethod
    def validate_applicability(cls,
                               segmenter_kwargs: Dict,
                               meta: Dict = None,
                               logs: Dict = None):
        """Validate the applicability of this segmenter for a dataset

        The applicability is defined by the metadata in the segmentation
        model.

        Parameters
        ----------
        segmenter_kwargs: dict
            Keyword arguments for the segmenter
        meta: dict
            Dictionary of metadata from an :class:`HDF5Data` instance
        logs: dict
            Dictionary of logs from an :class:`HDF5Data` instance

        Returns
        -------
        applicable: bool
            True if the segmenter is applicable to the dataset

        Raises
        ------
        SegmenterNotApplicable
            If the segmenter is not applicable to the dataset
        """
        if "model_file" not in segmenter_kwargs:
            raise ValueError("A `model_file` must be provided in the "
                             "`segmenter_kwargs` to validate applicability")

        model_file = segmenter_kwargs["model_file"]
        _, model_meta = load_model(model_file, device="cpu")

        reasons_list = []
        validators = {
            "meta": functools.partial(
                cls._validate_applicability_item,
                data_dict=meta,
                reasons_list=reasons_list),
            "logs": functools.partial(
                cls._validate_applicability_item,
                # convert logs to strings
                data_dict={key: "\n".join(val) for key, val in logs.items()},
                reasons_list=reasons_list)
        }
        for item in model_meta.get("validation", []):
            it = item["type"]
            if it in validators:
                validators[it](item)
            else:
                reasons_list.append(
                    f"invalid validation type {it} in {model_file}")

        if reasons_list:
            raise SegmenterNotApplicableError(segmenter_class=cls,
                                              reasons_list=reasons_list)

        return True

    @staticmethod
    def _validate_applicability_item(item, data_dict, reasons_list):
        """Populate `reasons_list` with invalid entries

        Example `data_dict`::

            {"type": "meta",
             "key": "setup:region",
             "allow-missing-key": False,
             "regexp": "^channel$",
             "regexp-negate": False,
             "reason": "only channel region supported",
             }
        """
        key = item["key"]
        if key in data_dict:
            regexp = re.compile(item["regexp"])
            matched = bool(regexp.match(data_dict[key]))
            negate = item.get("regexp-negate", False)
            valid = matched if not negate else not matched
            if not valid:
                reasons_list.append(item.get("reason", "unknown reason"))
        elif not item.get("allow-missing-key", False):
            reasons_list.append(f"Key '{key}' missing in {item['type']}")
