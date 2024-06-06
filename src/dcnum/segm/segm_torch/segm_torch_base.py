import pathlib

from ...meta import paths

from ..segmenter import Segmenter


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
