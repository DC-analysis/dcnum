import abc
import pathlib

import numpy as np
import scipy.ndimage as ndi


from .segmenter import Segmenter


class GPUSegmenter(Segmenter, abc.ABC):
    hardware_processor = "gpu"
    mask_postprocessing = False

    def __init__(self, *args, **kwargs):
        super(GPUSegmenter, self).__init__(*args, **kwargs)

    @staticmethod
    def _get_model_path(model_file):
        """Custom hook that may be defined by subclasses"""
        return pathlib.Path(model_file)

    def segment_batch(self,
                      image_data: np.ndarray,
                      start: int = None,
                      stop: int = None):
        if stop is None or start is None:
            start = 0
            stop = len(image_data)

        image_slice = image_data[start:stop]
        segm = self.segment_frame_wrapper()

        labels = segm(image_slice)

        # Make sure we have integer labels
        if labels.dtype == bool:
            new_labels = np.zeros_like(labels, dtype=np.uint16)
            for ii in range(len(labels)):
                ndi.label(
                    input=labels[ii],
                    output=new_labels[ii],
                    structure=ndi.generate_binary_structure(2, 2))
            labels = new_labels

        return labels
