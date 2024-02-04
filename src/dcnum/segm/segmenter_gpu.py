import abc
from typing import Dict

import numpy as np
import scipy.ndimage as ndi


from .segmenter import Segmenter


class GPUSegmenter(Segmenter, abc.ABC):
    hardware_processor = "gpu"
    mask_postprocessing = False

    def __init__(self,
                 *,
                 num_workers: int = None,
                 kwargs_mask: Dict = None,
                 debug: bool = False,
                 **kwargs
                 ):
        """GPU base segmenter

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for `segment_approach`
            defined in the subclass.
        """
        if num_workers not in [None, 1]:
            raise ValueError(f"Number of workers must not be larger than 1 "
                             f"for GPU segmenter, got '{num_workers}'!")
        super(GPUSegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)

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
