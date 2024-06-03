import abc
from typing import Dict

import numpy as np
import scipy.ndimage as ndi


from .segmenter import Segmenter


class STOSegmenter(Segmenter, abc.ABC):
    hardware_processor = "gpu"

    def __init__(self,
                 *,
                 num_workers: int = None,
                 kwargs_mask: Dict = None,
                 debug: bool = False,
                 **kwargs
                 ):
        """Segmenter with single thread operation

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for `segment_algorithm`
            defined in the subclass.
        """
        if num_workers not in [None, 1]:
            raise ValueError(f"Number of workers must not be larger than 1 "
                             f"for GPU segmenter, got '{num_workers}'!")
        super(STOSegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)

    def segment_batch(self,
                      images: np.ndarray,
                      start: int = None,
                      stop: int = None,
                      bg_off: np.ndarray = None,
                      ):
        """Perform batch segmentation of `images`

        Before segmentation, an optional background offset correction with
        `bg_off` is performed. After segmentation, mask postprocessing is
        performed according to the class definition.

        Parameters
        ----------
        images: 3d np.ndarray
            The time-series image data. First axis is time.
        start: int
            First index to analyze in `images`
        stop: int
            Index after the last index to analyze in `images`
        bg_off: 1D np.ndarray
            Optional 1D numpy array with background offset

        Notes
        -----
        - If the segmentation algorithm only accepts background-corrected
          images, then `images` must already be background-corrected,
          except for the optional `bg_off`.
        """
        if stop is None or start is None:
            start = 0
            stop = len(images)

        image_slice = images[start:stop]
        segm = self.segment_algorithm_wrapper()

        if bg_off is not None:
            if not self.requires_background_correction:
                raise ValueError(f"The segmenter {self.__class__.__name__} "
                                 f"does not employ background correction, "
                                 f"but the `bg_off` keyword argument was "
                                 f"passed to `segment_chunk`. Please check "
                                 f"your analysis pipeline.")
            image_slice = image_slice - bg_off.reshape(-1, 1, 1)
        labels = segm(image_slice)

        # Make sure we have integer labels and perform mask postprocessing
        if labels.dtype == bool:
            new_labels = np.zeros_like(labels, dtype=np.uint16)
            for ii in range(len(labels)):
                ndi.label(
                    input=labels[ii],
                    output=new_labels[ii],
                    structure=ndi.generate_binary_structure(2, 2))
            labels = new_labels

        # Perform mask postprocessing
        if self.mask_postprocessing:
            for ii in range(len(labels)):
                labels[ii] = self.process_mask(labels[ii], **self.kwargs_mask)

        return labels

    def segment_single(self, image, bg_off: float = None):
        """This is a convenience-wrapper around `segment_batch`"""
        if bg_off is None:
            bg_off_batch = None
        else:
            bg_off_batch = np.atleast_1d(bg_off)
        images = image[np.newaxis]
        return self.segment_batch(images, bg_off=bg_off_batch)[0]
