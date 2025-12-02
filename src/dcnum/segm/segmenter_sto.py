import abc
from typing import Dict

import numpy as np

from .segmenter import Segmenter, assert_labels


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
            Keyword arguments for mask post-processing (see `process_labels`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for ``segment_algorithm``
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
                      bg_off: np.ndarray = None,
                      ):
        """Perform batch segmentation of `images`

        Before segmentation, an optional background offset correction with
        ``bg_off`` is performed. After segmentation, mask postprocessing is
        performed according to the class definition.

        Parameters
        ----------
        images: 3d np.ndarray of shape (N, Y, X)
            The time-series image data. First axis is time.
        bg_off: 1D np.ndarray of length N
            Optional 1D numpy array with background offset

        Notes
        -----
        - If the segmentation algorithm only accepts background-corrected
          images, then `images` must already be background-corrected,
          except for the optional `bg_off`.
        """
        segm = self.segment_algorithm_wrapper()

        if bg_off is not None:
            if not self.requires_background_correction:
                raise ValueError(f"The segmenter {self.__class__.__name__} "
                                 f"does not employ background correction, "
                                 f"but the `bg_off` keyword argument was "
                                 f"passed to `segment_batch`. Please check "
                                 f"your analysis pipeline.")
            images = images - bg_off.reshape(-1, 1, 1)

        # obtain masks or labels
        mols = segm(images)

        # Put everything into a uint16 array
        if mols.dtype == bool:
            # Create output array
            labels = np.zeros_like(mols, dtype=np.uint16)
        else:
            # Modification in-place
            labels = np.asarray(mols, dtype=np.uint16)

        # TODO: Parallelize this
        # Perform mask postprocessing
        if self.mask_postprocessing:
            for ii in range(len(labels)):
                labels[ii] = self.process_labels(mols[ii], **self.kwargs_mask)
        else:
            for ii in range(len(labels)):
                labels[ii] = assert_labels(mols[ii])

        return labels

    def segment_single(self, image, bg_off: float = None):
        """This is a convenience-wrapper around `segment_batch`"""
        if bg_off is None:
            bg_off_batch = None
        else:
            bg_off_batch = np.atleast_1d(bg_off)
        images = image[np.newaxis]
        return self.segment_batch(images, bg_off=bg_off_batch)[0]
