import abc

import numpy as np

from .segmenter import Segmenter


class UNISegmenter(Segmenter, abc.ABC):
    hardware_processor = "cpu"

    def __init__(self,
                 *,
                 kwargs_mask: dict | None = None,
                 debug: bool = False,
                 **kwargs
                 ):
        """Segmenter for use with `UniversalWorker`

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
        super(UNISegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)

    def segment_batch(self,
                      images: np.ndarray,
                      bg_off: np.ndarray | None = None,
                      mask_out: np.ndarray | None = None,
                      ) -> np.ndarray:
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
        mask_out: 3d boolean np.ndarray of shape (N, Y, X)
            Optional output array into which mask data are written

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

        if mask_out is None:
            mask_out = np.zeros_like(images, dtype=np.bool)

        for idx in range(len(images)):
            mask_out[idx] = segm(images[idx])

        return mask_out

    def segment_single(self, image, bg_off: float | None = None):
        """This is a convenience-wrapper around `segment_batch`"""
        segm = self.segment_algorithm_wrapper()

        if bg_off is not None:
            image = image - bg_off

        return segm(image)
