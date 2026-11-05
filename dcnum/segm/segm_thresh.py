"""Segmentation by thresholding"""

from .segmenter_uni import UNISegmenter


class SegmentThresh(UNISegmenter):
    """Segmentation by thresholding the background-corrected image

    Pixels whose background-corrected value is below the (negative)
    threshold ``thresh``, i.e. pixels darker than the background, are
    considered part of an event. Since
    ``requires_background_correction`` is True, this segmenter
    receives the background-corrected image data (``image_corr``).
    """
    mask_postprocessing = True
    mask_default_kwargs = {
        "clear_border": True,
        "fill_holes": True,
        "closing_disk": 2,
    }
    requires_background_correction = True

    @staticmethod
    def segment_algorithm(images, *,
                          thresh: float = -6):
        """Mask retrieval using basic thresholding

        Parameters
        ----------
        images: 3d ndarray
            Background-corrected image data of shape (N, Y, X)
        thresh: float
            Threshold value for creation of binary mask; a negative value
            means that pixels darker than the background define the threshold
            level.

        Returns
        -------
        mask: 3d boolean ndarray
            Mask images with the same shape as ``images``
        """
        assert thresh < 0, "threshold values above zero not supported!"
        return images < thresh
