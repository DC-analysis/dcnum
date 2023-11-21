from .segmenter_cpu import CPUSegmenter


class SegmentThresh(CPUSegmenter):
    mask_postprocessing = True
    mask_default_kwargs = {
        "clear_border": True,
        "fill_holes": True,
        "closing_disk": 2,
    }

    def __init__(self, thresh=-6, *args, **kwargs):
        """Simple image thresholding segmentation

        Parameters
        ----------
        """
        super(SegmentThresh, self).__init__(thresh=thresh, *args, **kwargs)

    @staticmethod
    def segment_approach(image, *,
                         thresh: float = -6):
        """Mask retrieval as it is done in Shape-In

        Parameters
        ----------
        image: 2d ndarray
            Background-corrected frame image
        thresh: float
            Threshold value for creation of binary mask; a negative value
            means that pixels darker than the background define the threshold
            level.

        Returns
        -------
        mask: 2d boolean ndarray
            Mask image for the give index
        """
        assert thresh < 0, "threshold values above zero not supported!"
        return image < thresh
