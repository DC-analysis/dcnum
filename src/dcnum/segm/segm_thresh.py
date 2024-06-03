from .segmenter_mpo import MPOSegmenter


class SegmentThresh(MPOSegmenter):
    mask_postprocessing = True
    mask_default_kwargs = {
        "clear_border": True,
        "fill_holes": True,
        "closing_disk": 2,
    }
    requires_background_correction = True

    @staticmethod
    def segment_algorithm(image, *,
                          thresh: float = -6):
        """Mask retrieval using basic thresholding

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
            Mask image for the given index
        """
        assert thresh < 0, "threshold values above zero not supported!"
        return image < thresh
