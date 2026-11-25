import gc

from dcnum import segm
import numpy as np

import pytest


def test_segm_mpo_bg_off_single_no_background_correction():
    """
    When an MPOSegmenter does not employ background correction, a
    ValueError is raised when calling `segment_single` with bg_off
    (like for `segment_batch`)."""
    class SegmentMPOTest(segm.MPOSegmenter):
        requires_background_correction = False

        @staticmethod
        def segment_algorithm(image):
            return image > 0.5

    try:
        sm = SegmentMPOTest(num_workers=1, debug=True)
        img = np.full((10, 20), 1.0)
        assert np.all(sm.segment_single(img))

        with pytest.raises(ValueError, match="does not employ background"):
            sm.segment_single(img, bg_off=0.8)

        with pytest.raises(ValueError, match="does not employ background"):
            sm.segment_single_with_labeling(img, bg_off=0.8)
    finally:
        # Make sure other tests do not find this segmenter.
        del SegmentMPOTest
        gc.collect()
        segm.get_segmenters.cache_clear()
        segm.get_available_segmenters.cache_clear()
