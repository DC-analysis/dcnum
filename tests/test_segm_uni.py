import gc

import numpy as np
import pytest

from dcnum import segm


@pytest.fixture
def uni_segmenter_class():
    """A minimal UNISegmenter that does not require PyTorch"""
    class SegmentUNITest(segm.UNISegmenter):
        @staticmethod
        def segment_algorithm(images):
            return images > 0.5

    yield SegmentUNITest

    # Make sure other tests do not find this segmenter.
    del SegmentUNITest
    gc.collect()
    segm.get_segmenters.cache_clear()
    segm.get_available_segmenters.cache_clear()


def test_segm_uni_segment_single_bg_off(uni_segmenter_class):
    sm = uni_segmenter_class()
    sm.requires_background_correction = True
    image = np.full((10, 20), 1.0)
    assert np.all(sm.segment_single(image))
    assert not np.any(sm.segment_single(image, bg_off=0.8))


def test_segm_uni_segment_single_bg_off_no_background_correction(
        uni_segmenter_class):
    """
    When a segmenter does not employ background correction, a ValueError
    is raised when calling `segment_single` with bg_off (like for
    `segment_batch`)."""
    sm = uni_segmenter_class()
    sm.requires_background_correction = False
    image = np.full((10, 20), 1.0)

    with pytest.raises(ValueError, match="does not employ background"):
        sm.segment_batch(image[np.newaxis], bg_off=np.array([0.8]))

    with pytest.raises(ValueError, match="does not employ background"):
        sm.segment_single(image, bg_off=0.8)
