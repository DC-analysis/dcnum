# flake8: noqa: F821
import pytest

from dcnum import segm

SEGM_METH = segm.get_available_segmenters()
SEGM_KEYS = sorted(SEGM_METH.keys())


def test_ppid_nomask_segmenter():
    class SegmentNoMask(segm.MPOSegmenter):
        mask_postprocessing = False

        def __init__(self, thresh=-6, *args, **kwargs):
            """Threshold for testing without mask"""
            super(SegmentNoMask, self).__init__(thresh=thresh, *args, **kwargs)

        @staticmethod
        def segment_algorithm(image, *,
                              thresh: float = -6):
            return image < thresh

    segm.get_available_segmenters.cache_clear()

    ppid1 = SegmentNoMask.get_ppid_from_ppkw({"thresh": -3})
    assert ppid1 == "nomask:t=-3"

    ppkw1 = SegmentNoMask.get_ppkw_from_ppid("nomask:t=-5")
    assert "kwargs_mask" not in ppkw1
    assert len(ppkw1) == 1
    assert ppkw1["thresh"] == -5

    with pytest.raises(ValueError,
                       match="does not support mask postprocessing"):
        SegmentNoMask.get_ppid_from_ppkw(
            kwargs={"thresh": -3},
            kwargs_mask={"clear_border": True})

    # cleanup
    del SegmentNoMask
    segm.get_available_segmenters.cache_clear()


def test_ppid_nomask_segmenter_control():
    with pytest.raises(KeyError,
                       match="must be either specified as keyword argument"):
        segm.segm_thresh.SegmentThresh.get_ppid_from_ppkw({"thresh": -3})

    ppid2 = segm.segm_thresh.SegmentThresh.get_ppid_from_ppkw(
        kwargs={"thresh": -3},
        kwargs_mask={"clear_border": True})
    assert ppid2 == "thresh:t=-3:cle=1^f=1^clo=2"
