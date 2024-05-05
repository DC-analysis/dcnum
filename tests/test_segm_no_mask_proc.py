import pathlib

import pytest

from dcnum import segm

data_path = pathlib.Path(__file__).parent / "data"
SEGM_METH = segm.get_available_segmenters()
SEGM_KEYS = sorted(SEGM_METH.keys())


def test_ppid_nomask_segmenter():
    class SegmentNoMask(segm.segm_thresh.SegmentThresh):
        mask_postprocessing = False

    ppid1 = SegmentNoMask.get_ppid_from_ppkw({"thresh": -3})
    assert ppid1 == "nomask:t=-3"

    with pytest.raises(ValueError,
                       match="does not support mask postprocessing"):
        SegmentNoMask.get_ppid_from_ppkw(
            kwargs={"thresh": -3},
            kwargs_mask={"clear_border": True})

    # cleanup
    del SegmentNoMask


def test_ppid_nomask_segmenter_control():

    with pytest.raises(KeyError,
                       match="must be either specified as keyword argument"):
        segm.segm_thresh.SegmentThresh.get_ppid_from_ppkw({"thresh": -3})

    ppid2 = segm.segm_thresh.SegmentThresh.get_ppid_from_ppkw(
            kwargs={"thresh": -3},
            kwargs_mask={"clear_border": True})
    assert ppid2 == "thresh:t=-3:cle=1^f=1^clo=2"
