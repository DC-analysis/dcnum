import types

import pytest

from dcnum import segm
from dcnum.meta import ppid
import numpy as np


SEGM_METH = segm.get_available_segmenters()
SEGM_KEYS = sorted(SEGM_METH.keys())


def test_segmenter_properties():
    assert segm.MPOSegmenter.mask_postprocessing  # sanity check
    assert segm.STOSegmenter.mask_postprocessing  # fixed in 0.21.0
    assert segm.Segmenter.mask_postprocessing  # new default in 0.21.0


@pytest.mark.parametrize("segm_method", SEGM_KEYS)
def test_ppid_no_union_kwonlykwargs(segm_method):
    """Segmenters should define kw-only keyword arguements clear type hint

    This test makes sure that no `UnionType` is used
    (e.g. `str | pathlib.Path`).
    """
    segm_cls = SEGM_METH[segm_method]
    meta = ppid.get_class_method_info(segm_cls,
                                      static_kw_methods=["segment_algorithm"])
    assert meta["code"] == segm_method
    annot = meta["annotations"]["segment_algorithm"]
    for key in annot:
        assert not isinstance(annot[key], types.UnionType), segm_method


def test_segmenter_process_mask_clear_border():
    # labels image with al-filled border values
    label = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1],
        [2, 0, 1, 3, 3, 3, 0, 1],
        [2, 0, 0, 3, 3, 3, 0, 1],
        [2, 0, 0, 3, 3, 3, 0, 2],
        [2, 0, 2, 2, 2, 0, 0, 2],
        [2, 0, 2, 2, 2, 0, 0, 2],
        [2, 2, 2, 2, 2, 2, 2, 2],
    ], dtype=int)

    lbs = segm.Segmenter.process_mask(label,
                                      clear_border=True,
                                      fill_holes=False,
                                      closing_disk=False,
                                      )
    assert np.sum(lbs) > 0
    assert np.sum(lbs > 0) == 9
    assert lbs[:, 0].sum() == 0
    assert lbs[:, -1].sum() == 0
    assert lbs[0, :].sum() == 0
    assert lbs[-1, :].sum() == 0
    assert lbs[3, 3] == 3


def test_segmenter_labeled_mask_clear_border2():
    lab0 = np.array([
        [2, 2, 2, 0, 0, 0, 0, 0],  # bad seed position for floodfill
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    sm = segm.segm_thresh.SegmentThresh(thresh=-6)

    labels = sm.process_mask(lab0,
                             clear_border=False,
                             fill_holes=True,
                             closing_disk=False)

    assert np.sum(labels == 0) > 20, "background should be largest"
    assert np.sum(labels == 1) == 9


def test_segmenter_labeled_mask_spurious_noise_closing():
    lab0 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0],  # noise, 2
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0],  # filled, 1
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 0, 0],  # noise, 3
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    # Structuring element disk 1:
    # [0, 1, 0],
    # [1, 1, 1],
    # [0, 1, 0],

    # After erosion:
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 1, 0, 0, 0],
    # [0, 0, 1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],

    # After dilation:
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 1, 0, 0, 0],
    # [0, 1, 1, 1, 1, 1, 0, 0],
    # [0, 1, 1, 1, 1, 1, 0, 0],
    # [0, 0, 1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],

    sm = segm.segm_thresh.SegmentThresh(thresh=-6)

    labels = sm.process_mask(lab0,
                             clear_border=False,
                             fill_holes=True,
                             closing_disk=1)

    assert np.sum(labels == 0) > 20, "background should be largest"
    assert np.sum(labels == 1) == 16
    assert np.sum(labels) == 16, "only one label"
