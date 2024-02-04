import pathlib
import types

import pytest

from dcnum import segm
from dcnum.meta import ppid
import numpy as np

data_path = pathlib.Path(__file__).parent / "data"
SEGM_METH = segm.get_available_segmenters()
SEGM_KEYS = sorted(SEGM_METH.keys())


class MockImageData:
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # other, 3
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    def get_chunk(self, chunk_index):
        image = np.array(-(10 + chunk_index) * self.mask, dtype=np.int16)
        chunk = np.stack([image] * 100, dtype=np.int16)
        return chunk


def test_segmenter_labeled_mask():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # other, 3
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    sm1 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": True,
                                                      "closing_disk": 0,
                                                      })
    labels1 = sm1.segment_frame(-10 * mask)
    assert np.sum(labels1 != 0) == 21
    assert len(np.unique(labels1)) == 3  # (bg, filled, other)
    assert np.sum(labels1 == 1) == 9
    # due to the relabeling done in `fill_holes`, the index of "other" is "3"
    assert np.sum(labels1 == 2) == 12

    sm2 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels2 = sm2.segment_frame(-10 * mask)
    _, l2a, l2b = np.unique(labels2)
    assert np.sum(labels2 != 0) == 20
    assert len(np.unique(labels2)) == 3  # (bg, filled, other)
    assert np.sum(labels2 == l2a) == 8
    assert np.sum(labels2 == l2b) == 12

    sm3 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels3 = sm3.segment_frame(-10 * mask)
    assert np.sum(labels3 != 0) == 30
    assert len(np.unique(labels3)) == 4  # (bg, filled, border, other)
    assert np.sum(labels3 == 1) == 8
    assert np.sum(labels3 == 2) == 10
    assert np.sum(labels3 == 3) == 12

    sm4 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": True,
                                                      "closing_disk": 0,
                                                      })
    labels4 = sm4.segment_frame(-10 * mask)
    assert np.sum(labels4 != 0) == 31
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 10
    assert np.sum(labels4 == 3) == 12


def test_segmenter_labeled_mask_closing_disk():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],  # other, 3
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    sm1 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": True,
                                                      "closing_disk": 1,
                                                      })
    labels1 = sm1.segment_frame(-10 * mask)
    assert np.sum(labels1 != 0) == 32
    assert len(np.unique(labels1)) == 3  # (bg, filled, other)
    assert np.sum(labels1 == 1) == 9
    # due to the relabeling done in `fill_holes`, the index of "other" is "3"
    assert np.sum(labels1 == 2) == 23

    sm2 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": False,
                                                      "closing_disk": 1,
                                                      })
    labels2 = sm2.segment_frame(-10 * mask)
    _, l2a, l2b = np.unique(labels2)
    assert np.sum(labels2 != 0) == 27
    assert len(np.unique(labels2)) == 3  # (bg, filled, other)
    assert np.sum(labels2 == l2a) == 9
    assert np.sum(labels2 == l2b) == 18

    sm3 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": False,
                                                      "closing_disk": 1,
                                                      })
    labels3 = sm3.segment_frame(-10 * mask)
    assert np.sum(labels3 != 0) == 35
    assert len(np.unique(labels3)) == 4  # (bg, filled, border, other)
    assert np.sum(labels3 == 1) == 9
    assert np.sum(labels3 == 2) == 8
    assert np.sum(labels3 == 3) == 18

    sm4 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": True,
                                                      "closing_disk": 1,
                                                      })
    labels4 = sm4.segment_frame(-10 * mask)
    assert np.sum(labels4 != 0) == 40
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 8
    assert np.sum(labels4 == 3) == 23


def test_segmenter_labeled_mask_fill_holes_int32():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # other, 3
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    sm1 = segm.segm_thresh.SegmentThresh(thresh=-6)
    labels = np.array(sm1.segment_frame(-10 * mask), dtype=np.int64)
    # sanity checks
    assert labels.dtype == np.int64
    assert labels.dtype != np.int32
    labels_2 = sm1.process_mask(labels,
                                clear_border=False,
                                fill_holes=True,
                                closing_disk=False)
    assert np.allclose(labels, labels_2)
    assert labels_2.dtype == np.int32


def test_segmenter_segment_chunk():
    with segm.segm_thresh.SegmentThresh(thresh=-12, debug=True) as sm:
        image_data = MockImageData()
        labels_1 = np.copy(sm.segment_chunk(image_data, 0))  # below threshold
        assert sm.image_array.min() == -10
        labels_2 = np.copy(sm.segment_chunk(image_data, 10))  # above threshold
        assert sm.image_array.min() == -20
        assert np.all(labels_1 == 0)
        assert not np.all(labels_2 == 0)


def test_cpu_segmenter_getsetstate():
    sm1 = segm.segm_thresh.SegmentThresh(thresh=-12, debug=True)
    with segm.segm_thresh.SegmentThresh(thresh=-12, debug=True) as sm2:
        image_data = MockImageData()
        # Do some processing so that we have workers
        sm2.segment_chunk(image_data, 0)
        # get the state
        state = sm2.__getstate__()
        # set the state
        sm1.__setstate__(state)
        # and here we test for the raw data that was transferred
        assert not np.all(sm1.image_array == sm2.image_array)
        assert np.all(sm1.mp_image_raw == sm2.mp_image_raw)


@pytest.mark.parametrize("segm_method", SEGM_KEYS)
def test_ppid_no_union_kwonlykwargs(segm_method):
    """Segmenters should define kw-only keyword arguements clear type hint

    This test makes sure that no `UnionType` is used
    (e.g. `str | pathlib.Path`).
    """
    segm_cls = SEGM_METH[segm_method]
    meta = ppid.get_class_method_info(segm_cls,
                                      static_kw_methods=["segment_approach"])
    assert meta["code"] == segm_method
    annot = meta["annotations"]["segment_approach"]
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
