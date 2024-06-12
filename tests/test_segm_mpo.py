from dcnum import segm
import numpy as np

import pytest

from helper_methods import MockImageData


def test_segm_mpo_bg_off_batch():
    img = np.array([
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0, -5, -5, -5, 0, 0],
        [0, 0,  0, -5, -5, -5, 0, 0],  # above threshold
        [0, 0,  0, -5, -5, -5, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0, -9, -9, -9,  0, 0, 0],
        [0, 0, -9,  0, -9,  0, 0, 0],  # filled, below threshold
        [0, 0, -9, -9, -9,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
    ], dtype=int)

    with segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"clear_border": True,
                                                     "fill_holes": True,
                                                     "closing_disk": 0,
                                                     }) as sm:
        labels = sm.segment_batch(images=np.array([img, img]),
                                  bg_off=np.array([1.5, 0.9])
                                  )
    label1_exp = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],  # above threshold
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0],  # filled, below threshold
        [0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    label2_exp = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],  # not enough offset
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # filled, below threshold
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    assert np.all(labels[0] == label1_exp)
    assert np.all(labels[1] == label2_exp)


def test_segm_sto_bg_off_no_background_correction():
    """
    When a segmenter does not employ background correction, a ValueError
    is raised when calling `segment_chunk` with bg_off."""
    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"clear_border": True,
                                                     "fill_holes": True,
                                                     "closing_disk": 0,
                                                     })
    # This will raise the value error below
    sm.requires_background_correction = False

    im = MockImageData()
    with pytest.raises(ValueError, match="does not employ background"):
        sm.segment_chunk(im, chunk=1, bg_off=np.ones(100, dtype=float))


def test_segm_mpo_bg_off_single():
    img = np.array([
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0, -5, -5, -5, 0, 0],
        [0, 0,  0, -5, -5, -5, 0, 0],  # above threshold
        [0, 0,  0, -5, -5, -5, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0, -9, -9, -9,  0, 0, 0],
        [0, 0, -9,  0, -9,  0, 0, 0],  # filled, below threshold
        [0, 0, -9, -9, -9,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
        [0, 0,  0,  0,  0,  0, 0, 0],
    ], dtype=int)

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"clear_border": True,
                                                     "fill_holes": True,
                                                     "closing_disk": 0,
                                                     })
    label1 = sm.segment_single(image=img, bg_off=1.5)
    label1_exp = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],  # above threshold
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0],  # filled, below threshold
        [0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    assert np.all(label1 == label1_exp)

    label2 = sm.segment_single(image=img, bg_off=0.9)
    label2_exp = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],  # not enough offset
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # filled, below threshold
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    assert np.all(label2 == label2_exp)


def test_segm_mpo_labeled_mask():
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
    labels1 = sm1.segment_single(-10 * mask)
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
    labels2 = sm2.segment_single(-10 * mask)
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
    labels3 = sm3.segment_single(-10 * mask)
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
    labels4 = sm4.segment_single(-10 * mask)
    assert np.sum(labels4 != 0) == 31
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 10
    assert np.sum(labels4 == 3) == 12


def test_segm_mpo_labeled_mask_fill_holes():
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
                                                      "closing_disk": 0,
                                                      })
    labels1 = sm1.segment_single(-10 * mask)
    assert np.sum(labels1 != 0) == 32
    assert len(np.unique(labels1)) == 3  # (bg, filled, other)
    assert np.sum(labels1 == 1) == 9
    # due to the relabeling done in `fill_holes`, the index of "other" is "3"
    assert np.sum(labels1 == 2) == 23

    sm2 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels2 = sm2.segment_single(-10 * mask)
    _, l2a, l2b = np.unique(labels2)
    assert np.sum(labels2 != 0) == 23
    assert len(np.unique(labels2)) == 3  # (bg, filled, other)
    assert np.sum(labels2 == l2a) == 8
    assert np.sum(labels2 == l2b) == 15

    sm3 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels3 = sm3.segment_single(-10 * mask)
    assert np.sum(labels3 != 0) == 31
    assert len(np.unique(labels3)) == 4  # (bg, filled, border, other)
    assert np.sum(labels3 == 1) == 8
    assert np.sum(labels3 == 2) == 8
    assert np.sum(labels3 == 3) == 15

    sm4 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": True,
                                                      "closing_disk": 0,
                                                      })
    labels4 = sm4.segment_single(-10 * mask)
    assert np.sum(labels4 != 0) == 40
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 8
    assert np.sum(labels4 == 3) == 23


def test_segm_mpo_labeled_mask_fill_holes_int32():
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
    labels = np.array(sm1.segment_single(-10 * mask), dtype=np.int64)
    # sanity checks
    assert labels.dtype == np.int64
    assert labels.dtype != np.int32
    labels_2 = sm1.process_mask(labels,
                                clear_border=False,
                                fill_holes=True,
                                closing_disk=False)
    assert np.allclose(labels, labels_2)
    assert labels_2.dtype == np.int32


def test_segm_mpo_segment_chunk():
    with segm.segm_thresh.SegmentThresh(
            thresh=-12,
            kwargs_mask={"closing_disk": 0},
            debug=True
    ) as sm:
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


def test_segm_mpo_labeled_mask_clear_border():
    lab0 = np.array([
        [2, 0, 0, 0, 0, 0, 0, 0],  # bad seed position for floodfill
        [0, 2, 0, 0, 0, 0, 0, 0],
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
