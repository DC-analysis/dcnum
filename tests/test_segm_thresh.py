import multiprocessing as mp
import pathlib

from dcnum import segm
import h5py
import numpy as np
from skimage import morphology

import pytest

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_segm_thresh_basic():
    """Basic thresholding segmenter

    The segmenter is equivalent to the old dcevent legacy segmenter with
    the options legacy:t=-6^bl=0^bi=0^d=1:cle=1^f=1^clo=3
    (no blur, no binaryops, clear borders, fill holes, closing disk 3).
    Since in the dcevent pipeline, the data are gated and small objects
    are removed, we have to do this here manually before comparing mask
    images.
    """
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    # Get all the relevant information
    with h5py.File(path) as h5:
        image = h5["events/image"][:]
        image_bg = h5["events/image_bg"][:]
        mask = h5["events/mask"][:]
        frame = h5["events/frame"][:]

    # Concatenate the masks
    frame_u, indices = np.unique(frame, return_index=True)
    image_u = image[indices]
    image_bg_u = image_bg[indices]
    mask_u = np.zeros_like(image_u, dtype=bool)
    for ii, fr in enumerate(frame):
        idx = np.where(frame_u == fr)[0]
        mask_u[idx] = np.logical_or(mask_u[idx], mask[ii])

    image_u_c = np.array(image_u, dtype=int) - image_bg_u

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"closing_disk": 3})
    assert sm.requires_background_correction

    for ii in range(len(frame_u)):
        labels_seg = sm.segment_frame(image_u_c[ii])
        mask_seg = np.array(labels_seg, dtype=bool)
        # Remove small objects, because this is not implemented in the
        # segmenter class as it would be part of gating.
        mask_seg = morphology.remove_small_objects(mask_seg, min_size=10)
        assert np.all(mask_seg == mask_u[ii]), f"masks not matching at {ii}"


def test_segm_thresh_get_ppid_from_ppkw():
    segm_kwargs = {"kwargs_mask": {"closing_disk": 3}}
    cls = segm.get_available_segmenters()["thresh"]
    assert cls.get_ppid_from_ppkw(segm_kwargs) == "thresh:t=-6:cle=1^f=1^clo=3"


@pytest.mark.parametrize("worker_type", ["thread", "process"])
def test_segm_thresh_segment_batch(worker_type):
    debug = worker_type == "thread"
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    # Get all the relevant information
    with h5py.File(path) as h5:
        image = h5["events/image"][:]
        image_bg = h5["events/image_bg"][:]
        mask = h5["events/mask"][:]
        frame = h5["events/frame"][:]

    # Concatenate the masks
    frame_u, indices = np.unique(frame, return_index=True)
    image_u = image[indices]
    image_bg_u = image_bg[indices]
    mask_u = np.zeros_like(image_u, dtype=bool)
    for ii, fr in enumerate(frame):
        idx = np.where(frame_u == fr)[0]
        mask_u[idx] = np.logical_or(mask_u[idx], mask[ii])

    image_u_c = np.array(image_u, dtype=int) - image_bg_u

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        debug=debug,
                                        kwargs_mask={"closing_disk": 3})

    labels_seg = sm.segment_batch(image_u_c, start=0, stop=5)
    assert labels_seg is sm.labels_array
    assert np.all(np.array(labels_seg, dtype=bool) == sm.mask_array)
    # tell workers to stop
    sm.join_workers()

    for ii in range(len(frame_u)):
        mask_seg = np.array(labels_seg[ii], dtype=bool)
        # Remove small objects, because this is not implemented in the
        # segmenter class as it would be part of gating.
        mask_seg = morphology.remove_small_objects(mask_seg, min_size=10)
        assert np.all(mask_seg == mask_u[ii]), f"masks not matching at {ii}"


@pytest.mark.parametrize("worker_type", ["thread", "process"])
def test_segm_thresh_segment_batch_large(worker_type):
    debug = worker_type == "thread"

    # Create fake data
    mask = np.zeros((121, 80, 200), dtype=bool)
    mask[:, 10:71, 100:161] = morphology.disk(30).reshape(-1, 61, 61)
    image = -10 * mask

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"closing_disk": 3},
                                        debug=debug)

    labels_seg_1 = np.copy(
        sm.segment_batch(image, start=0, stop=101))

    assert labels_seg_1.dtype == np.uint16  # uint8 is not enough
    assert sm.mp_batch_index.value == 0
    if worker_type == "thread":
        assert len(sm._mp_workers) == 1
        assert sm.mp_batch_worker.value == 1
    else:
        # This will fail if you have too many CPUs in your system
        assert len(sm._mp_workers) == mp.cpu_count()
        # Check whether all processes did their deeds
        assert sm.mp_batch_worker.value == mp.cpu_count()

    labels_seg_2 = np.copy(
        sm.segment_batch(image, start=101, stop=121))

    # tell workers to stop
    sm.join_workers()

    for ii in range(101):
        mask_seg = np.array(labels_seg_1[ii], dtype=bool)
        assert np.all(mask_seg == mask[ii]), f"masks not matching at {ii}"

    for jj in range(101, 121):
        mask_seg = np.array(labels_seg_2[jj - 101], dtype=bool)
        assert np.all(mask_seg == mask[jj]), f"masks not matching at {jj}"
