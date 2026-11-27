import json
import logging

import h5py
import numpy as np
import pytest

from dcnum import logic, read
from dcnum.read.chopped import ChoppedImageDataset

from helper_methods import extract_data, find_data


def get_dclab_data():
    """Return path to the dclab-chopped test data and reference frames

    The file was created from "fmt-hdf5_cytoshot_full-features_2023.zip"
    with dclab 0.74 (`dclab-compress --crop-event-images` and an
    additionally chopped "mask" feature); the reference frames are the
    frames reconstructed by dclab.
    """
    edest = extract_data("fmt-hdf5_cytoshot_chopped-image-mask_dclab-0.74.zip")
    path = find_data(edest)[0]
    ref = np.load(edest / "reference_dclab.npz")
    return path, ref["image"], ref["mask"]


def write_synthetic_chopped_file(path):
    """Write a file with chopped image data in two datasets

    Returns the true frames that the chopped data must reproduce.
    """
    rng = np.random.default_rng(42)
    roi = (20, 30)
    num = 5
    image_bg = np.asarray(rng.integers(90, 110, size=(num,) + roi),
                          dtype=np.uint8)
    # event geometry: dataset, sub-index, offy, offx, other
    # - events 1 and 2 are in the same frame
    # - event 3 is at the right edge of the frame and its stored image
    #   is wider than the remaining frame
    index = np.array([[0, 0, 2, 3, 0],
                      [1, 0, 10, 2, 2],
                      [0, 1, 1, 15, 1],
                      [1, 1, 12, 25, 3],
                      [0, 2, 5, 9, 4]])
    shapes = {0: (4, 6), 1: (5, 8)}
    frames = np.array(image_bg, copy=True)
    # events 1 and 2 share the same frame (and background)
    image_bg[2] = image_bg[1]
    frames[2] = frames[1]
    stored = {0: np.zeros((3,) + shapes[0], dtype=np.uint8),
              1: np.zeros((2,) + shapes[1], dtype=np.uint8)}
    for ii, (ds, sub, offy, offx, _) in enumerate(index):
        sy, sx = shapes[ds]
        # object inside the event region
        obj = np.zeros((sy, sx), dtype=np.uint8)
        obj[1:-1, 1:-1] = 30 + 10 * ii
        if ii == 3:
            # only 5 columns fit into the frame
            obj[:, 5:] = 0
            obj[0, 6] = 200  # must be ignored
        stored[ds][sub] = obj
        # the true frame (including the siblings)
        target = [ii] if ii not in (1, 2) else [1, 2]
        for jj in target:
            region = frames[jj, offy:offy + sy, offx:offx + sx]
            mask = obj[:region.shape[0], :region.shape[1]] != 0
            region[mask] = (image_bg[jj, offy:offy + sy, offx:offx + sx][mask]
                            + obj[:region.shape[0], :region.shape[1]][mask])

    with h5py.File(path, "w") as h5:
        h5.attrs["imaging:roi size y"] = roi[0]
        h5.attrs["imaging:roi size x"] = roi[1]
        h5.attrs["imaging:pixel size"] = 0.26
        h5.attrs["experiment:event count"] = num
        h5["events/image_bg"] = image_bg
        h5["events/deform"] = np.linspace(0, 0.1, num)
        grp = h5.require_group("basin_events/image")
        grp.attrs["inverse_background_feature"] = "image_bg"
        grp["index"] = index
        for ds, data in stored.items():
            grp[str(ds)] = data
        bn_dict = {"name": "chopped-image",
                   "type": "internal",
                   "format": "h5datasetchop",
                   "paths": ["basin_events"],
                   "features": ["image"],
                   "mapping": "same",
                   }
        h5["basins/chopped"] = [
            ln.encode() for ln in json.dumps(bn_dict, indent=2).split("\n")]
    return frames


def test_chopped_dclab_reference():
    """Frames are reconstructed exactly like dclab does it"""
    path, ref_image, ref_mask = get_dclab_data()
    with read.HDF5Data(path) as hd:
        assert hd.is_chopped("image")
        assert hd.is_chopped("mask")
        assert "image" in hd
        assert np.array_equal(hd.image[:], ref_image)
        assert np.array_equal(hd.mask[:], ref_mask)
        assert hd.mask[0].dtype == bool
        # single events
        for ii in range(len(hd)):
            assert np.array_equal(hd.image[ii], ref_image[ii])
        # background-corrected image
        assert np.array_equal(
            hd.image_corr[:],
            np.asarray(ref_image, dtype=np.int16) - hd.image_bg[:])


def test_chopped_dclab_reference_index_mapping():
    path, ref_image, _ = get_dclab_data()
    with read.HDF5Data(path, index_mapping=slice(3, 30, 2)) as hd:
        assert len(hd) == 14
        assert np.array_equal(hd.image[:], ref_image[3:30:2])
    with read.HDF5Data(path, index_mapping=[5, 20, 21, 33]) as hd:
        assert np.array_equal(hd.image[:], ref_image[[5, 20, 21, 33]])


def test_chopped_synthetic(tmp_path):
    """Several datasets, events sharing a frame, and the frame edge"""
    path = tmp_path / "chopped.rtdc"
    frames = write_synthetic_chopped_file(path)
    with read.HDF5Data(path) as hd:
        assert hd.is_chopped("image")
        assert np.array_equal(hd.image[:], frames)
        assert np.array_equal(hd.image[3], frames[3])
    # direct access with numpy-like indexing
    with h5py.File(path) as h5:
        ds = ChoppedImageDataset(h5group=h5["basin_events/image"],
                                 roi_shape=(20, 30),
                                 inverse_background=h5["events/image_bg"])
        assert ds.shape == (5, 20, 30)
        assert ds.dtype == np.uint8
        assert np.array_equal(ds[[4, 1]], frames[[4, 1]])
        assert np.array_equal(ds[2:4, 10:12], frames[2:4, 10:12])
        assert np.array_equal(ds[3, 12:16, 25:], frames[3, 12:16, 25:])


def test_chopped_missing_inverse_background(tmp_path):
    path = tmp_path / "chopped.rtdc"
    write_synthetic_chopped_file(path)
    with (h5py.File(path) as h5,
          pytest.raises(ValueError, match="require the background")):
        ChoppedImageDataset(h5group=h5["basin_events/image"],
                            roi_shape=(20, 30))


@pytest.mark.parametrize("basin_strategy", ["tap", "drain"])
def test_chopped_pipeline_input(caplog, basin_strategy):
    """The pipeline can process chopped data (with a warning)"""
    path, ref_image, _ = get_dclab_data()
    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path.with_name("out.rtdc"),
                                 basin_strategy=basin_strategy,
                                 debug=True)
    with (caplog.at_level(logging.WARNING, logger="dcnum"),
          logic.DCNumJobRunner(job=job) as runner):
        runner.run()
    assert "The input image data are chopped" in caplog.text

    with read.HDF5Data(job["path_out"]) as hd:
        assert len(hd) > 0
        if basin_strategy == "tap":
            # The output file refers to the chopped input file via a
            # file basin.
            assert "image" not in hd.h5["events"]
            assert hd.is_chopped("image")
        else:
            # The full frames are stored in the output file.
            assert "image" in hd.h5["events"]
            assert not hd.is_chopped("image")
        # The output contains the same frames as the input
        frame_in = read.HDF5Data(path)["frame"]
        idx = np.searchsorted(frame_in, hd["frame"])
        assert np.array_equal(hd.image[:], ref_image[idx])
