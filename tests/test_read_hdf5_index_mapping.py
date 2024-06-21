import pathlib

import h5py
import numpy as np
import pytest

from dcnum import read, write

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("index_mapping,np_slice", [
    (11, slice(None, None, None)),
    (5, slice(None, 5, None)),
    (slice(0, 5), slice(None, 5, None)),
    (slice(1, 5), slice(1, 5, None)),
    (slice(3, 6), slice(3, 6, None)),
    ([1, 4, 6], [1, 4, 6]),
    # repetitions supported in dcnum 0.24.0
    ([1, 1, 4, 6], [1, 1, 4, 6]),
])
def test_features_scalar_frame(index_mapping, np_slice):
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    with h5py.File(path) as h5:
        tdata = h5["/events/time"][:][np_slice]

    with read.HDF5Data(path, index_mapping=index_mapping) as hd:
        assert "time" in hd
        assert "time" in hd.features_scalar_frame
        assert len(hd) == len(tdata)
        assert len(hd["time"]) == len(tdata)
        assert np.all(tdata == hd["time"])


@pytest.mark.parametrize("index_mapping,np_slice", [
    (11, slice(None, None, None)),
    (5, slice(None, 5, None)),
    (slice(0, 5), slice(None, 5, None)),
    (slice(1, 5), slice(1, 5, None)),
    (slice(3, 6), slice(3, 6, None)),
    ([1, 4, 6], [1, 4, 6]),
    # repetitions supported in dcnum 0.24.0
    ([1, 1, 4, 6], [1, 1, 4, 6]),
])
def test_features_scalar_frame_from_basin(index_mapping, np_slice):
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_2 = path.with_name("basin_based.rtdc")
    write.create_with_basins(path_2, basin_paths=[path])

    # Get the first index in a very complicated way
    start = np_slice.start or 0 if isinstance(np_slice, slice) else np_slice[0]

    with h5py.File(path) as h5:
        tdata = h5["/events/time"][:][np_slice]
        tfull = h5["/events/time"][:]
        image0 = h5["/events/image"][:][start]

    with read.HDF5Data(path_2, index_mapping=index_mapping) as hd:
        assert "time" not in hd.h5["events"], "sanity checkts"
        assert "time" in hd
        assert "time" in hd.features_scalar_frame
        assert len(hd) == len(tdata)
        assert len(hd["time"]) == len(tdata)
        assert np.all(tdata == hd["time"])
        assert tfull[start] == hd["time"][0]
        assert np.all(image0 == hd["image"][0])
        assert np.all(image0 == hd.image[0])


@pytest.mark.parametrize("index_mapping,np_slice", [
    (11, slice(None, None, None)),
    (5, slice(None, 5, None)),
    (slice(0, 5), slice(None, 5, None)),
    (slice(1, 5), slice(1, 5, None)),
    (slice(3, 6), slice(3, 6, None)),
    ([1, 4, 6], [1, 4, 6]),
    # repetitions supported in dcnum 0.24.0
    ([1, 1, 4, 6], [1, 1, 4, 6]),
])
def test_features_scalar_frame_from_basin_nested(index_mapping, np_slice):
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_2 = path.with_name("basin_based.rtdc")
    path_basin_nest = path.with_name("basin_nested.rtdc")
    write.create_with_basins(path_out=path_2, basin_paths=[path])
    write.create_with_basins(path_out=path_basin_nest, basin_paths=[path_2])

    with h5py.File(path) as h5:
        tdata = h5["/events/time"][:][np_slice]

    with read.HDF5Data(path_basin_nest, index_mapping=index_mapping) as hd:
        assert "time" in hd
        assert "time" in hd.features_scalar_frame
        assert len(hd) == len(tdata)
        assert len(hd["time"]) == len(tdata)
        assert np.all(tdata == hd["time"])
