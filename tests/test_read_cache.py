from dcnum.write import HDF5Writer
from dcnum.read import HDF5Data, concatenated_hdf5_data

import pytest

from helper_methods import retrieve_data


def test_image_read_cache_min_size():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with HDF5Data(h5path) as hd:
        assert hd.image.chunk_size == 40, "because that is the total size"


def test_image_read_cache_auto_min_size():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with HDF5Data(h5path, image_chunk_size=23) as hd:
        assert hd.image.chunk_size == 40, "because that is the minimum size"


def test_image_read_cache_auto_max_size():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(10 * [h5path], path_out=path):
        pass

    # edit the file with chunks
    with HDF5Writer(path) as hw:
        images = hw.h5["events/image"][:]
        del hw.h5["events/image"]
        hw.store_feature_chunk("image", images)

    with HDF5Data(path, image_chunk_size=30) as hd:
        assert hd.h5["events/image"].chunks == (32, 80, 400)
        assert hd.image.chunk_size == 32, "chunking increased to minimum 32"


def test_image_read_cache_auto_reduced():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(10 * [h5path], path_out=path):
        pass

    # edit the file with chunks
    with HDF5Writer(path) as hw:
        images = hw.h5["events/image"][:]
        del hw.h5["events/image"]
        hw.store_feature_chunk("image", images)

    with HDF5Data(path, image_chunk_size=81) as hd:
        assert hd.h5["events/image"].chunks == (32, 80, 400)
        assert hd.image.chunk_size == 64, "chunking reduced to max below 81"


@pytest.mark.parametrize("chunk_size", (32, 64, 1000))
def test_image_read_cache_chunk_size(chunk_size):
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(10 * [h5path], path_out=path):
        # This creates HDF5 chunks of size 32. Total length is 400.
        # There will be one "remainder" chunk of size `400 % 32 = 16`.
        pass

    data = HDF5Data(path, image_chunk_size=chunk_size)
    chunk_size_act = min(chunk_size, len(data.image))
    assert data.image_chunk_size == chunk_size
    assert data.image.chunk_size == chunk_size_act

    assert data.image.get_chunk_size(0) == chunk_size_act
    if chunk_size_act != len(data.image):
        assert data.image.get_chunk_size(data.image.num_chunks - 1) == 16


def test_image_read_cache_contiguous():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(10 * [h5path], path_out=path):
        pass

    with HDF5Data(path, image_chunk_size=81) as hd:
        assert hd.h5["events/image"].chunks is None
        assert hd.image.chunk_size == 81, "chunking to 81, contiguous arrays"
