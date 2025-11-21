import h5py

from dcnum.write import HDF5Writer
from dcnum.read import concatenated_hdf5_data
from dcnum.read.cache import HDF5ImageCache

from helper_methods import retrieve_data


def test_image_read_cache_min_size():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with h5py.File(h5path) as h5:
        im = HDF5ImageCache(h5["events/image"])
        assert im.chunk_size == 40, "because that is the total size"


def test_image_read_cache_auto_min_size():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with h5py.File(h5path) as h5:
        im = HDF5ImageCache(h5["events/image"],
                            chunk_size=23)
        assert im.chunk_size == 40, "because that is the minimum size"


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

    with h5py.File(path) as h5:
        assert h5["events/image"].chunks == (32, 80, 400)
        im = HDF5ImageCache(h5["events/image"],
                            chunk_size=30)
        assert im.chunk_size == 32, "chunking increased to minimum 32"


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

    with h5py.File(path) as h5:
        assert h5["events/image"].chunks == (32, 80, 400)
        im = HDF5ImageCache(h5["events/image"],
                            chunk_size=81)
        assert im.chunk_size == 64, "chunking reduced to maximum below 81"


def test_image_read_cache_contiguous():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(10 * [h5path], path_out=path):
        pass

    with h5py.File(path) as h5:
        assert h5["events/image"].chunks is None
        im = HDF5ImageCache(h5["events/image"],
                            chunk_size=81)
        assert im.chunk_size == 81, "chunking as requested, contiguous arrays"
