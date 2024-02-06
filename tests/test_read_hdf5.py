import pathlib
import pickle

import h5py
import numpy as np
import pytest

from dcnum import read, write

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_features_scalar_frame():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    with read.HDF5Data(path) as hd:
        assert "time" in hd
        assert "time" in hd.features_scalar_frame


def test_features_scalar_frame_from_basin():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_2 = path.with_name("basin_based.rtdc")
    write.create_with_basins(path_2, basin_paths=[path])

    with read.HDF5Data(path_2) as hd:
        assert "time" in hd
        assert "time" in hd.features_scalar_frame


def test_features_scalar_frame_from_basin_nested():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_2 = path.with_name("basin_based.rtdc")
    path_basin_nest = path.with_name("basin_nested.rtdc")
    write.create_with_basins(path_out=path_2, basin_paths=[path])
    write.create_with_basins(path_out=path_basin_nest, basin_paths=[path_2])

    with read.HDF5Data(path_basin_nest) as hd:
        assert "time" in hd
        assert "time" in hd.features_scalar_frame


def test_get_ppid():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    with read.HDF5Data(path) as hd:
        assert hd.get_ppid() == "hdf:p=0.2645"

    with read.HDF5Data(path, pixel_size=0.49) as hd:
        assert hd.get_ppid() == "hdf:p=0.49"


def test_get_ppkw_from_ppid_error_bad_code():
    with pytest.raises(ValueError,
                       match="Could not find data method 'peter'"):
        read.HDF5Data.get_ppkw_from_ppid("peter:p=0.44")


def test_get_ppkw_from_ppid_error_bad_parameter():
    with pytest.raises(ValueError,
                       match="Invalid parameter 'k'"):
        read.HDF5Data.get_ppkw_from_ppid("hdf:k=0.44")


def test_get_ppkw_from_ppid_pixel_size():
    ppkw = read.HDF5Data.get_ppkw_from_ppid("hdf:p=0.44")
    assert np.allclose(ppkw["pixel_size"], 0.44)
    assert len(ppkw.keys()) == 1


def test_image_cache(tmp_path):
    path = tmp_path / "test.hdf5"
    with h5py.File(path, "w") as hw:
        hw["events/image"] = np.random.rand(210, 80, 180)

    with h5py.File(path, "r") as h5:
        hic = read.HDF5ImageCache(h5["events/image"],
                                  chunk_size=100,
                                  cache_size=2)

        # Get something from the first chunk
        assert np.allclose(hic[10], h5["events/image"][10])
        assert len(hic.cache) == 1
        assert 0 in hic.cache

        # Get something from the last chunk
        assert np.allclose(hic[205], h5["events/image"][205])
        assert len(hic.cache) == 2
        assert 0 in hic.cache
        assert 2 in hic.cache
        assert np.allclose(hic.cache[2], h5["events/image"][200:])

        # Get something from the first chunk again
        assert np.allclose(hic[90], h5["events/image"][90])
        assert len(hic.cache) == 2
        assert 0 in hic.cache
        assert 2 in hic.cache

        # Get something from the middle chunk
        assert np.allclose(hic[140], h5["events/image"][140])
        assert len(hic.cache) == 2  # limited to two
        assert 0 not in hic.cache  # first item gets removed
        assert 1 in hic.cache
        assert 2 in hic.cache


def test_image_cache_index_out_of_range(tmp_path):
    path = tmp_path / "test.hdf5"
    size = 20
    chunk_size = 8
    with h5py.File(path, "w") as hw:
        hw["events/image"] = np.random.rand(size, 80, 180)
    with h5py.File(path, "r") as h5:
        hic = read.HDF5ImageCache(h5["events/image"],
                                  chunk_size=chunk_size,
                                  cache_size=2)
        # Get something from first chunk. This should just work
        hic.__getitem__(10)
        # Now test out-of-bounds error
        with pytest.raises(IndexError, match="of bounds for HDF5ImageCache"):
            hic.__getitem__(20)


def test_image_chache_get_chunk_size(tmp_path):
    path = tmp_path / "test.hdf5"
    size = 20
    chunk_size = 8
    with h5py.File(path, "w") as hw:
        hw["events/image"] = np.random.rand(size, 80, 180)
    with h5py.File(path, "r") as h5:
        hic = read.HDF5ImageCache(h5["events/image"],
                                  chunk_size=chunk_size,
                                  cache_size=2)
        # Get something from first chunk. This should just work
        assert hic.get_chunk_size(0) == 8
        assert hic.get_chunk_size(1) == 8
        assert hic.get_chunk_size(2) == 4
        with pytest.raises(IndexError, match="only has 3 chunks"):
            hic.get_chunk_size(3)


@pytest.mark.parametrize("size, chunks", [(209, 21),
                                          (210, 21),
                                          (211, 22)])
def test_image_cache_iter_chunks(size, chunks, tmp_path):
    path = tmp_path / "test.hdf5"
    with h5py.File(path, "w") as hw:
        hw["events/image"] = np.random.rand(size, 80, 180)
    with h5py.File(path, "r") as h5:
        hic = read.HDF5ImageCache(h5["events/image"],
                                  chunk_size=10,
                                  cache_size=2)
        assert list(hic.iter_chunks()) == list(range(chunks))


def test_keyerror_when_image_is_none(tmp_path):
    path = tmp_path / "test.hdf5"
    with h5py.File(path, "w") as hw:
        hw["events/deform"] = np.random.rand(100)

    h5dat = read.HDF5Data(path)
    with pytest.raises(KeyError, match="image"):
        _ = h5dat["image"]


def test_meta_nest():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    with read.HDF5Data(path) as hd:
        meta = hd.meta_nest
        print(meta)
        assert meta["imaging"]["pixel size"] == .2645
        assert meta["experiment"]["time"] == '15:24:17'
        assert meta["user"]["dcevent ppid feature"] == 'legacy:b=1^h=1'


def test_pixel_size_getset(tmp_path):
    path = tmp_path / "test.hdf5"
    with h5py.File(path, "w") as hw:
        hw["events/image"] = np.random.rand(10, 80, 180)
        hw.attrs["imaging:pixel size"] = 0.123

    h5dat = read.HDF5Data(path)
    assert np.allclose(h5dat.pixel_size, 0.123)
    h5dat.pixel_size = 0.321
    assert np.allclose(h5dat.pixel_size, 0.321)


def test_open_real_data():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    with read.HDF5Data(path) as h5dat:  # context manager
        # properties
        assert len(h5dat) == 40
        assert h5dat.md5_5m == "599c8c7a112632d007be60b9c37961c5"

        # scalar features
        fsc = h5dat.features_scalar_frame
        # Changed in version 0.17.1: bg_med is not returned anymore,
        # because it is computed from the background image which
        # depends on the background method employed.
        exp = ['frame', 'time']
        assert set(fsc) == set(exp)

        # feature names
        assert len(h5dat.keys()) == 48
        assert "deform" in h5dat.keys()
        assert "deform" in h5dat


def test_pickling_state():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")

    h5d1 = read.HDF5Data(path)
    h5d1.pixel_size = 0.124
    pstate = pickle.dumps(h5d1)
    h5d2 = pickle.loads(pstate)
    assert h5d1.md5_5m == h5d2.md5_5m
    assert h5d1.md5_5m == h5d2.md5_5m
    assert h5d1.pixel_size == h5d2.pixel_size
    assert np.allclose(h5d2.pixel_size, 0.124)
    assert np.all(h5d1.image[0] == h5d2.image[0])
    assert len(h5d1) == 40
    assert len(h5d1) == 40
    # cache size changed from 5 to 2 in dcnum 0.16.3
    assert h5d1.image_cache_size == 2
    assert h5d2.image_cache_size == 2
    assert len(h5d1.basins) == 0
    assert len(h5d2.basins) == 0


def test_pickling_state_logs():
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    h5d1 = read.HDF5Data(path)
    h5d1.pixel_size = 0.124
    pstate = pickle.dumps(h5d1)
    h5d2 = pickle.loads(pstate)
    assert h5d1.logs
    for lk in h5d1.logs:
        assert h5d1.logs[lk] == h5d2.logs[lk]


def test_pickling_state_tables():
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # The original file does not contain any tables, so we write
    # generate a table
    columns = ["alot", "of", "tables"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [float] * len(columns)})
    tab_data = np.zeros((11, len(columns)))
    tab_data[:, 0] = np.arange(11)
    tab_data[:, 1] = 1000
    tab_data[:, 2] = np.linspace(1, np.sqrt(2), 11)
    rec_arr = np.rec.array(tab_data, dtype=ds_dt)

    # add table to source file
    with h5py.File(path, "a") as h5:
        h5tab = h5.require_group("tables")
        h5tab.create_dataset(name="sample_table",
                             data=rec_arr)

    h5d1 = read.HDF5Data(path)
    h5d1.pixel_size = 0.124
    pstate = pickle.dumps(h5d1)
    h5d2 = pickle.loads(pstate)
    assert h5d1.tables
    table = h5d1.tables["sample_table"]
    assert len(table) == 3
    for lk in table:
        assert np.allclose(h5d1.tables["sample_table"][lk],
                           h5d2.tables["sample_table"][lk])


def test_read_empty_logs():
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    with h5py.File(path, "a") as h5:
        h5.require_group("logs").create_dataset(name="empty_log",
                                                data=[])
    h5r = read.HDF5Data(path)
    assert "empty_log" not in h5r.logs


def test_read_zero_size():
    path = retrieve_data(data_path / "fmt-hdf5_shapein_empty.zip")
    with read.HDF5Data(path) as hd:
        assert len(hd) == 0
        with pytest.warns(read.cache.EmptyDatasetWarning,
                          match=f"{path.name} has zero length"):
            assert len(hd.image) == 0


def test_table_with_length_one():
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # The original file does not contain any tables, so we write
    # generate a table
    columns = ["alot", "of", "tables"]
    ds_dt = np.dtype({'names': columns,
                      'formats': [float] * len(columns)})
    tab_data = np.zeros((1, len(columns)))
    tab_data[:, 0] = np.arange(1)
    tab_data[:, 1] = 1000
    tab_data[:, 2] = 3
    rec_arr = np.rec.array(tab_data, dtype=ds_dt)

    # add table to source file
    with h5py.File(path, "a") as h5:
        h5tab = h5.require_group("tables")
        h5tab.create_dataset(name="sample_table",
                             data=rec_arr)

    h5d1 = read.HDF5Data(path)
    h5d1.pixel_size = 0.124
    assert h5d1.tables
    table = h5d1.tables["sample_table"]
    assert len(table) == 3
    assert h5d1.tables["sample_table"]["alot"].shape == (1,)
