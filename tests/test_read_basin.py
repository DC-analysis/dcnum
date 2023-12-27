import h5py
import numpy as np

from dcnum.write import HDF5Writer, create_with_basins
from dcnum.read import HDF5Data

from helper_methods import retrieve_data


def test_basin_features():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_basin = h5path.with_name("basin.rtdc")
    create_with_basins(path_out=h5path_basin, basin_paths=[h5path])
    with HDF5Data(h5path_basin) as hd:
        assert len(hd.basins) == 1
        h5dat, features = hd.get_basin_data(0)
        assert "time" in h5dat.h5["events"]
        assert "time" in features
        assert "time" in h5dat


def test_basin_features_nested():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_basin = h5path.with_name("basin.rtdc")
    h5path_basin_nest = h5path.with_name("basin_nested.rtdc")
    create_with_basins(path_out=h5path_basin, basin_paths=[h5path])
    create_with_basins(path_out=h5path_basin_nest, basin_paths=[h5path_basin])
    with HDF5Data(h5path_basin_nest) as hd:
        assert len(hd.basins) == 2
        assert "time" in hd
        for ii, bn_dict in enumerate(hd.basins):
            h5dat, features = hd.get_basin_data(ii)
            if str(bn_dict["paths"][0]) == str(h5path):
                # First basin-based dataset, should contain features.
                assert "time" in features
                assert "time" in h5dat
            else:
                # Nested basin, only contains basin, no features available.
                assert bn_dict["paths"][0] == str(h5path_basin)
                assert "time" not in features
                assert "time" in h5dat


def test_basin_not_available():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, HDF5Writer(h5path_small, "w") as hw:
        dst = hw.h5
        dst.require_group("events")
        # first, copy all the scalar features to the new file
        for feat in src["events"]:
            if feat not in ["image", "image_bg", "mask"]:
                dst["events"][feat] = src["events"][feat][:]
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=["fake.rtdc",  # fake path
                              str(h5path),  # absolute path name
                              ])
        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]

    h5path.unlink()

    # Now open the scalar dataset and check whether basins missing
    with HDF5Data(h5path_small) as hd:
        assert "image" not in hd
        assert hd.image is None
        assert hd.image_bg is None
        assert hd.image_corr is None
        assert hd.mask is None
        _, features = hd.get_basin_data(0)
        assert not features


def test_basin_nothing_available():
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, HDF5Writer(h5path_small, "w") as hw:
        dst = hw.h5
        # first, copy all the scalar features to the new file
        for feat in src["events"]:
            if feat not in ["image", "image_bg", "mask"]:
                dst["events"][feat] = src["events"][feat][:]
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=["fake.rtdc",  # fake path
                              ])

        # sanity checks
        assert "deform" in dst["events"]
        assert "image" not in dst["events"]

    h5path.unlink()

    # Now open the scalar dataset and check whether basins missing
    with HDF5Data(h5path_small) as hd:
        assert "image" not in hd
        _, features = hd.get_basin_data(0)


def test_basin_path_absolute():
    """Create a dataset that refers to a basin in an absolute path"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, HDF5Writer(h5path_small, "w") as hw:
        dst = hw.h5
        # first, copy all the scalar features to the new file
        for feat in src["events"]:
            if feat not in ["image", "image_bg", "mask"]:
                dst["events"][feat] = src["events"][feat][:]
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=["fake.rtdc",  # fake path
                              str(h5path.resolve())
                              ])

    # Now open the scalar dataset and check whether basins are defined
    with HDF5Data(h5path_small) as hd:
        assert "image" in hd.get_basin_data(0)[1]
        assert "image" in hd.keys()
        assert np.median(hd["image"][0]) == 187


def test_basin_relative():
    """Create a dataset that refers to a basin in a relative path"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, HDF5Writer(h5path_small, "w") as hw:
        dst = hw.h5
        # first, copy all the scalar features to the new file
        for feat in src["events"]:
            if feat not in ["image", "image_bg", "mask"]:
                dst["events"][feat] = src["events"][feat][:]
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=["fake.rtdc",  # fake path
                              h5path.name
                              ])

    # Now open the scalar dataset and check whether basins are defined
    with HDF5Data(h5path_small) as hd:
        assert "image" in hd.get_basin_data(0)[1]
        assert "image" in hd.keys()
        assert np.median(hd["image"][0]) == 187
        assert np.median(hd.image[0]) == 187
        assert np.median(hd.image_corr[0]) == 1


def test_basin_scalar_features():
    """Create a dataset that refers to a basin in a relative path"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    h5path_small = h5path.with_name("smaller.rtdc")

    # Dataset creation
    with h5py.File(h5path) as src, HDF5Writer(h5path_small, "w") as hw:
        dst = hw.h5
        # only copy one feature
        dst["events"]["deform"] = src["events"]["deform"][:]
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=["fake.rtdc",  # fake path
                              h5path.name
                              ])

    # Now open the scalar dataset and check whether basins are defined
    with HDF5Data(h5path_small) as hd:
        assert "image" in hd.get_basin_data(0)[1]
        assert "image" in hd.keys()
        assert "area_um" in hd.keys()
        assert "deform" in hd.keys()
        assert np.median(hd["image"][0]) == 187
        assert np.median(hd.image[0]) == 187
        assert np.median(hd.image_corr[0]) == 1
        assert np.allclose(hd["deform"][0], 0.0740563677588885)
        assert np.allclose(hd["area_um"][0], 0.559682)
        assert np.allclose(hd["area_um"][1], 91.193185875)


def test_basin_self_reference():
    """Paths can self-reference in basins, no recursion errors"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")

    # Dataset creation
    with HDF5Writer(h5path, "a") as hw:
        # Next, store the basin information in the new dataset
        hw.store_basin(name="test",
                       paths=[h5path])

    # Now open the scalar dataset and check whether basins are defined
    with HDF5Data(h5path) as hd:
        assert "image" in hd.get_basin_data(0)[1]
        assert "image" in hd.keys()
        assert np.median(hd["image"][0]) == 187
        assert np.median(hd.image[0]) == 187
        assert np.median(hd.image_corr[0]) == 1
        assert np.allclose(np.mean(hd["deform"]),
                           0.23354564471483724,
                           atol=0, rtol=1e-7)
