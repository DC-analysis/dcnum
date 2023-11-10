import h5py
import numpy as np

from dcnum.write import HDF5Writer
from dcnum.read import HDF5Data


from helper_methods import retrieve_data


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
