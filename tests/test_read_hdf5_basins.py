import pathlib

import h5py
import numpy as np
import pytest

from dcnum import read, write

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_basin_features_path_absolute():
    """Create a dataset that refers to a basin in an absolute path"""
    path_src = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path = path_src.with_name("input.rtdc")

    # Dataset creation
    with h5py.File(path_src) as src, h5py.File(path, "w") as dst:
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=dst)
        # store the basin information in the new dataset
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin",
                       paths=[path_src],
                       )
        # sanity checks
        assert "image" in src["events"]
        assert "image" not in dst["events"]

    # Now open the basin-based dataset and check whether basins are defined
    with read.HDF5Data(path) as hd:
        assert "image" in hd
        assert np.median(hd["image"][0]) == 187
        assert np.median(hd["image"][4]) == 186


def test_basin_features_path_absolute_mapped():
    """Create a dataset that refers to a mapped basin in an absolute path"""
    path_src = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path = path_src.with_name("input.rtdc")

    # Dataset creation
    with h5py.File(path_src) as src, h5py.File(path, "w") as dst:
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=dst)
        dst.attrs["experiment:event count"] = 6
        # store the basin information in the new dataset
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin",
                       paths=[path_src],
                       mapping=np.arange(4, 10),
                       )
        # sanity checks
        assert "image" in src["events"]
        assert "image" not in dst["events"]

    # Now open the basin-based dataset and check whether basins are defined
    with read.HDF5Data(path) as hd, read.HDF5Data(path_src) as hd_src:
        # sanity check for basin
        assert len(hd_src) == 11
        assert "image" in hd_src
        assert len(hd_src["image"]) == 11
        # basin-based dataset
        assert len(hd) == 6
        assert "image" in hd
        assert len(hd["image"]) == 6

        # Note the difference to the above test
        assert np.median(hd["image"][0]) == 186
        assert np.allclose(hd["image"][:], hd_src["image"][4:10])


def test_basin_mapped_with_mapped_dataset():
    """You can have a mapped dataset opening a mapped basin"""
    path_src = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path = path_src.with_name("input.rtdc")
    mapping = np.arange(1, 9)

    # Dataset creation
    with h5py.File(path_src) as src, h5py.File(path, "w") as dst:
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=dst,
                            )
        dst.attrs["experiment:event count"] = 6
        # store the basin information in the new dataset
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin",
                       paths=[path_src],
                       mapping=mapping,
                       )
        # sanity checks
        assert "image" in src["events"]
        assert "image" not in dst["events"]
        image = src["events/image"][:]
        deform = src["events/deform"][:]

    indexing_array = mapping[slice(2, 5)]

    with read.HDF5Data(path, index_mapping=slice(2, 5)) as hd:
        assert np.all(hd["image"][:] == image[indexing_array])
        assert np.all(hd["deform"][:] == deform[indexing_array])


@pytest.mark.parametrize("mapping,numevents", [
    [None, 11],
    [np.arange(3, dtype=int), 3],
    [np.arange(1, 5, 2, dtype=int), 2],
])
def test_basin_multiple(mapping, numevents):
    """Check that defining multiple basins in one file works
    """
    path_src = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    path_b1 = path_src.with_name("basin1.rtdc")
    path_b2 = path_src.with_name("basin2.rtdc")
    path = path_src.with_name("output.rtdc")

    # create basin files
    with (h5py.File(path_src) as src,
          h5py.File(path_b1, "w") as bn1,
          h5py.File(path_b2, "w") as bn2):
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=bn1,
                            )
        # store the basin information in the new dataset
        hw1 = write.HDF5Writer(bn1)
        # also store feature information
        hw1.store_feature_chunk("deform", src["events/deform"][:])

        write.copy_metadata(h5_src=src,
                            h5_dst=bn2,
                            )
        hw2 = write.HDF5Writer(bn2)
        # also store feature information
        hw2.store_feature_chunk("aspect", src["events/aspect"][:])

        # sanity checks
        assert "deform" in bn1["events"]
        assert "aspect" in bn2["events"]
        assert len(bn1["events/deform"]) == 11

    # create basin-based dataset
    with h5py.File(path_src) as src, h5py.File(path, "w") as dst:
        write.copy_metadata(h5_src=src,
                            h5_dst=dst,
                            )
        dst.attrs["experiment:event count"] = numevents
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin 1",
                       paths=[path_b1],
                       mapping=mapping,
                       )
        hw.store_basin(name="test basin 2",
                       paths=[path_b2],
                       mapping=mapping,
                       )
        # sanity check
        if mapping is not None:
            assert "basinmap0" in dst["events"]
            assert "basinmap1" not in dst["events"]  # (b/c same mapping)

    # Checks for level 2
    with read.HDF5Data(path_src) as hd0, read.HDF5Data(path) as hd:
        if mapping is not None:
            assert np.all(hd["basinmap0"] == mapping)
        # We only have one basin.
        assert len(hd.basins) == 2
        # Deformation is stored in the basin file, so we can access it.
        assert np.all(hd["deform"][:] == hd0["deform"][mapping][:])
        assert np.all(hd["aspect"][:] == hd0["aspect"][mapping][:])
        # We did not explicitly define features above.
        assert "features" not in hd.basins[0]
        # basins are sorted according to name
        bn_group1, bn_feats1, _ = hd.get_basin_data(0)
        assert "deform" in bn_feats1
        assert "deform" in bn_group1
        bn_group2, bn_feats2, _ = hd.get_basin_data(1)
        assert "aspect" in bn_feats2
        assert "aspect" in bn_group2
        # The "aspect" feature is in level1, so we can access it.
        assert "aspect" in hd
        # The "image" and "area_um" features are not in level1, theay are
        # only defined there as a basin. But we don't do basin nesting
        # in dcnum, so they should not be accessible.
        assert "image" not in hd
        assert "area_um" not in hd


def test_basin_no_inception():
    """We do not support basin inception in dcnum

    With basin inception, I mean opening basins that are defined
    in basins.
    """
    path_src = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    path_l1 = path_src.with_name("level1.rtdc")
    basin_map1 = np.array([1, 7, 10], dtype=np.uint64)
    path_l2 = path_src.with_name("level2.rtdc")
    basin_map2 = np.array([1, 2], dtype=np.uint64)

    # level 1
    with h5py.File(path_src) as src, h5py.File(path_l1, "w") as dst:
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=dst,
                            )
        dst.attrs["experiment:event count"] = 3
        # store the basin information in the new dataset
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin",
                       paths=[path_src],
                       mapping=basin_map1,
                       features=["area_um", "image"]
                       )
        # also store feature information
        hw.store_feature_chunk("deform", src["events/deform"][basin_map1])
        hw.store_feature_chunk("aspect", src["events/aspect"][basin_map1])
        # sanity checks
        assert "image" in src["events"]
        assert "image" not in dst["events"]
        assert "deform" in dst["events"]

    # Checks for level 1
    with read.HDF5Data(path_src) as hd0, read.HDF5Data(path_l1) as hd1:
        assert np.all(hd1["basinmap0"] == basin_map1)
        assert len(hd1.basins) == 1
        assert "mapping" in str(hd1.basins[0])
        assert "area_um" in hd1.basins[0]["features"]
        assert "image" in hd1.basins[0]["features"]
        assert np.all(hd1["area_um"][:] == hd0["area_um"][basin_map1])
        assert np.all(hd1["image"][:] == hd0["image"][basin_map1])
        assert "aspect" in hd1.h5["events"]

    # level 2
    with h5py.File(path_src) as src, h5py.File(path_l2, "w") as dst:
        # first, copy all the scalar features to the new file
        write.copy_metadata(h5_src=src,
                            h5_dst=dst,
                            )
        dst.attrs["experiment:event count"] = 2
        # store the basin information in the new dataset
        hw = write.HDF5Writer(dst)
        hw.store_basin(name="test basin",
                       paths=[path_l1],
                       mapping=basin_map2,
                       )
        # also store feature information
        hw.store_feature_chunk("deform",
                               src["events/deform"][basin_map1[basin_map2]])
        # sanity checks
        assert "deform" in dst["events"]

    # Checks for level 2
    with read.HDF5Data(path_src) as hd0, read.HDF5Data(path_l2) as hd2:
        # Since we don't do feature nesting, it is fine to have the
        # basinmap0 feature redefined here.
        assert np.all(hd2["basinmap0"] == basin_map2)
        # We only have one basin.
        assert len(hd2.basins) == 1
        # Deformation is stored in the basin file, so we can access it.
        assert np.all(hd2["deform"] == hd0["deform"][basin_map1][basin_map2])
        # We did not explicitly define features above.
        assert "features" not in hd2.basins[0]
        # The "aspect" feature is in level1, so we can access it.
        assert "aspect" in hd2
        # The "image" and "area_um" features are not in level1, theay are
        # only defined there as a basin. But we don't do basin nesting
        # in dcnum, so they should not be accessible.
        assert "image" not in hd2
        assert "area_um" not in hd2
