import hashlib
import json

import h5py
import numpy as np
import pytest

from dcnum import read, write
from dcnum import __version__ as version

from helper_methods import retrieve_data


def test_copy_metadata_empty_log_variable_length_string():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    # Add an empty log file
    with h5py.File(path, "a") as h5:
        data = np.asarray([], dtype=h5py.string_dtype(length=None))
        h5["logs"].create_dataset(name="empty_log_entry",
                                  data=data)

    path_wrt = path.with_name("written.hdf5")
    with h5py.File(path, "r") as h5_src, h5py.File(path_wrt, "w") as h5_dst:
        write.copy_metadata(h5_src=h5_src, h5_dst=h5_dst)
        assert "empty_log_entry" not in h5_dst["logs"]


def test_copy_metadata_logs():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    with h5py.File(path, "r") as h5_src, h5py.File(path_wrt, "w") as h5_dst:
        write.copy_metadata(h5_src=h5_src, h5_dst=h5_dst)
        # Make sure that worked
        assert dict(h5_src.attrs) == dict(h5_dst.attrs)
        assert len(h5_src["logs"]) == 2
        for key in h5_src["logs"]:
            assert np.all(h5_src[f"logs/{key}"][:] == h5_dst[f"logs/{key}"][:])
            assert h5_dst[f"logs/{key}"].attrs["software"] \
                   == f"dcnum {version}"
        assert h5_dst["logs/dcevent-analyze"][0] == b"{"
        assert h5_dst["logs/dcevent-analyze"][1] == b'  "dcevent": {'


def test_copy_metadata_logs_variable_length():
    """
    Old versions of dclab and Shape-In store the logs as variable-length
    string. This does not support the fletcher32 filter.
    """
    path = retrieve_data(
        "fmt-hdf5_shapein_raw-with-variable-length-logs.zip")
    path_wrt = path.with_name("written.hdf5")
    with h5py.File(path, "r") as h5_src, h5py.File(path_wrt, "w") as h5_dst:
        write.copy_metadata(h5_src=h5_src, h5_dst=h5_dst)
        # Make sure that worked
        assert dict(h5_src.attrs) == dict(h5_dst.attrs)
        assert len(h5_src["logs"]) == 2
        for key in h5_src["logs"]:
            assert np.all(h5_src[f"logs/{key}"][:] == h5_dst[f"logs/{key}"][:])
            assert h5_dst[f"logs/{key}"].attrs["software"] \
                   == f"dcnum {version}"
        assert h5_dst["logs/M1_para.ini"][0] == b"[General]"
        assert h5_dst["logs/M1_camera.ini"][1] == b"Shutter Time = 20"


def test_copy_metadata_without_basins():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[path.resolve()])
    path_test = path.with_name("test.hdf5")

    with h5py.File(path_wrt) as hin, h5py.File(path_test, "a") as h5:
        assert "basins" in hin
        assert "basins" not in h5
        write.copy_metadata(h5_src=hin, h5_dst=h5, copy_basins=False)
        assert "basins" not in h5
        write.copy_metadata(h5_src=hin, h5_dst=h5)  # defaults to True
        assert "basins" in h5


def test_create_with_basins_absolute():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[path.resolve()])
    with h5py.File(path_wrt) as h5:
        assert h5.attrs["setup:software version"].count("CytoShot 0.0.6")
        assert h5.attrs["experiment:event count"] == 11
        assert "basins" in h5
        assert len(h5["events"].keys()) == 0
    with read.HDF5Data(path_wrt) as hd:
        assert len(hd.basins) == 1
        assert len(hd.basins[0]["features"]) == 48
        assert len(hd.basins[0]["paths"]) == 1
        assert np.allclose(hd["deform"][0], 0.07405636775888857)


def test_create_with_basins_invalid_file():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[["fake.rtdc"
                                                              ]])
    with h5py.File(path_wrt) as h5:
        assert not h5.attrs
        assert "basins" in h5
        assert len(h5["events"].keys()) == 0
    with read.HDF5Data(path_wrt) as hd:
        assert len(hd.basins) == 1
        assert "features" not in hd.basins[0]
        assert len(hd.basins[0]["paths"]) == 1
        with pytest.raises(KeyError, match="'deform' not found"):
            assert hd["deform"][0]


def test_create_with_basins_invalid_and_relative_file():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[["fake.rtdc",
                                                              path.name,
                                                              ]])
    with h5py.File(path_wrt) as h5:
        assert h5.attrs["setup:software version"].count("CytoShot 0.0.6")
        assert h5.attrs["experiment:event count"] == 11
        assert "basins" in h5
        assert len(h5["events"].keys()) == 0
    with read.HDF5Data(path_wrt) as hd:
        assert len(hd.basins) == 1
        assert len(hd.basins[0]["features"]) == 48
        assert len(hd.basins[0]["paths"]) == 2
        assert np.allclose(hd["deform"][0], 0.07405636775888857)


def test_create_with_basins_relative():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[path.name])
    with h5py.File(path_wrt) as h5:
        assert h5.attrs["setup:software version"].count("CytoShot 0.0.6")
        assert h5.attrs["experiment:event count"] == 11
        assert "basins" in h5
        assert len(h5["events"].keys()) == 0
    with read.HDF5Data(path_wrt) as hd:
        assert len(hd.basins) == 1
        assert len(hd.basins[0]["features"]) == 48
        assert len(hd.basins[0]["paths"]) == 1
        assert np.allclose(hd["deform"][0], 0.07405636775888857)


def test_create_with_basins_relative_and_absolute():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write.create_with_basins(path_out=path_wrt, basin_paths=[[path.name,
                                                              path.resolve()]])
    with h5py.File(path_wrt) as h5:
        assert h5.attrs["setup:software version"].count("CytoShot 0.0.6")
        assert h5.attrs["experiment:event count"] == 11
        assert "basins" in h5
        assert len(h5["events"].keys()) == 0
    with read.HDF5Data(path_wrt) as hd:
        assert len(hd.basins) == 1
        assert len(hd.basins[0]["features"]) == 48
        assert len(hd.basins[0]["paths"]) == 2
        assert np.allclose(hd["deform"][0], 0.07405636775888857)


def test_writer_basic():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    with h5py.File(path) as h5, write.HDF5Writer(path_wrt) as hw:
        deform = h5["events"]["deform"][:]
        image = h5["events"]["image"][:]

        hw.store_feature_chunk(feat="deform", data=deform)
        hw.store_feature_chunk(feat="deform", data=deform)
        hw.store_feature_chunk(feat="deform", data=deform[:10])

        hw.store_feature_chunk(feat="image", data=image)
        hw.store_feature_chunk(feat="image", data=image)
        hw.store_feature_chunk(feat="image", data=image[:10])

    with h5py.File(path_wrt) as ho:
        events = ho["events"]
        size = deform.shape[0]
        assert events["deform"].shape[0] == 2*size + 10
        assert events["image"].shape[0] == 2 * size + 10
        assert events["image"].shape[1:] == image.shape[1:]


def test_writer_basin_file():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with write.HDF5Writer(path_test) as hw, h5py.File(path) as h5:
        hw.store_basin(name="get-out",
                       paths=[path],
                       description="A basin-only dataset",
                       )
        hw.h5.attrs.update(h5.attrs)

    # OK, now open the dataset and make sure that it contains all information.
    with h5py.File(path_test) as h5:
        assert "basins" in h5
        key = list(h5["basins"].keys())[0]
        data = "\n".join([s.decode() for s in h5["basins"][key][:].tolist()])
        data_hash = hashlib.md5(data.encode("utf-8",
                                            errors="ignore")).hexdigest()
        assert key == data_hash
        data_dict = json.loads(data)
        assert data_dict["name"] == "get-out"
        assert data_dict["paths"][0] == str(path)
        assert data_dict["description"] == "A basin-only dataset"
        assert data_dict["type"] == "file"
        assert data_dict["format"] == "hdf5"
        assert "features" not in data_dict


def test_writer_basin_file_relative():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with write.HDF5Writer(path_test) as hw, h5py.File(path) as h5:
        hw.store_basin(name="get-out",
                       paths=[path.name],
                       description="A basin-only dataset",
                       features=["deform", "area_um"],
                       )
        hw.h5.attrs.update(h5.attrs)

    # OK, now open the dataset and make sure that it contains all information.
    with h5py.File(path_test) as h5:
        assert "basins" in h5
        key = list(h5["basins"].keys())[0]
        data = "\n".join([s.decode() for s in h5["basins"][key][:].tolist()])
        data_hash = hashlib.md5(data.encode("utf-8",
                                            errors="ignore")).hexdigest()
        assert key == data_hash
        data_dict = json.loads(data)
        assert data_dict["name"] == "get-out"
        assert data_dict["paths"][0] == path.name
        assert data_dict["description"] == "A basin-only dataset"
        assert data_dict["type"] == "file"
        assert data_dict["format"] == "hdf5"
        assert data_dict["features"] == ["deform", "area_um"]


def test_writer_logs(tmp_path):
    path_test = tmp_path / "test.h5"
    # We basically create a file that consists only of the metadata.
    with write.HDF5Writer(path_test) as hw:
        hw.store_log("peter", ["McNulty", "Freamon", "Omar"])

    with read.HDF5Data(path_test) as hd:
        assert hd.logs["peter"] == ["McNulty", "Freamon", "Omar"]


def test_writer_logs_override(tmp_path):
    path_test = tmp_path / "test.h5"
    # We basically create a file that consists only of the metadata.
    with write.HDF5Writer(path_test) as hw:
        hw.store_log("peter", ["McNulty", "Freamon", "Omar"])

    with read.HDF5Data(path_test) as hd:
        assert hd.logs["peter"] == ["McNulty", "Freamon", "Omar"]

    with write.HDF5Writer(path_test) as hw:
        with pytest.raises(ValueError, match="peter"):
            hw.store_log("peter", ["Omar", "McNulty", "Freamon"])
        hw.store_log("peter", ["Omar", "McNulty", "Freamon"],
                     override=True)

    with read.HDF5Data(path_test) as hd:
        assert hd.logs["peter"] == ["Omar", "McNulty", "Freamon"]
