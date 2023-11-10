import hashlib
import json

import h5py

from dcnum import write

from helper_methods import retrieve_data


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


def test_basin_file():
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


def test_basin_file_relative():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_test = path.parent / "test.h5"
    # We basically create a file that consists only of the metadata.
    with write.HDF5Writer(path_test) as hw, h5py.File(path) as h5:
        hw.store_basin(name="get-out",
                       paths=[path.name],
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
        assert data_dict["paths"][0] == path.name
        assert data_dict["description"] == "A basin-only dataset"
        assert data_dict["type"] == "file"
        assert data_dict["format"] == "hdf5"
