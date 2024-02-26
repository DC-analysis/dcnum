import pathlib

from dcnum import read
import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("path_out", [None, True])
def test_concat_basic(path_out):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    assert len(data) == 120


@pytest.mark.parametrize("path_out", [None, True])
def test_concat_basic_frame(path_out):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    with h5py.File(path) as h5:
        frame = h5["events/frame"][:]
    assert frame[0] == 101

    assert np.allclose(data["frame"][:frame.size],
                       frame - 101 + 1)
    offset1 = frame[-1] - 101 + 1
    assert np.allclose(offset1, data["frame"][frame.size-1])
    assert np.allclose(offset1 + 1, data["frame"][frame.size])
    assert np.allclose(data["frame"][frame.size:2*frame.size],
                       frame - 101 + offset1 + 1)
    diff = frame[-1] - frame[0]
    assert np.allclose(data["frame"][-1], 3 * (diff + 1))


def test_concat_basic_to_file(tmp_path):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    path_out = tmp_path / "test.rtdc"
    assert not path_out.exists()
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    assert len(data) == 120
    assert path_out.exists()


def test_concat_ignore_contour(tmp_path):
    # get file wtih contour information
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    cdata = np.array([[2, 2], [2, 3], [2, 5], [4, 5], [4, 2]])
    with h5py.File(path, mode="a") as h5:
        contour = h5["events"].create_group("contour")

        for ii in range(40):
            contour.create_dataset(name=f"{ii}", data=cdata, dtype=np.uint64)

    path_out = tmp_path / "test.rtdc"
    with pytest.warns(UserWarning,
                      match="Ignoring contour; not implemented yet!"):
        data = read.concatenated_hdf5_data([path, path, path],
                                           path_out=path_out)

    assert "deform" in data
    assert "conotur" not in data


def test_concat_invalid_input_feature_number():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    path2 = retrieve_data(data_path /
                          "fmt-hdf5_cytoshot_full-features_2023.zip")

    # Add a new feature to path2
    with h5py.File(path2, mode="a") as h5:
        deform = h5["/events/deform"][:]
        h5["events/userdef1"] = deform * 10

    with pytest.raises(ValueError, match="contains more features"):
        read.concatenated_hdf5_data([path2, path])


def test_concat_invalid_input_path():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    invalid_output = 42
    with pytest.raises(ValueError, match="Invalid type"):
        read.concatenated_hdf5_data([path, path, path],
                                    path_out=invalid_output)


def test_concat_invalid_input_path_number():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    with pytest.raises(ValueError, match="Please specify at least two"):
        read.concatenated_hdf5_data([path])


def test_concat_specify_input_feature_number():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    path2 = retrieve_data(data_path /
                          "fmt-hdf5_cytoshot_full-features_2023.zip")

    # Add a new feature to path2
    with h5py.File(path2, mode="a") as h5:
        deform = h5["/events/deform"][:]
        h5["events/userdef1"] = deform * 10

    path_out = path.with_name("output.rtdc")

    read.concatenated_hdf5_data([path2, path],
                                path_out=path_out,
                                features=["deform", "area_um"],
                                compute_frame=False,
                                )
    # Check outcome
    with h5py.File(path_out) as h5:
        assert len(h5["events"].keys()) == 2
        assert np.allclose(h5["events/deform"][0],
                           0.07405636775888857,
                           atol=0, rtol=1e-7)
        assert np.allclose(h5["events/area_um"][0],
                           0.559682,
                           atol=0, rtol=1e-5)
