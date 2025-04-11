import pathlib

from dcnum import read

from helper_methods import retrieve_data


def test_concatenated_hdf5_as_file(tmp_path):
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_concat = tmp_path / "concatenated.rtdc"
    with read.concatenated_hdf5_data([path] * 10, path_out=path_concat):
        pass
    assert path_concat.exists()


def test_concatenated_hdf5_new_file():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with read.concatenated_hdf5_data([path] * 10, path_out=True) as hd:
        pass
        path_out = hd.path
    assert path_out.exists()


def test_concatenated_hdf5_in_memory():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with read.concatenated_hdf5_data([path] * 10, path_out=None) as hd:
        pass
        path_out = hd.path
    assert not isinstance(path_out, (pathlib.Path, str))
