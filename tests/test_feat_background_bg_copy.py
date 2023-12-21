import h5py
import numpy as np

from dcnum.feat.feat_background import bg_copy

from helper_methods import retrieve_data


def test_copy_simple():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_out = path.with_name("output.rtdc")

    with h5py.File(path) as h5:
        assert "image_bg" in h5["events"], "sanity check"

    assert not path_out.exists(), "sanity check"

    with bg_copy.BackgroundCopy(input_data=path,
                                output_path=path_out) as bic:
        assert bic.get_ppid() == "copy:"
        # process the data
        assert bic.get_progress() == 0
        bic.process()
        assert bic.get_progress() == 1
    assert path_out.exists()

    with h5py.File(path_out) as h5:
        assert "image_bg" in h5["events"]
        assert np.median(h5["events/image_bg"][0]) == 186.0


def test_copy_simple_same_path():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_out = path  # [sic!]

    with h5py.File(path) as h5:
        assert "image_bg" in h5["events"], "sanity check"

    with bg_copy.BackgroundCopy(input_data=path,
                                output_path=path_out) as bic:
        # process the data
        assert bic.get_progress() == 0
        bic.process()
        assert bic.get_progress() == 1
    assert path_out.exists()

    with h5py.File(path_out) as h5:
        assert "image_bg" in h5["events"]
        assert np.median(h5["events/image_bg"][0]) == 186.0
