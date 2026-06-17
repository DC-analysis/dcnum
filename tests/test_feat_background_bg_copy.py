import h5py
import numpy as np

from dcnum.feat.feat_background import bg_copy
from dcnum.read import HDF5Data
from dcnum.write import HDF5Writer

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


def test_copy_simple_with_bg_off():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_out = path.with_name("output.rtdc")

    with h5py.File(path, "a") as h5:
        assert "image_bg" in h5["events"], "sanity check"
        bg_off = np.linspace(-1, 1, len(h5["events/deform"]))
        h5["events/bg_off"] = bg_off

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
        assert np.allclose(h5["events/bg_off"], bg_off)
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


def test_copy_basin_based_background():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    path_out = path.with_name("output.rtdc")

    with h5py.File(path, "a") as h5:
        image_bg = h5["events/image_bg"][:]
        # remove the background data
        del h5["events/image_bg"]

        bg1 = image_bg[0]
        bg2 = image_bg[0] + 1
        bg = np.vstack((bg1[np.newaxis], bg2[np.newaxis]))
        mapping = np.zeros(len(image_bg), dtype=int)
        mapping[1:] = 1
        assert len(mapping) > 2
        assert not np.all(bg1 == bg2)

        with HDF5Writer(h5) as hw:
            hw.store_basin(
                name="test basin with background",
                mapping=mapping,
                internal_data={"image_bg": bg}
            )

    # sanity check
    with HDF5Data(path) as hd:
        assert np.all(hd.image_bg[0] == bg1)
        assert np.all(hd.image_bg[1] == bg2)
        assert np.all(hd.image_bg[2] == bg2)
        assert np.all(hd.image_corr[0]
                      == np.array(hd.image[0], dtype=int) - bg1)
        assert np.all(hd.image_corr[1]
                      == np.array(hd.image[1], dtype=int) - bg2)
        assert np.all(hd.image_corr[2]
                      == np.array(hd.image[2], dtype=int) - bg2)

    # background "computation"
    with bg_copy.BackgroundCopy(input_data=path,
                                output_path=path_out) as bic:
        # process the data
        assert bic.get_progress() == 0
        bic.process()
        assert bic.get_progress() == 1
    assert path_out.exists()

    # basic data reading
    with h5py.File(path_out) as h5:
        assert "image_bg" in h5["basin_events"]
        assert h5["basin_events/image_bg"].shape == (2, 80, 400)
        assert np.all(h5["basin_events/image_bg"][0] == bg1)
        assert np.all(h5["basin_events/image_bg"][1] == bg2)
        assert "image_bg" not in h5["events"]

    # read with HDF5Data
    with HDF5Data(path_out) as hd:
        assert np.all(hd.image_bg[0] == bg1)
        assert np.all(hd.image_bg[1] == bg2)
        assert np.all(hd.image_bg[2] == bg2)
        assert np.all(hd.image_corr[0]
                      == np.array(hd.image[0], dtype=int) - bg1)
        assert np.all(hd.image_corr[1]
                      == np.array(hd.image[1], dtype=int) - bg2)
        assert np.all(hd.image_corr[2]
                      == np.array(hd.image[2], dtype=int) - bg2)
        assert np.all(hd.image_corr[10]
                      == np.array(hd.image[10], dtype=int) - bg2)
