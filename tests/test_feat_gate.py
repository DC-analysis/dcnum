from dcnum.feat import Gate
from dcnum.read import HDF5Data
import h5py

import pytest

from helper_methods import retrieve_data


def test_features():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20
        h5.attrs["online_filter:deform min"] = 0.01
        h5.attrs["online_filter:deform max"] = 0.2
        h5.attrs["online_filter:deform soft limit"] = False
        h5.attrs["online_filter:area_um min"] = 50
        h5.attrs["online_filter:area_um soft limit"] = False

    with HDF5Data(path) as hd:
        gt = Gate(data=hd, online_gates=True)
        # there is also size_x and size_y, so we don't test for entire list
        assert "area_um" in gt.features
        assert "deform" in gt.features
        assert gt.features.count("deform") == 1
        assert gt.features.count("area_um") == 1


def test_get_ppkw_from_ppid():
    kw = Gate.get_ppkw_from_ppid("norm:o=true^s=23")
    assert len(kw) == 2
    assert kw["online_gates"] is True
    assert kw["size_thresh_mask"] == 23


def test_get_ppkw_from_ppid_error_bad_code():
    with pytest.raises(ValueError,
                       match="Could not find gating method 'peter'"):
        Gate.get_ppkw_from_ppid("peter:o=true^s=23")


def test_parse_online_features_size_thresh_mask():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20

    with HDF5Data(path) as hd:
        gt1 = Gate(data=hd)
        assert gt1.kwargs["size_thresh_mask"] == 10, "default in dcnum"

        gt2 = Gate(data=hd, online_gates=True)
        assert gt2.kwargs["size_thresh_mask"] == 20, "from file"

        gt3 = Gate(data=hd, online_gates=True, size_thresh_mask=22)
        assert gt3.kwargs["size_thresh_mask"] == 22, "user override"

        assert gt3.get_ppid() == "norm:o=1^s=22"


def test_ppid():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20

    with HDF5Data(path) as hd:
        gt1 = Gate(data=hd)
        assert gt1.get_ppid_code() == "norm"
        assert gt1.get_ppid() == "norm:o=0^s=10"

        gt2 = Gate(data=hd, online_gates=True)
        assert gt2.get_ppid() == "norm:o=1^s=20"

        gt3 = Gate(data=hd, online_gates=True, size_thresh_mask=22)
        assert gt3.get_ppid() == "norm:o=1^s=22"
