from dcnum.feat import Gate
from dcnum.read import HDF5Data
import h5py
import numpy as np

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


def test_gate_event():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20
        h5.attrs["online_filter:deform min"] = 0.1
        h5.attrs["online_filter:deform max"] = 0.5
        h5.attrs["online_filter:deform soft limit"] = False
        h5.attrs["online_filter:area_um min"] = 50
        h5.attrs["online_filter:area_um soft limit"] = False

    with HDF5Data(path) as hd:
        skw = {"size_x": 200, "size_y": 200}
        gt = Gate(data=hd, online_gates=True)

        assert gt.gate_event(dict(deform=0.3, area_um=55, **skw))
        assert gt.gate_event(dict(deform=0.3, area_um=55,
                                  userdef1=0.3, userdef2=55, **skw))

        assert not gt.gate_event(dict(deform=0.01, area_um=55, **skw))
        assert not gt.gate_event(dict(deform=0.3, area_um=40, **skw))
        assert not gt.gate_event(dict(deform=0.6, area_um=55, **skw))
        assert not gt.gate_event(dict(deform=0.6, area_um=22, **skw))


def test_gate_events():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20
        h5.attrs["online_filter:deform min"] = 0.1
        h5.attrs["online_filter:deform max"] = 0.5
        h5.attrs["online_filter:deform soft limit"] = False
        h5.attrs["online_filter:area_um min"] = 50
        h5.attrs["online_filter:area_um soft limit"] = False

    with HDF5Data(path) as hd:
        gt = Gate(data=hd, online_gates=True)

        exp = np.array([True, True, False, False, False, False],
                       dtype=bool)
        act = gt.gate_events(dict(
            deform=np.array([0.3, 0.3, 0.01, 0.3, 0.6, 0.6], dtype=float),
            area_um=np.array([55, 55, 55, 40, 55, 22], dtype=float),
            size_x=np.full(6, 200, dtype=float),
            size_y=np.full(6, 200, dtype=float),
        ))

        assert isinstance(act, np.ndarray)
        assert np.all(exp == act)


def test_gate_feature():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")
    # Since these are CytoShot data, there are no online filters here.
    with h5py.File(path, "a") as h5:
        h5.attrs["online_contour:bin area min"] = 20
        h5.attrs["online_filter:deform min"] = 0.1
        h5.attrs["online_filter:deform max"] = 0.5
        h5.attrs["online_filter:deform soft limit"] = False
        h5.attrs["online_filter:area_um min"] = 50
        h5.attrs["online_filter:area_um soft limit"] = False

    with HDF5Data(path) as hd:
        # sanity checks for later maybe
        assert np.allclose(
            hd["deform"][0],
            0.0740563677588885,
            atol=0, rtol=1e-5)
        assert np.allclose(
            hd["deform"][1],
            0.4113239579639607,
            atol=0, rtol=1e-5)
        assert np.allclose(
            hd["area_um"][0],
            0.559682,
            atol=0, rtol=1e-5)
        assert np.allclose(
            hd["area_um"][1],
            91.193185875,
            atol=0, rtol=1e-5)

        gt = Gate(data=hd, online_gates=True)
        assert gt.gate_feature(feat="deform", data=0.3)
        assert not gt.gate_feature(feat="deform", data=0.51)
        assert gt.gate_feature(feat="area_um", data=55)
        assert not gt.gate_feature(feat="area_um", data=48)


def test_gate_mask():
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    with HDF5Data(path) as hd:
        gt = Gate(data=hd, online_gates=False, size_thresh_mask=10)

        mask1 = np.zeros((20, 20), dtype=bool)
        mask1[2:4, 2:4] = True
        assert not gt.gate_mask(mask1)
        assert not gt.gate_mask(mask1, mask_sum=np.sum(mask1))

        mask2 = np.zeros((20, 20), dtype=bool)
        mask2[2:10, 2:10] = True
        assert gt.gate_mask(mask2)
        assert gt.gate_mask(mask2, mask_sum=np.sum(mask2))


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
