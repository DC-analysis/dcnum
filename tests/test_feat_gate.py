from dcnum.feat import Gate
from dcnum.read import HDF5Data
import h5py

from helper_methods import retrieve_data


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
