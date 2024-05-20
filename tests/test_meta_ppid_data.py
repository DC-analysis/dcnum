from dcnum.read import HDF5Data

import pytest


def test_ppid_decoding_dat_check_kwargs():
    dat_ppid = "hdf:p=0.2658"
    kwargs = HDF5Data.get_ppkw_from_ppid(dat_ppid)
    assert kwargs["pixel_size"] == 0.2658


@pytest.mark.parametrize("imppid,value", [
    ["0", None],
    ["10", 10],
    ["10-20-n", slice(10, 20)],
    ["10-20-2", slice(10, 20, 2)],
    ["n-n-2", slice(None, None, 2)],
])
def test_ppid_decoding_dat_check_kwargs_index_mapping(imppid, value):
    dat_ppid = f"hdf:p=0.2658^i={imppid}"
    kwargs = HDF5Data.get_ppkw_from_ppid(dat_ppid)
    assert kwargs["index_mapping"] == value


def test_ppid_encoding_dat_check_kwargs():
    kwargs = {"pixel_size": 0.34}
    ppid = HDF5Data.get_ppid_from_ppkw(kwargs)
    assert ppid == "hdf:p=0.34^i=0"


def test_ppid_encoding_dat_check_kwargs_acc():
    # accuracy for pixel_size is 8 digits after the decimal point
    kwargs = {"pixel_size": 0.3400000036}
    ppid = HDF5Data.get_ppid_from_ppkw(kwargs)
    assert ppid == "hdf:p=0.34^i=0"


def test_ppid_required_method_definitions():
    dat_code = "hdf"
    dat_class = HDF5Data
    assert hasattr(dat_class, "get_ppid")
    assert hasattr(dat_class, "get_ppid_code")
    assert hasattr(dat_class, "get_ppid_from_ppkw")
    assert hasattr(dat_class, "get_ppkw_from_ppid")
    assert dat_class.get_ppid_code() == dat_code
