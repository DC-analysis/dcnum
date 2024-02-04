from dcnum import segm

import pytest


@pytest.mark.parametrize("segm_ppid", [
    "thresh:t=-3:cle=1^f=1^clo=2",
    "thresh:t=-5:cle=1^f=0^clo=3",
    "thresh:t=-5:cle=0^f=1^clo=2",
    "thresh:t=-5:cle=1^f=1^clo=4",
])
def test_ppid_decoding_thresh(segm_ppid):
    segm_class = segm.get_available_segmenters()["thresh"]
    kwargs = segm_class.get_ppkw_from_ppid(segm_ppid)
    segm_inst = segm_class(**kwargs)
    assert segm_inst.get_ppid() == segm_ppid


def test_ppid_decoding_thresh_check_kwargs():
    segm_ppid = "thresh:t=-3:cle=0^f=1^clo=3"
    segm_class = segm.get_available_segmenters()["thresh"]
    kwargs = segm_class.get_ppkw_from_ppid(segm_ppid)
    assert kwargs["thresh"] == -3
    assert kwargs["kwargs_mask"]["clear_border"] is False
    assert kwargs["kwargs_mask"]["fill_holes"] is True
    assert kwargs["kwargs_mask"]["closing_disk"] == 3


@pytest.mark.parametrize("segm_code", segm.get_available_segmenters().keys())
def test_ppid_required_method_definitions(segm_code):
    segm_class = segm.get_available_segmenters()[segm_code]
    assert hasattr(segm_class, "get_ppid")
    assert hasattr(segm_class, "get_ppid_code")
    assert hasattr(segm_class, "get_ppid_from_ppkw")
    assert hasattr(segm_class, "get_ppkw_from_ppid")
    assert segm_class.get_ppid_code() == segm_code


def test_ppid_segm_base_with_thresh():
    scls = segm.get_available_segmenters()["thresh"]
    sthr = scls(thresh=-3)
    assert sthr.get_ppid_code() == "thresh"
    assert sthr.get_ppid() == "thresh:t=-3:cle=1^f=1^clo=2"
