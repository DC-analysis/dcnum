from dcnum.feat.gate import Gate


def test_ppid_decoding_extr_check_kwargs():
    gate_ppid = "norm:o=1^s=12"
    kwargs = Gate.get_ppkw_from_ppid(gate_ppid)
    assert kwargs["size_thresh_mask"] == 12
    assert kwargs["online_gates"] is True


def test_ppid_encoding_extr_check_kwargs():
    kwargs = {"size_thresh_mask": 11, "online_gates": False}
    ppid = Gate.get_ppid_from_ppkw(kwargs)
    assert ppid == "norm:o=0^s=11"


def test_ppid_required_method_definitions():
    extr_code = "norm"
    extr_class = Gate
    assert hasattr(extr_class, "get_ppid")
    assert hasattr(extr_class, "get_ppid_code")
    assert hasattr(extr_class, "get_ppid_from_ppkw")
    assert hasattr(extr_class, "get_ppkw_from_ppid")
    assert extr_class.get_ppid_code() == extr_code
