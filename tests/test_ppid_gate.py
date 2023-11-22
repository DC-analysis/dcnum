from dcnum.feat.gate import Gate


def test_ppid_decoding_gate_check_kwargs():
    gate_ppid = "norm:o=1^s=12"
    kwargs = Gate.get_ppkw_from_ppid(gate_ppid)
    assert kwargs["size_thresh_mask"] == 12
    assert kwargs["online_gates"] is True


def test_ppid_encoding_gate_check_kwargs():
    kwargs = {"size_thresh_mask": 11, "online_gates": False}
    ppid = Gate.get_ppid_from_ppkw(kwargs)
    assert ppid == "norm:o=0^s=11"


def test_ppid_required_method_definitions():
    gate_code = "norm"
    gate_class = Gate
    assert hasattr(gate_class, "get_ppid")
    assert hasattr(gate_class, "get_ppid_code")
    assert hasattr(gate_class, "get_ppid_from_ppkw")
    assert hasattr(gate_class, "get_ppkw_from_ppid")
    assert gate_class.get_ppid_code() == gate_code
