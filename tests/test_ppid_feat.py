from dcnum.feat import QueueEventExtractor


def test_ppid_decoding_extr_check_kwargs():
    extr_ppid = "legacy:b=1^h=0"
    kwargs = QueueEventExtractor.get_ppkw_from_ppid(extr_ppid)
    assert kwargs["haralick"] is False
    assert kwargs["brightness"] is True


def test_ppid_encoding_extr_check_kwargs():
    kwargs = {"haralick": True, "brightness": False}
    ppid = QueueEventExtractor.get_ppid_from_ppkw(kwargs)
    assert ppid == "legacy:b=0^h=1"


def test_ppid_required_method_definitions():
    extr_code = "legacy"
    extr_class = QueueEventExtractor
    assert hasattr(extr_class, "get_ppid")
    assert hasattr(extr_class, "get_ppid_code")
    assert hasattr(extr_class, "get_ppid_from_ppkw")
    assert hasattr(extr_class, "get_ppkw_from_ppid")
    assert extr_class.get_ppid_code() == extr_code
