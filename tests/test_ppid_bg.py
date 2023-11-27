import numpy as np

from dcnum.feat import feat_background as fbg

import pytest


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
@pytest.mark.parametrize("bg_ppid", [
    "sparsemed:k=200^s=1^t=0^f=0.8",
    "sparsemed:k=210^s=1^t=0^f=0.8",
    "sparsemed:k=200^s=2^t=0^f=0.8",
    "sparsemed:k=200^s=1^t=0.1^f=0.8",
    "sparsemed:k=200^s=1^t=0^f=0.9",
])
def test_ppid_decoding_sparsemed(bg_ppid, tmp_path):
    input_data = np.arange(5 * 7).reshape(1, 5, 7) * np.ones((120, 1, 1))
    path_out = tmp_path / "test.h5"
    bg_class = fbg.get_available_background_methods()["sparsemed"]
    kwargs = bg_class.get_ppkw_from_ppid(bg_ppid)
    with bg_class(input_data=input_data,
                  output_path=path_out,
                  **kwargs) as bg_inst:
        assert bg_inst.get_ppid() == bg_ppid


def test_ppid_decoding_sparsemed_check_kwargs():
    bg_ppid = "sparsemed:k=210^s=1^t=0^f=0.8"
    bg_class = fbg.get_available_background_methods()["sparsemed"]
    kwargs = bg_class.get_ppkw_from_ppid(bg_ppid)
    assert kwargs["kernel_size"] == 210
    assert np.allclose(kwargs["split_time"], 1.0)
    assert np.allclose(kwargs["thresh_cleansing"], 0.0)
    assert np.allclose(kwargs["frac_cleansing"], 0.8)


@pytest.mark.parametrize("bg_code",
                         fbg.get_available_background_methods().keys())
def test_ppid_required_method_definitions(bg_code):
    bg_class = fbg.get_available_background_methods()[bg_code]
    assert hasattr(bg_class, "get_ppid")
    assert hasattr(bg_class, "get_ppid_code")
    assert hasattr(bg_class, "get_ppid_from_ppkw")
    assert hasattr(bg_class, "get_ppkw_from_ppid")
    assert bg_class.get_ppid_code() == bg_code


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_ppid_bg_base_with_sparsemed(tmp_path):
    input_data = np.arange(5 * 7).reshape(1, 5, 7) * np.ones((120, 1, 1))
    path_out = tmp_path / "test.h5"
    scls = fbg.get_available_background_methods()["sparsemed"]
    with scls(input_data=input_data,
              output_path=path_out,
              thresh_cleansing=.22) as sthr:
        assert sthr.get_ppid_code() == "sparsemed"
        assert sthr.get_ppid() == "sparsemed:k=200^s=1^t=0.22^f=0.8"
