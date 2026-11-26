import pytest

from helper_methods import retrieve_model

torch = pytest.importorskip("torch")

from dcnum.segm.segm_torch import torch_model  # noqa: E402


def test_missing_model_file_type():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")
    with pytest.raises(ValueError, match="Unknown model file format"):
        torch_model.load_model(model_file,
                               backend="inductor",
                               device="cpu")


def test_metadata_loading_from_g2_0901b():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")
    _, meta = torch_model.load_model(model_file,
                                     backend="inductor",
                                     device="cpu")
    assert isinstance(meta, dict)
    assert meta["model_file_format"] == "ExportedProgram"
    assert "preprocessing" not in meta
    assert meta["image_shape"] == [80, 320]
    assert meta["batch_size"] == 0


def test_segm_torch_invalid_no_gpu():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")

    with pytest.raises(ValueError,
                       match="not supported or invalid"):
        torch_model.get_model_meta(model_file, device="gpu")


def test_segm_torch_complement_wiring_options():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")

    meta = torch_model.get_model_meta(model_file)
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"

    meta = torch_model.get_model_meta(model_file, device=torch.device("cpu"))
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"

    meta2 = torch_model.get_model_meta(model_file, backend="openvino")
    assert meta2["backend"] == "openvino"
    assert meta2["device"] == "cpu"


@pytest.mark.parametrize("backend,layout_opt_disabled", [
    ("inductor", True),
    ("cudagraphs", False),
])
def test_compile_inductor_without_layout_optimization(monkeypatch, backend,
                                                      layout_opt_disabled):
    """Inductor's layout optimization (channels-last) must be disabled"""
    calls = []

    def compile_mock(model, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(torch_model.torch, "compile", compile_mock)
    # make sure the model is not taken from the cache
    monkeypatch.setattr(torch_model, "cache_model", {})
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")
    torch_model.load_model(model_file, backend=backend, device="cpu")
    assert len(calls) == 1
    assert calls[0]["backend"] == backend
    options = calls[0].get("options", {})
    if layout_opt_disabled:
        assert options["layout_optimization"] is False
    else:
        assert "layout_optimization" not in options


def test_wiring_available_cuda(monkeypatch):
    monkeypatch.setattr(torch_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_model.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch_model, "has_triton", lambda: False)
    is_available = torch_model.is_wiring_available
    assert is_available({"backend": "torch.eager", "device": "cuda"})
    assert is_available({"backend": "torch.eager", "device": "cuda:0"})
    # there is only one GPU
    assert not is_available({"backend": "torch.eager", "device": "cuda:1"})
    # inductor requires Triton on CUDA devices
    assert not is_available({"backend": "inductor", "device": "cuda"})
    monkeypatch.setattr(torch_model, "has_triton", lambda: True)
    assert is_available({"backend": "inductor", "device": "cuda"})


def test_wiring_available_no_cuda(monkeypatch):
    monkeypatch.setattr(torch_model.torch.cuda, "is_available",
                        lambda: False)
    is_available = torch_model.is_wiring_available
    assert is_available({"backend": "torch.eager", "device": "cpu"})
    assert not is_available({"backend": "torch.eager", "device": "cuda"})
    assert not is_available({"backend": "cudagraphs", "device": "cuda:0"})


def test_wiring_skip_inductor_without_cpp_compiler(monkeypatch):
    """The first wiring option of the model is not available"""
    monkeypatch.setattr(torch_model, "has_cpp_compiler", lambda: False)
    monkeypatch.setattr(torch_model, "cache_model_meta", {})
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_0901b.zip")
    # The wiring options are ("inductor", "cpu") and ("openvino", "cpu").
    if torch_model.openvino.module_available():
        meta = torch_model.get_model_meta(model_file)
        assert meta["backend"] == "openvino"
        assert meta["device"] == "cpu"
    else:
        with pytest.raises(ValueError, match="not have wiring options"):
            torch_model.get_model_meta(model_file)
    # explicitly asking for a backend skips the backend check
    meta = torch_model.get_model_meta(model_file, backend="inductor")
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"
