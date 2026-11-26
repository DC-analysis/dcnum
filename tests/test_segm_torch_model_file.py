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


def test_metadata_loading_from_g2_cb32a():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
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
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")

    with pytest.raises(ValueError,
                       match="not supported or invalid"):
        torch_model.get_model_meta(model_file, device="gpu")


def test_segm_torch_complement_wiring_options():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")

    # The wiring options are "inductor", "openvino", and "torch.eager"
    # on the CPU; the first available one is used.
    if torch_model.has_cpp_compiler():
        expected = "inductor"
    elif torch_model.openvino.module_available():
        expected = "openvino"
    else:
        expected = "torch.eager"

    meta = torch_model.get_model_meta(model_file)
    assert meta["backend"] == expected
    assert meta["device"] == "cpu"

    meta = torch_model.get_model_meta(model_file, device=torch.device("cpu"))
    assert meta["backend"] == expected
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
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
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
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
    # The wiring options are "inductor", "openvino", and "torch.eager"
    # on the CPU.
    meta = torch_model.get_model_meta(model_file)
    if torch_model.openvino.module_available():
        assert meta["backend"] == "openvino"
    else:
        assert meta["backend"] == "torch.eager"
    assert meta["device"] == "cpu"
    # explicitly asking for a backend skips the backend check
    meta = torch_model.get_model_meta(model_file, backend="inductor")
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"


def test_compile_inductor_dynamic_batch_size(monkeypatch):
    """Inductor must not recompile the model for a new batch size"""
    from torch._inductor import cpp_builder
    try:
        cpp_builder.get_cpp_compiler()
    except BaseException:
        pytest.skip("No C++ compiler found")
    # make sure the model is not taken from the cache
    monkeypatch.setattr(torch_model, "cache_model", {})
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
    model, _ = torch_model.load_model(model_file,
                                      backend="inductor",
                                      device="cpu")
    with torch.inference_mode():
        # batch sizes like the GPU batches of a chunk with 1000 frames
        # and of a last chunk with 137 frames
        mask_a = model(torch.zeros((250, 80, 320), dtype=torch.uint8))
        with torch.compiler.set_stance("fail_on_recompile"):
            mask_b = model(torch.zeros((69, 80, 320), dtype=torch.uint8))
            mask_c = model(torch.zeros((68, 80, 320), dtype=torch.uint8))
            # Note that PyTorch's convolution adds a guard for batch
            # sizes below 16, so very small batches compile once more.
            mask_d = model(torch.zeros((16, 80, 320), dtype=torch.uint8))
    assert mask_a.shape == (250, 80, 320)
    assert mask_b.shape == (69, 80, 320)
    assert mask_c.shape == (68, 80, 320)
    assert mask_d.shape == (16, 80, 320)


def test_wiring_none_available_clear_error(monkeypatch):
    """Model files must only be used with the wiring options they list"""
    reason_orig = torch_model.get_wiring_unavailable_reason

    def reason_mock(wiring, check_backend=True):
        if wiring["backend"] == "torch.eager":
            return "mocked unavailable"
        return reason_orig(wiring, check_backend=check_backend)

    monkeypatch.setattr(torch_model, "get_wiring_unavailable_reason",
                        reason_mock)
    monkeypatch.setattr(torch_model, "has_cpp_compiler", lambda: False)
    monkeypatch.setattr(torch_model.openvino, "module_available",
                        lambda: False)
    monkeypatch.setattr(torch_model, "cache_model_meta", {})
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
    with pytest.raises(torch_model.WiringUnavailableError) as exc:
        torch_model.get_model_meta(model_file)
    msg = str(exc.value)
    assert "'inductor' on 'cpu' (no C++ compiler found)" in msg
    assert "'openvino' on 'cpu' (OpenVINO is not installed)" in msg
    assert "'torch.eager' on 'cpu' (mocked unavailable)" in msg
    with pytest.raises(torch_model.WiringUnavailableError):
        torch_model.get_model_meta(model_file, device="cpu")
    # specifying backend and device manually always works
    meta = torch_model.get_model_meta(model_file, backend="torch.eager",
                                      device="cpu")
    assert meta["backend"] == "torch.eager"


def test_metadata_wiring_options_cb32a():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_cb32a.zip")
    meta = torch_model.get_model_meta(model_file, backend="torch.eager",
                                      device="cpu")
    assert meta["wiring_options"] == [
        {"backend": "inductor", "device": "cpu"},
        {"backend": "openvino", "device": "cpu"},
        {"backend": "torch.eager", "device": "cpu"},
    ]
