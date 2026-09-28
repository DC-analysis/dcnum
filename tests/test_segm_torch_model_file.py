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
