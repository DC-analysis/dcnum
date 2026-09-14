import pytest

from helper_methods import retrieve_model

torch = pytest.importorskip("torch")

from dcnum.segm.segm_torch import torch_model  # noqa: E402


def test_metadata_loading_from_g2_02dcd():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")
    _, meta = torch_model.load_model(model_file,
                                     backend="inductor",
                                     device="cpu")
    assert isinstance(meta, dict)
    assert "preprocessing" not in meta
    assert meta["image_shape"] == [80, 320]
    assert meta["batch_size"] == 10


def test_metadata_loading_from_g2_a8773():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_a8773.zip")
    _, meta = torch_model.load_model(model_file)
    assert isinstance(meta, dict)
    assert "preprocessing" not in meta
    assert meta["image_shape"] == [80, 320]
    assert "batch_size" not in meta


def test_segm_torch_invalid_wiring_options():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")

    with pytest.raises(ValueError,
                       match="does not have wiring options"):
        torch_model.get_model_meta(model_file)

    with pytest.raises(ValueError,
                       match="not supported or invalid"):
        torch_model.get_model_meta(model_file, device="gpu")


def test_segm_torch_complement_wiring_options():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_a8773.zip")

    meta = torch_model.get_model_meta(model_file)
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"

    meta = torch_model.get_model_meta(model_file, device=torch.device("cpu"))
    assert meta["backend"] == "inductor"
    assert meta["device"] == "cpu"

    meta2 = torch_model.get_model_meta(model_file, backend="openvino")
    assert meta2["backend"] == "openvino"
    assert meta2["device"] == "cpu"
