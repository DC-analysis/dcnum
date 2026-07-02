import h5py
import numpy as np

import pytest

from dcnum import read, segm

from helper_methods import retrieve_data, retrieve_model

torch = pytest.importorskip("torch")

from dcnum.segm.segm_torch import segm_torch_base  # noqa: E402
from dcnum.segm.segm_torch import torch_model  # noqa: E402


def test_metadata_loading_from_unet_1316_naiad_g1_abd2a():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")
    device = torch.device("cpu")
    _, meta = torch_model.load_model(model_file, device)
    assert isinstance(meta, dict)
    assert "preprocessing" not in meta.keys()
    assert meta["image_shape"] == [80, 320]
    assert meta["batch_size"] == 10


def test_segm_torch_validate_model_file_logs_negate():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")
    sm = segm.segm_torch.SegmentTorchUNI

    # Creating a specific log file will mak the model invalid
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_2024.zip")

    with read.HDF5Data(path) as hd:
        # sanity check
        assert "dclab-compress" in hd.logs
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="must not be compressed 2024-05-07"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )

    # Remove the offending log
    with h5py.File(path, "a") as h5:
        del h5["logs/dclab-compress"]

    # Try again, this should work now.
    with read.HDF5Data(path) as hd:
        # sanity check
        assert "dclab-compress" not in hd.logs
        sm.validate_applicability(
            segmenter_kwargs={"model_file": model_file},
            meta=hd.meta,
            logs=hd.logs
        )


def test_segm_torch_validate_model_file_meta_value():
    """Test whether model validation fails for invalid metadata"""
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")
    sm = segm.segm_torch.SegmentTorchUNI

    # Create a test dataset with metadata that will make the model invalid
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_2023.zip")

    with h5py.File(path, "a") as h5:
        h5.attrs["setup:channel width"] = 30.

    with read.HDF5Data(path) as hd:
        # sanity check
        assert hd.meta["setup:channel width"] == 30
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="channel width must be 20 micrometers"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )

    # Repeat the same thing, this time fixing the attribute
    with h5py.File(path, "a") as h5:
        h5.attrs["setup:channel width"] = 20.

    with read.HDF5Data(path) as hd:
        # sanity check
        assert hd.meta["setup:channel width"] == 20
        sm.validate_applicability(
            segmenter_kwargs={"model_file": model_file},
            meta=hd.meta,
            logs=hd.logs
        )


def test_segm_torch_uni():
    """Basic PyTorch segmenter"""
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_2024.zip")
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g1_cb45f.zip")

    sm = segm.segm_torch.SegmentTorchUNI(model_file=model_file)
    assert not sm.requires_background_correction
    assert sm.mask_postprocessing
    assert not sm.mask_default_kwargs["closing_disk"]
    assert sm.get_ppid() == f"torchuni:m={model_file.name}:cle=1^f=1^clo=0"

    with read.HDF5Data(path) as hd:
        labels_seg = sm.segment_batch_with_labeling(
            hd.image[:10][:, 8:-8, 32:-32])
        assert np.all(np.unique(labels_seg[0]) == [0, 1, 2])
        assert np.sum(labels_seg[0] == 0) == 14978  # background
        assert np.sum(labels_seg[0] == 1) == 831  # first label
        assert np.sum(labels_seg[0] == 2) == 575  # first label


def test_segm_torch_uni_bad_model():
    """Basic PyTorch segmenter"""
    path = retrieve_data(
        "fmt-hdf5_cytoshot_full-features_2024.zip")
    # this is the wrong model
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_17ec6.zip")  # [sic]

    sm = segm.segm_torch.SegmentTorchUNI(model_file=model_file)

    with read.HDF5Data(path) as hd:
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="requires  version 2.0"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )
