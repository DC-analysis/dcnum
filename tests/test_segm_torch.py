import pathlib

import cv2
import h5py
import numpy as np

import pytest

from dcnum import read, segm, write

from helper_methods import extract_data, retrieve_data, retrieve_model

data_path = pathlib.Path(__file__).parent / "data"

torch = pytest.importorskip("torch")

from dcnum.segm.segm_torch import segm_torch_base  # noqa: E402


def test_segm_torch_validate_model_file_logs():
    """Test whether model validation fails for invalid logs"""
    model_file = retrieve_model(
        data_path / "segm-torch-model_unet-dcnum-test_g1_910c2.zip")
    sm = segm.segm_torch.SegmentTorchMPO

    # Creating a specific log file will mak the model invalid
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_2024.zip")

    with write.HDF5Writer(path) as hw:
        hw.store_log("testing-fail",
                     ["dear user of this software, I am here to "
                      "inform you that the measurement failed"])

    with read.HDF5Data(path) as hd:
        # sanity check
        assert "testing-fail" in hd.logs
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="measurement failed according to the logs"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )


def test_segm_torch_validate_model_file_meta():
    """Test whether model validation fails for invalid metadata"""
    model_file = retrieve_model(
        data_path / "segm-torch-model_unet-dcnum-test_g1_910c2.zip")
    sm = segm.segm_torch.SegmentTorchMPO

    # Create a test dataset with metadata that will make the model invalid
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_2024.zip")

    with h5py.File(path, "a") as h5:
        h5.attrs["setup:chip region"] = "reservoir"

    with read.HDF5Data(path) as hd:
        # sanit check
        assert hd.meta["setup:chip region"] == "reservoir"
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="only experiments in channel region supported"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )

    # Repeat the same thing, this time deleting the attribute
    with h5py.File(path, "a") as h5:
        del h5.attrs["setup:chip region"]

    with read.HDF5Data(path) as hd:
        # sanity check
        assert "setup:chip region" not in hd.meta
        with pytest.raises(
                segm_torch_base.SegmenterNotApplicableError,
                match="'setup:chip region' missing in meta"):
            sm.validate_applicability(
                segmenter_kwargs={"model_file": model_file},
                meta=hd.meta,
                logs=hd.logs
            )


def test_segm_torch_mpo():
    """Basic PyTorch segmenter"""
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_2024.zip")
    model_file = retrieve_model(
        data_path / "segm-torch-model_unet-dcnum-test_g1_910c2.zip")

    sm = segm.segm_torch.SegmentTorchMPO(model_file=model_file)
    assert not sm.requires_background_correction
    assert sm.mask_postprocessing
    assert not sm.mask_default_kwargs["closing_disk"]
    assert sm.get_ppid() == f"torchmpo:m={model_file.name}:cle=1^f=1^clo=0"

    with read.HDF5Data(path) as hd:
        labels_seg = sm.segment_single(hd.image[0])
        assert np.all(np.unique(labels_seg) == [0, 1, 2])
        assert np.sum(labels_seg == 0) == 24194  # background
        assert np.sum(labels_seg == 1) == 831  # first label
        assert np.sum(labels_seg == 2) == 575  # first label


@pytest.mark.parametrize("image_stem", ["cell_image_1", "cell_image_2",
                                        "cell_image_3", "cell_image_4",
                                        "cell_image_5"])
def test_segm_torch_mpo_explicit(image_stem):
    model_file = retrieve_model(
        data_path / "segm-torch-model_unet-dcnum-test_g1_910c2.zip")
    img_dir = extract_data(
        data_path / "segm-torch-test-data_unet-dcnum-test_g1_910c2.zip")

    image = cv2.imread(str(img_dir / f"{image_stem}.png"))[:, :, 0]
    mask_exp_segm = np.array(
        cv2.imread(str(img_dir / f"{image_stem}_mask_segm.png"))[:, :, 0],
        dtype=bool)
    mask_exp = np.array(
        cv2.imread(str(img_dir / f"{image_stem}_mask.png"))[:, :, 0],
        dtype=bool)

    sg = segm.segm_torch.SegmentTorchMPO

    # simple segmentation
    mask_segm = sg.segment_algorithm(image, model_file=model_file)
    # cv2.imwrite(str(f"{image_stem}_mask_segm.png"),
    #             np.array(mask_segm*255, dtype=np.uint8))
    assert np.all(mask_exp_segm == np.array(mask_segm, dtype=bool))

    # segmentation + mask postprocessing
    mask_act = sg.process_mask(np.copy(mask_segm), **sg.mask_default_kwargs)
    # cv2.imwrite(str(f"{image_stem}_mask.png"),
    #             np.array(mask_act*255, dtype=np.uint8))
    assert np.all(mask_exp == np.array(mask_act, dtype=bool))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available -> skipping GPU tests")
@pytest.mark.parametrize("image_stem", ["cell_image_1", "cell_image_2",
                                        "cell_image_3", "cell_image_4",
                                        "cell_image_5"])
def test_segm_torch_mpo_sto_similarity(image_stem):
    model_file = retrieve_model(
        data_path / "segm-torch-model_unet-dcnum-test_g1_910c2.zip")
    img_dir = extract_data(
        data_path / "segm-torch-test-data_unet-dcnum-test_g1_910c2.zip")

    image = cv2.imread(str(img_dir / f"{image_stem}.png"))[:, :, 0]

    images = image[None, ...]
    mask_mpo = segm.segm_torch.SegmentTorchMPO.segment_algorithm(
        image, model_file=model_file)
    mask_sto = segm.segm_torch.SegmentTorchSTO.segment_algorithm(
        images, model_file=model_file)
    assert np.all(mask_mpo == mask_sto[0])
