import shutil

from dcnum import common, logic
from dcnum.segm.segm_torch import segm_torch_base
import h5py

import pytest

from helper_methods import retrieve_data, retrieve_model


def test_basic_job():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path)
    assert job["path_out"] == path.with_name(path.stem + "_dcn.rtdc")
    assert job["num_procs"] == common.cpu_count()
    assert not job["debug"]


def test_compression_options_default():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path)
    assert job.get_hdf5_dataset_kwargs() == {"fletcher32": True,
                                             "compression": 32015,
                                             "compression_opts": (5,)}


def test_compression_options_none():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path, compression="none")
    assert job.get_hdf5_dataset_kwargs() == {"fletcher32": True,
                                             "compression": None,
                                             "compression_opts": None}


def test_compression_options_zstd_high():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path, compression="zstd-9")
    assert job.get_hdf5_dataset_kwargs() == {"fletcher32": True,
                                             "compression": 32015,
                                             "compression_opts": (9,)}


def test_compression_options_zstd_invalid():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path, compression="foobar-9")
    with pytest.raises(ValueError, match="Unsupported compression"):
        job.get_hdf5_dataset_kwargs()


def test_copied_data():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path,
                                 segmenter_code="thresh",
                                 segmenter_kwargs=None,
                                 )
    _, pdict = job.get_ppid(ret_dict=True)
    assert pdict["seg_id"] == "thresh:t=-6:cle=1^f=1^clo=2"
    seg_kwargs = job["segmenter_kwargs"]
    seg_kwargs["closing_disk"] = 10
    # changing dictionary does not change keys
    assert pdict["seg_id"] == "thresh:t=-6:cle=1^f=1^clo=2"


def test_kwargs_not_modified(tmp_path):
    """The job must not modify the dictionaries passed by the caller"""
    path1 = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path2 = tmp_path / "other_pixel_size.rtdc"
    shutil.copy(path1, path2)
    with h5py.File(path2, "a") as h5:
        h5.attrs["imaging:pixel size"] = 0.34

    data_kwargs = {}
    segmenter_kwargs = {"thresh": -6}
    job1 = logic.DCNumPipelineJob(path_in=path1,
                                  data_kwargs=data_kwargs,
                                  segmenter_kwargs=segmenter_kwargs)
    job2 = logic.DCNumPipelineJob(path_in=path2,
                                  data_kwargs=data_kwargs,
                                  segmenter_kwargs=segmenter_kwargs)
    assert data_kwargs == {}
    assert segmenter_kwargs == {"thresh": -6}
    assert job1["data_kwargs"]["pixel_size"] == 0.2645
    # pixel size of the second input file, not of the first one
    assert job2["data_kwargs"]["pixel_size"] == 0.34


def test_segmenter_mask():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path,
                                 segmenter_code="thresh",
                                 segmenter_kwargs={
                                     "kwargs_mask": {"closing_disk": 3}},
                                 )
    _, pdict = job.get_ppid(ret_dict=True)
    assert pdict["seg_id"] == "thresh:t=-6:cle=1^f=1^clo=3"


def test_validate_invalid_model():
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g1_910c2.zip")

    # Create a test dataset with metadata that will make the model invalid
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2024.zip")

    with h5py.File(path, "a") as h5:
        h5.attrs["setup:chip region"] = "reservoir"

    job = logic.DCNumPipelineJob(path_in=path,
                                 segmenter_code="torchmpo",
                                 segmenter_kwargs={
                                     "model_file": model_file},
                                 )

    with pytest.raises(
            segm_torch_base.SegmenterNotApplicableError,
            match="only experiments in channel region supported"):
        job.validate()


def test_validate_ok():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path,
                                 segmenter_code="thresh",
                                 segmenter_kwargs={
                                     "kwargs_mask": {"closing_disk": 3}},
                                 )
    assert job.validate()
