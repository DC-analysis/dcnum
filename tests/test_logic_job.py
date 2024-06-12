import multiprocessing as mp

from dcnum import logic
from dcnum.segm.segm_torch import segm_torch_base  # noqa: E402
import h5py

import pytest

from helper_methods import retrieve_data, retrieve_model


def test_basic_job():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    job = logic.DCNumPipelineJob(path_in=path)
    assert job["path_out"] == path.with_name(path.stem + "_dcn.rtdc")
    assert job["num_procs"] == mp.cpu_count()
    assert not job["debug"]


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
