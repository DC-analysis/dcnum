import multiprocessing as mp

from dcnum import logic

from helper_methods import retrieve_data


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
