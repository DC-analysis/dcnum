from dcnum import logic
from dcnum.logic.universal_worker import UniversalWorker

from helper_methods import retrieve_data, retrieve_model


def test_universal_worker_get_worker_dedications_single():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path_orig.with_name("out.rtdc")

    job = logic.DCNumPipelineJob(path_in=path_orig,
                                 path_out=path_out,
                                 segmenter_code="thresh",
                                 segmenter_kwargs={"thresh": -5},
                                 )

    dcs = UniversalWorker.get_worker_dedications(job, 1)
    assert len(dcs) == 1
    all = {
        "load_all",
        "segment_images",
        "label_masks",
        "process_labels",
        "extract_features",
    }
    assert all == set(dcs[0])


def test_universal_worker_get_worker_dedications_two():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path_orig.with_name("out.rtdc")

    job = logic.DCNumPipelineJob(path_in=path_orig,
                                 path_out=path_out,
                                 segmenter_code="thresh",
                                 segmenter_kwargs={"thresh": -5},
                                 )

    dcs = UniversalWorker.get_worker_dedications(job, 2)
    assert len(dcs) == 2
    all = {
        "load_all",
        "segment_images",
        "label_masks",
        "process_labels",
        "extract_features",
    }
    assert all == set(dcs[0])
    assert (all - {"load_all"}) == set(dcs[1])


def test_universal_worker_get_worker_dedications_torchuni_cpu():
    """segmentation in all workers, load_all only in first"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path_orig.with_name("out.rtdc")
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")

    job = logic.DCNumPipelineJob(path_in=path_orig,
                                 path_out=path_out,
                                 segmenter_code="torchuni",
                                 segmenter_kwargs={
                                     "model_file": model_file,
                                     "backend": "inductor",
                                     "device": "cpu",
                                 },
                                 )

    dcs = UniversalWorker.get_worker_dedications(job, 2)
    assert len(dcs) == 2
    all = {
        "load_all",
        "segment_images",
        "label_masks",
        "process_labels",
        "extract_features",
    }
    assert all == set(dcs[0])
    assert (all - {"load_all"}) == set(dcs[1])


def test_universal_worker_get_worker_dedications_torchuni_gpu():
    """Segmentation only done in 1st worker"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path_orig.with_name("out.rtdc")
    model_file = retrieve_model(
        "segm-torch-model_unet-dcnum-test_g2_02dcd.zip")

    job = logic.DCNumPipelineJob(path_in=path_orig,
                                 path_out=path_out,
                                 segmenter_code="torchuni",
                                 segmenter_kwargs={
                                     "model_file": model_file,
                                     "backend": "cudagraphs",
                                     "device": "gpu",
                                     },
                                 )

    dcs = UniversalWorker.get_worker_dedications(job, 2)
    assert len(dcs) == 2
    all = {
        "load_all",
        "segment_images",
        "label_masks",
        "process_labels",
        "extract_features",
    }
    assert (all - {"load_all"}) == set(dcs[0])
    assert (all - {"segment_images"}) == set(dcs[1])
