import h5py
import numpy as np

from dcnum import logic, read
from dcnum.meta import ppid

from helper_methods import retrieve_data


def test_simple_pipeline():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    # this is the default pipeline
    gen_id = ppid.DCNUM_PPID_GENERATION
    dat_id = "hdf:p=0.2645"
    bg_id = "sparsemed:k=200^s=1^t=0^f=0.8"
    seg_id = "thresh:t=-6:cle=1^f=1^clo=2"
    feat_id = "legacy:b=1^h=1"
    gate_id = "norm:o=0^s=10"
    jobid = "|".join([gen_id, dat_id, bg_id, seg_id, feat_id, gate_id])

    job = logic.DCNumPipelineJob(path_in=path, debug=True)
    assert job.get_ppid() == jobid
    runner = logic.DCNumJobRunner(job=job)
    assert len(runner.data) == 200
    runner.run()

    assert job["path_out"].exists(), "output file must exist"
    assert not runner.path_temp_in.exists(), "temp input file must not exist"
    assert not runner.path_temp_out.exists(), "temp out file must not exist"

    with read.HDF5Data(job["path_out"]) as hd:
        assert "image" in hd
        assert "image_bg" in hd
        assert "deform" in hd
        assert "inert_ratio_prnc" in hd
        assert len(hd) == 390

    with h5py.File(job["path_out"]) as h5:
        assert h5.attrs["pipeline:dcnum generation"] == gen_id
        assert h5.attrs["pipeline:dcnum data"] == dat_id
        assert h5.attrs["pipeline:dcnum background"] == bg_id
        assert h5.attrs["pipeline:dcnum segmenter"] == seg_id
        assert h5.attrs["pipeline:dcnum feature"] == feat_id
        assert h5.attrs["pipeline:dcnum gate"] == gate_id
        assert h5.attrs["pipeline:dcnum yield"] == 390
        assert h5.attrs["experiment:event count"] == 390


def test_task_background():
    """Just test this one task, without running the full job"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with read.HDF5Data(path_orig) as hd:
        assert "image" in hd
        assert "image_bg" in hd
        assert np.allclose(np.mean(hd["image"][0]), 180.772375)
        assert np.allclose(np.mean(hd["image_bg"][0]),
                           180.4453125,
                           rtol=0,
                           atol=0.01)

    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    # this is the default pipeline
    gen_id = ppid.DCNUM_PPID_GENERATION
    dat_id = "hdf:p=0.2645"
    bg_id = "sparsemed:k=200^s=1^t=0^f=0.8"
    seg_id = "thresh:t=-6:cle=1^f=1^clo=2"
    feat_id = "legacy:b=1^h=1"
    gate_id = "norm:o=0^s=10"
    jobid = "|".join([gen_id, dat_id, bg_id, seg_id, feat_id, gate_id])

    job = logic.DCNumPipelineJob(path_in=path, debug=True)
    assert job.get_ppid() == jobid
    runner = logic.DCNumJobRunner(job=job)

    assert not runner.path_temp_in.exists()
    runner.task_background()
    assert runner.path_temp_in.exists()
    assert not runner.path_temp_out.exists()

    with h5py.File(runner.path_temp_in) as h5:
        image_bg = h5["events/image_bg"]
        assert image_bg.attrs["dcnum ppid background"] == bg_id
        assert image_bg.attrs["dcnum ppid generation"] == gen_id

    with read.HDF5Data(runner.path_temp_in) as hd:
        assert "image" not in hd, "we have not created a basin dataset"
        assert "image_bg" in hd
        assert np.allclose(np.mean(hd["image_bg"][0]),
                           180.5675625,
                           rtol=0, atol=0.01)
