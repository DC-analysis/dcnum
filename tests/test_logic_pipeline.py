import h5py
import numpy as np

import pytest

from dcnum import logic, read
from dcnum.meta import ppid

from helper_methods import retrieve_data


def test_simple_pipeline():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with read.HDF5Data(path) as hd:
        assert len(hd) == 200, "sanity check"

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

    with logic.DCNumJobRunner(job=job) as runner:
        assert len(runner.draw) == 200
        runner.run()

        assert job["path_out"].exists(), "output file must exist"
        assert runner.path_temp_in.exists(), "tmp input still exists"

    assert not runner.path_temp_in.exists(), "tmp input file mustn't exist"
    assert not runner.path_temp_out.exists(), "tmp out file must not exist"

    with read.HDF5Data(job["path_out"]) as hd:
        assert "image" in hd
        assert "image_bg" in hd
        assert "deform" in hd
        assert "inert_ratio_prnc" in hd
        assert len(hd) == 395

    with h5py.File(job["path_out"]) as h5:
        assert h5.attrs["pipeline:dcnum generation"] == gen_id
        assert h5.attrs["pipeline:dcnum data"] == dat_id
        assert h5.attrs["pipeline:dcnum background"] == bg_id
        assert h5.attrs["pipeline:dcnum segmenter"] == seg_id
        assert h5.attrs["pipeline:dcnum feature"] == feat_id
        assert h5.attrs["pipeline:dcnum gate"] == gate_id
        assert h5.attrs["pipeline:dcnum yield"] == 395
        assert h5.attrs["experiment:event count"] == 395


@pytest.mark.parametrize("attr,oldval,newbg", [
    # Changes that trigger computation of new background
    ["pipeline:dcnum generation", "1", True],
    ["pipeline:dcnum data", "hdf:p=0.2656", True],
    ["pipeline:dcnum background", "sparsemed:k=100^s=1^t=0^f=0.8", True],
    # Changes that don't trigger background computation
    ["pipeline:dcnum segmenter", "thresh:t=-1:cle=1^f=1^clo=2", False],
    ["pipeline:dcnum feature", "thresh:t=-1:cle=1^f=1^clo=2", False],
    ["pipeline:dcnum gate", "norm:o=0^s=5", False],
    ["pipeline:dcnum yield", 5000, False],
    ["pipeline:dcnum hash", "asdasd", False],
])
def test_recomputation_of_background_metadata_changed(attr, oldval, newbg):
    """Recompute background when one of these metadata change

    Background computation is only triggered when the following
    metadata do not match:
    - pipeline:dcnum generation
    - pipeline:dcnum data
    - pipeline:dcnum background
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")

    # Create a concatenated output file
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with h5py.File(path, "a") as h5:
        # marker for identifying recomputation of background
        h5["events/image_bg"][:, 0, 0] = 200

        # Set the default values
        h5.attrs["pipeline:dcnum generation"] = ppid.DCNUM_PPID_GENERATION
        h5.attrs["pipeline:dcnum data"] = "hdf:p=0.2645"
        h5.attrs["pipeline:dcnum background"] = "sparsemed:k=200^s=1^t=0^f=0.8"
        h5.attrs["pipeline:dcnum segmenter"] = "thresh:t=-6:cle=1^f=1^clo=2"
        h5.attrs["pipeline:dcnum feature"] = "legacy:b=1^h=1"
        h5.attrs["pipeline:dcnum gate"] = "norm:o=0^s=10"
        h5.attrs["pipeline:dcnum yield"] = h5["events/image"].shape[0]

        if attr == "pipeline:dcnum hash":
            # set just the pipeline hash
            h5.attrs["pipeline:dcnum hash"] = oldval
        else:
            # set the test value
            h5.attrs[attr] = oldval
            # compute a valid pipeline hash
            job = logic.DCNumPipelineJob(path_in=path_orig)
            _, h5.attrs["pipeline:dcnum hash"] = job.get_ppid(ret_hash=True)

    job = logic.DCNumPipelineJob(path_in=path,
                                 debug=True)

    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(job["path_out"]) as h5:
        assert h5.attrs[attr] != oldval, "sanity check"
        has_old_bg = np.all(h5["events/image_bg"][:, 0, 0] == 200)
        assert not has_old_bg == newbg


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

    with logic.DCNumJobRunner(job=job) as runner:
        assert not runner.path_temp_in.exists()
        runner.task_background()
        assert runner.path_temp_in.exists(), "running bg task creates basin"
        assert not runner.path_temp_out.exists()

        with h5py.File(runner.path_temp_in) as h5:
            assert "image" not in h5["events"], "image is in the basin file"
            image_bg = h5["events/image_bg"]
            assert image_bg.attrs["dcnum ppid background"] == bg_id
            assert image_bg.attrs["dcnum ppid generation"] == gen_id

        with read.HDF5Data(runner.path_temp_in) as hd:
            assert "image" in hd, "image is in the basin file"
            assert "image_bg" in hd
            assert np.allclose(np.mean(hd["image_bg"][0]),
                               180.5675625,
                               rtol=0, atol=0.01)


def test_task_background_data_properties():
    """.draw and .dtin should return reasonable values"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with h5py.File(path, "a") as h5:
        # marker for identifying recomputation of background
        h5["events/image_bg"][:, 0, 0] = 200

    job = logic.DCNumPipelineJob(path_in=path, debug=True)

    with logic.DCNumJobRunner(job=job) as runner:
        runner.task_background()

        assert runner._data_temp_in is None
        assert runner.path_temp_in.exists()

        with read.HDF5Data(runner.path_temp_in) as hd:
            assert "image_bg" in hd
            assert "image_bg" in hd.h5["events"]

        assert "image_bg" in runner.dtin.h5["events"]

        assert np.all(runner.draw.h5["events/image_bg"][:, 0, 0] == 200)
        assert not np.all(runner.dtin.h5["events/image_bg"][:, 0, 0] == 200)
