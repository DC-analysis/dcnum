import time

import h5py
import numpy as np

import pytest

from dcnum import logic, read, write
from dcnum.meta import ppid

from helper_methods import retrieve_data


def get_log(hd: read.HDF5Data,
            startswith: str):
    """Return log entry that starts with `startswith`"""
    for key in hd.logs:
        if key.startswith(startswith):
            return hd.logs[key]
    else:
        raise KeyError(f"Log starting with {startswith} not found!")


def test_chained_pipeline():
    """Test running two pipelines consecutively"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path2,
                                 background_kwargs={"kernel_size": 150},
                                 debug=True)

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path2) as h5:
        assert h5.attrs["pipeline:dcnum background"] \
               == "sparsemed:k=150^s=1^t=0^f=0.8^o=1"

    # now when we do everything again, not a things should be done
    job2 = logic.DCNumPipelineJob(path_in=path2,
                                  path_out=path2.with_name("final_out.rtdc"),
                                  background_kwargs={"kernel_size": 250},
                                  debug=True)
    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()

    with h5py.File(job2["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" in h5["events"]
        assert len(h5["events/deform"]) == 285
        assert h5.attrs["pipeline:dcnum background"] \
               == "sparsemed:k=250^s=1^t=0^f=0.8^o=1"


def test_duplicate_pipeline():
    """Test running the same pipeline twice

    When the pipeline is run on a file that has been run with the same
    pipeline identifier, then we do not run the pipeline. Instead, we
    copy the data from the first file.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    job = logic.DCNumPipelineJob(
        path_in=path,
        path_out=path2,
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    assert job.kwargs["data_kwargs"].get("index_mapping") is None

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    # Sanity checks for initial job
    with read.HDF5Data(job["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" in logdat
        assert "Finished background computation" in logdat
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    # get the first image for reference
    with h5py.File(path) as h5:
        im0 = h5["/events/image"][0]

    # remove all logs just to be sure nothing interferes
    with h5py.File(path2, "a") as h5:
        assert h5.attrs["pipeline:dcnum mapping"] == "0"
        assert len(h5["events/deform"]) == 395
        del h5["logs"]

    # now when we do everything again, not a thing should be done
    job2 = logic.DCNumPipelineJob(
        path_in=path2,
        path_out=path2.with_name("final_out.rtdc"),
        no_basins_in_output=True,
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()
    # Real check for second run (not the `not`s [sic]!)
    with read.HDF5Data(job2["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" not in logdat
        assert "Finished background computation" not in logdat
        assert "Starting segmentation and feature extraction" not in logdat
        assert "Flushing data to disk" not in logdat
        assert "Finished segmentation and feature extraction" not in logdat

    with h5py.File(job2["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" in h5["events"]
        assert len(h5["events/deform"]) == 395
        assert h5.attrs["pipeline:dcnum mapping"] == "0"
        assert np.all(h5["events/image"][0] == im0)


def test_duplicate_pipeline_redo_index_mapping():
    """Test running the same pipeline twice

    When the pipeline is run on a file that has been run with the same
    pipeline identifier, then we do not run the pipeline. Instead, we
    copy the data from the first file.

    However, if something is odd, such as index mapping defined in the
    pipeline then redo the computations.
    This is the purpose of this test.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    job = logic.DCNumPipelineJob(
        path_in=path,
        path_out=path2,
        data_kwargs={"index_mapping": 10},
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    assert job.kwargs["data_kwargs"].get("index_mapping") == 10

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    # Sanity checks for initial job
    with read.HDF5Data(job["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" in logdat
        assert "Finished background computation" in logdat
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    with h5py.File(path2, "a") as h5:
        # sanity checks
        assert h5.attrs["pipeline:dcnum mapping"] == "10"
        assert len(h5["events/deform"]) == 24
        assert h5.attrs["pipeline:dcnum yield"] == 24
        # remove all logs just to be sure nothing interferes
        del h5["logs"]
        # Modify the yield, triggering a new pipeline run
        h5.attrs["pipeline:dcnum yield"] = 111111

    # now when we do everything again, not a thing should be done
    job2 = logic.DCNumPipelineJob(
        path_in=path2,
        path_out=path2.with_name("final_out.rtdc"),
        no_basins_in_output=True,
        data_kwargs={"index_mapping": 10},
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()
    # Real check for second run (not the `not`s [sic]!)
    with read.HDF5Data(job2["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        # Background computation is not repeated
        assert "Starting background computation" not in logdat
        assert "Finished background computation" not in logdat
        # Segmentation is repeated
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    with h5py.File(job2["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" in h5["events"]
        # We have not 24 here, because the index mapping enumerates events,
        # not frames.
        assert len(h5["events/deform"]) == 11
        assert h5.attrs["pipeline:dcnum mapping"] == "10"
        assert h5.attrs["pipeline:dcnum yield"] == 11


def test_duplicate_pipeline_redo_yield():
    """Test running the same pipeline twice

    When the pipeline is run on a file that has been run with the same
    pipeline identifier, then we do not run the pipeline. Instead, we
    copy the data from the first file.

    However, if something is odd, such as the yield of the pipeline not
    matching the data in the output file, then redo the computations.
    This is the purpose of this test.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    job = logic.DCNumPipelineJob(
        path_in=path,
        path_out=path2,
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    assert job.kwargs["data_kwargs"].get("index_mapping") is None

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    # Sanity checks for initial job
    with read.HDF5Data(job["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" in logdat
        assert "Finished background computation" in logdat
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    with h5py.File(path2, "a") as h5:
        # sanity checks
        assert h5.attrs["pipeline:dcnum mapping"] == "0"
        assert len(h5["events/deform"]) == 395
        assert h5.attrs["pipeline:dcnum yield"] == 395
        # remove all logs just to be sure nothing interferes
        del h5["logs"]
        # Modify the yield, triggering a new pipeline run
        h5.attrs["pipeline:dcnum yield"] = 111111

    # now when we do everything again, not a thing should be done
    job2 = logic.DCNumPipelineJob(
        path_in=path2,
        path_out=path2.with_name("final_out.rtdc"),
        no_basins_in_output=True,
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()
    # Real check for second run (not the `not`s [sic]!)
    with read.HDF5Data(job2["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        # Background computation is not repeated
        assert "Starting background computation" not in logdat
        assert "Finished background computation" not in logdat
        # Segmentation is repeated
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    with h5py.File(job2["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" in h5["events"]
        assert len(h5["events/deform"]) == 395
        assert h5.attrs["pipeline:dcnum mapping"] == "0"
        assert h5.attrs["pipeline:dcnum yield"] == 395


@pytest.mark.parametrize("index_mapping,size,mapping_out", [
    (5, 11, "5"),
    (slice(3, 5, None), 6, "3-5-n"),
    ([3, 5, 6, 7], 7, "h-6e582938"),
])
def test_index_mapping_pipeline(index_mapping, size, mapping_out):
    """Test running the same pipeline twice

    When the pipeline is run on a file with the same pipeline
    identifier, data are just copied over. Nothing much fancy else.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    job = logic.DCNumPipelineJob(
        path_in=path,
        path_out=path2,
        data_kwargs={"index_mapping": index_mapping},
        background_code="copy",
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -6,
                          "kwargs_mask": {"closing_disk": 0}},
        debug=True)
    assert job.kwargs["data_kwargs"]["index_mapping"] == index_mapping

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    # Sanity checks for initial job
    with read.HDF5Data(job["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" in logdat
        assert "Finished background computation" in logdat
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat

    with h5py.File(job["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" in h5["events"]
        assert len(h5["events/deform"]) == size
        assert h5.attrs["pipeline:dcnum mapping"] == mapping_out


def test_duplicate_transfer_basin_data():
    """task_transfer_basin_data should not copy basin data from input"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path2 = path.with_name("path_intermediate.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with write.HDF5Writer(path) as hw:
        path_basin = path.with_name("data_basin.rtdc")
        # store the basin in the original file
        hw.store_basin(name="test", paths=[path_basin], features=["peter"])
        # store the peter data in the basin
        with h5py.File(path_basin, "a") as hb:
            hb["events/peter"] = 3.14 * hw.h5["events/deform"][:]

    job = logic.DCNumPipelineJob(path_in=path, path_out=path2, debug=True)

    # perform the initial pipeline
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path2) as h5:
        # The feature comes from the input file and will *not* be copied.
        assert "peter" not in h5["events"]

    # Now we change things. The input file now contains basins that should
    # also be present in the output file. Using the `no_basins_in_output`
    # option, the feature data from the input should actually be stored
    # in the output file.
    with write.HDF5Writer(path2) as hw2:
        del hw2.h5["logs"]  # remove logs
        path_basin2 = path2.with_name("data_basin_2.rtdc")
        # store the basin in the original file
        hw2.store_basin(name="test", paths=[path_basin2], features=["peter2"])
        # store the peter data in the basin
        with h5py.File(path_basin2, "a") as hb2:
            hb2["events/peter2"] = 3.15 * hw2.h5["events/deform"][:]

    job2 = logic.DCNumPipelineJob(path_in=path2,
                                  path_out=path2.with_name("final_out.rtdc"),
                                  no_basins_in_output=True,
                                  debug=True)
    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()

    with h5py.File(job2["path_out"]) as h52:
        # The feature comes from the input file and *will* be copied.
        assert "peter2" in h52["events"]


def test_error_file_exists():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    path_out = path.with_name("test_out.rtdc")
    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 debug=True)
    path_out.touch()
    with logic.DCNumJobRunner(job=job) as runner:
        with pytest.raises(FileExistsError, match=path_out.name):
            runner.run()


def test_error_file_exists_in_thread():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    path_out = path.with_name("test_out.rtdc")
    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 debug=True)
    path_out.touch()
    runner = logic.DCNumJobRunner(job=job)
    runner.start()
    runner.join()
    assert runner.error_tb is not None
    assert "FileExistsError" in runner.error_tb


def test_error_pipeline_log_file_remains():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path.with_name("test1.rtdc"),
                                 debug=True)

    # control
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    assert not runner.path_log.exists(), "no log file expected"

    job2 = logic.DCNumPipelineJob(path_in=path,
                                  path_out=path.with_name("test2.rtdc"),
                                  debug=True)

    with pytest.raises(ValueError, match="My Test Error In The Context"):
        with logic.DCNumJobRunner(job=job2) as runner:
            runner.run()
            raise ValueError("My Test Error In The Context")
    # log file should still be there
    assert runner.path_log.exists(), "log file expected"


def test_get_status():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass
    job = logic.DCNumPipelineJob(path_in=path, debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        assert runner.get_status() == {
            "progress": 0,
            "segm rate": 0,
            "state": "init",
        }
        runner.run()
        final_status = runner.get_status()
        assert final_status["progress"] == 1
        assert final_status["segm rate"] > 0
        assert final_status["state"] == "done"


def test_logs_in_pipeline():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path, debug=True)

    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with read.HDF5Data(job["path_out"]) as hd:
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "Starting background computation" in logdat
        assert "Finished background computation" in logdat
        assert "Starting segmentation and feature extraction" in logdat
        assert "Flushing data to disk" in logdat
        assert "Finished segmentation and feature extraction" in logdat
        assert "Run duration" in logdat

        jobdat = " ".join(get_log(hd, time.strftime("dcnum-job-%Y")))
        assert "identifiers" in jobdat


def test_no_events_found():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    # Set image data to zero (no events)
    with h5py.File(path, "a") as h5:
        zeros = np.zeros_like(h5["events/image"][:])
        del h5["events/image"]
        h5["events/image"] = zeros

    job = logic.DCNumPipelineJob(path_in=path, debug=True)

    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with read.HDF5Data(job["path_out"]) as hd:
        assert len(hd) == 0
        # Check the logs
        logdat = " ".join(get_log(hd, time.strftime("dcnum-log-%Y")))
        assert "No events found" in logdat


@pytest.mark.parametrize("debug", [True, False])
def test_simple_pipeline(debug):
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with read.HDF5Data(path) as hd:
        assert len(hd) == 200, "sanity check"

    # this is the default pipeline
    gen_id = ppid.DCNUM_PPID_GENERATION
    dat_id = "hdf:p=0.2645^i=0"
    bg_id = "sparsemed:k=200^s=1^t=0^f=0.8^o=1"
    seg_id = "thresh:t=-6:cle=1^f=1^clo=0"
    feat_id = "legacy:b=1^h=1^v=1"
    gate_id = "norm:o=0^s=10"
    jobid = "|".join([gen_id, dat_id, bg_id, seg_id, feat_id, gate_id])

    job = logic.DCNumPipelineJob(
        path_in=path,
        debug=debug,
        segmenter_kwargs={"kwargs_mask": {"closing_disk": 0}},
    )
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
        assert hd["nevents"][0] == 1
        assert hd["nevents"][1] == 2
        assert np.all(hd["nevents"][:11] == [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
        assert np.all(hd["frame"][:11] == [1, 2, 2, 4, 4, 5, 5, 5, 6, 6, 6])
        assert np.allclose(hd["area_um"][2], 36.694151125,
                           atol=0.5, rtol=0)
        assert np.allclose(hd["deform"][2], 0.29053587689236526,
                           atol=0.001, rtol=0)

    with h5py.File(job["path_out"]) as h5:
        assert h5.attrs["pipeline:dcnum generation"] == gen_id
        assert h5.attrs["pipeline:dcnum data"] == dat_id
        assert h5.attrs["pipeline:dcnum background"] == bg_id
        assert h5.attrs["pipeline:dcnum segmenter"] == seg_id
        assert h5.attrs["pipeline:dcnum feature"] == feat_id
        assert h5.attrs["pipeline:dcnum gate"] == gate_id
        assert h5.attrs["pipeline:dcnum yield"] == 395
        assert h5.attrs["experiment:event count"] == 395
        pp_hash = h5.attrs["pipeline:dcnum hash"]
        # test for general metadata
        assert h5.attrs["experiment:sample"] == "data"
        assert h5.attrs["experiment:date"] == "2022-04-21"
        assert h5.attrs["experiment:run identifier"] == f"dcn-{pp_hash[:7]}"


@pytest.mark.parametrize("debug", [True, False])
def test_simple_pipeline_no_offset_correction(debug):
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with read.HDF5Data(path) as hd:
        assert len(hd) == 200, "sanity check"

    # this is the default pipeline
    gen_id = ppid.DCNUM_PPID_GENERATION
    dat_id = "hdf:p=0.2645^i=0"
    bg_id = "sparsemed:k=200^s=1^t=0^f=0.8^o=0"
    seg_id = "thresh:t=-6:cle=1^f=1^clo=0"
    feat_id = "legacy:b=1^h=1^v=1"
    gate_id = "norm:o=0^s=10"
    jobid = "|".join([gen_id, dat_id, bg_id, seg_id, feat_id, gate_id])

    job = logic.DCNumPipelineJob(
        path_in=path,
        debug=debug,
        background_kwargs={"offset_correction": False},
        segmenter_kwargs={"kwargs_mask": {"closing_disk": 0}},
    )
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
        assert hd["nevents"][0] == 2
        assert np.all(hd["nevents"][:11] == [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        assert np.all(hd["frame"][:11] == [1, 1, 2, 2, 4, 4, 5, 5, 5, 6, 6])
        assert np.allclose(hd["area_um"][3], 36.694151125,
                           atol=0.5, rtol=0)
        assert np.allclose(hd["deform"][3], 0.29053587689236526,
                           atol=0.001, rtol=0)

    with h5py.File(job["path_out"]) as h5:
        assert h5.attrs["pipeline:dcnum generation"] == gen_id
        assert h5.attrs["pipeline:dcnum data"] == dat_id
        assert h5.attrs["pipeline:dcnum background"] == bg_id
        assert h5.attrs["pipeline:dcnum segmenter"] == seg_id
        assert h5.attrs["pipeline:dcnum feature"] == feat_id
        assert h5.attrs["pipeline:dcnum gate"] == gate_id
        assert h5.attrs["pipeline:dcnum yield"] == 395
        assert h5.attrs["experiment:event count"] == 395
        pp_hash = h5.attrs["pipeline:dcnum hash"]
        # test for general metadata
        assert h5.attrs["experiment:sample"] == "data"
        assert h5.attrs["experiment:date"] == "2022-04-21"
        assert h5.attrs["experiment:run identifier"] == f"dcn-{pp_hash[:7]}"


def test_simple_pipeline_in_thread():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path, debug=True)

    # The context manager's __exit__ and runner.join both call runner.close()
    with logic.DCNumJobRunner(job=job) as runner:
        runner.start()
        runner.join()


@pytest.mark.parametrize("attr,oldval,newbg", [
    # Changes that trigger computation of new background
    ["pipeline:dcnum generation", "1", True],
    ["pipeline:dcnum data", "hdf:p=0.2656^i=0", True],
    ["pipeline:dcnum background", "sparsemed:k=100^s=1^t=0^f=0.8^o=1", True],
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
        h5.attrs["pipeline:dcnum data"] = "hdf:p=0.2645^i=0"
        h5.attrs["pipeline:dcnum background"] = \
            "sparsemed:k=200^s=1^t=0^f=0.8^o=1"
        h5.attrs["pipeline:dcnum segmenter"] = "thresh:t=-6:cle=1^f=1^clo=2"
        h5.attrs["pipeline:dcnum feature"] = "legacy:b=1^h=1^v=1"
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
    dat_id = "hdf:p=0.2645^i=0"
    bg_id = "sparsemed:k=200^s=1^t=0^f=0.8^o=1"
    seg_id = "thresh:t=-6:cle=1^f=1^clo=2"
    feat_id = "legacy:b=1^h=1^v=1"
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


def test_task_background_close_input_file_on_demand():
    """Tests whether the background task can close the input file"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    with h5py.File(path, "a") as h5:
        # marker for identifying recomputation of background
        h5["events/image_bg"][:, 0, 0] = 200

    job = logic.DCNumPipelineJob(path_in=path, debug=True)

    with logic.DCNumJobRunner(job=job) as runner:
        assert runner.dtin  # access the temporary input file
        assert runner._data_temp_in is not None

        runner.task_background()

        assert runner._data_temp_in is None
        assert runner.path_temp_in.exists()

        with read.HDF5Data(runner.path_temp_in) as hd:
            assert "image_bg" in hd
            assert "image_bg" in hd.h5["events"]


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
