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


@pytest.mark.parametrize("override_attrs,delete_keys,meas_id", [
    # standard
    [{"experiment:run identifier": "peter"},
     [],
     "peter"],
    # from metadata (input file)
    [{"experiment:run identifier": ""},
     [],
     "d5a40aed-0b6c-0412-e87c-59789fdd28d0"],
    # from metadata (input file)
    [{},
     ["experiment:run identifier"],
     "d5a40aed-0b6c-0412-e87c-59789fdd28d0"],
    # from metadata (manual)
    [{"experiment:time": "12:01",
      "experiment:date": "2024-06-23",
      "setup:identifier": "RC-peter-pan-and-friends",
      },
     ["experiment:run identifier"],
     "f0cfdc6f-93c7-c093-d135-a20f3e5bdbfa"],
    # delete everything, this is the hash of none-none-none
    [{},
     ["setup:identifier", "experiment:time", "experiment:run identifier",
      "setup:identifier"],
     "07d18822-3045-4e6b-da0b-f1a16c80132a"],
])
def test_basin_experiment_identifier_correct(
        override_attrs, delete_keys, meas_id):
    """Make sure the measurement identifier computed by dcnum is correct"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path_orig.with_name("out.rtdc")

    # modify input file
    with h5py.File(path_orig, "a") as h5in:
        for key in override_attrs:
            h5in.attrs[key] = override_attrs[key]
        for key in delete_keys:
            if key in h5in.attrs:
                h5in.attrs.pop(key)

    job = logic.DCNumPipelineJob(path_in=path_orig,
                                 path_out=path_out,
                                 background_code="copy",
                                 segmenter_code="thresh",
                                 segmenter_kwargs={"thresh": -5},
                                 feature_kwargs={"volume": False,
                                                 "haralick": False,
                                                 "brightness": False,
                                                 },
                                 basin_strategy="tap",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path_out) as hout:
        appid = hout.attrs["pipeline:dcnum hash"]
        eri_exp = f"{meas_id}_dcn-{appid[:7]}"
        # this is the actual test
        assert hout.attrs["experiment:run identifier"] == eri_exp


def test_basin_strategy_drain_mapped_input():
    """When basin strategy is "drain", features are mapped from the input

    This test also makes sure that basin index mapping works for input files
    that are opened with the "index_mapping" keyword argument of HDF5Data.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path_out = path_orig.with_name("out.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 data_kwargs={"index_mapping": slice(2, 5)},
                                 basin_strategy="drain",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path_out) as ho, h5py.File(path) as hi:
        assert "image_bg" not in ho["events"]
        assert "image_bg" in ho["basin_events"]
        assert "image" in ho["events"]
        assert "bg_off" in ho["events"]
        assert "deform" in ho["events"]
        assert "basinmap0" in ho["events"]
        basinmap0 = np.array([2, 2, 3, 3, 4, 4, 4])
        assert np.all(ho["events/basinmap0"][:] == basinmap0)
        assert np.all(hi["events/frame"][:][basinmap0]
                      == ho["events/frame"][:])

        for feat in ho["events"]:
            assert len(ho["events"][feat]) == 7


def test_basin_strategy_drain_mapped_input_rollmed():
    """When basin strategy is "drain", features are mapped from the input

    This test also makes sure that basin index mapping works for input files
    that are opened with the "index_mapping" keyword argument of HDF5Data.
    """
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path_out = path_orig.with_name("out.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 background_code="rollmed",
                                 background_kwargs={"kernel_size": 80,
                                                    "batch_size": 150},
                                 data_kwargs={"index_mapping": slice(2, 5)},
                                 basin_strategy="drain",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path_out) as ho, h5py.File(path) as hi:
        assert "image_bg" in ho["events"]
        assert "basin_events" not in ho, "rollmed does not do internal basins"
        assert "bg_off" not in ho["events"], "rollmed does not do bg_off"
        assert "image" in ho["events"]
        assert "deform" in ho["events"]
        assert "basinmap0" in ho["events"]
        basinmap0 = np.array([2, 2, 3, 3, 4, 4, 4])
        assert np.all(ho["events/basinmap0"][:] == basinmap0)
        assert np.all(hi["events/frame"][:][basinmap0]
                      == ho["events/frame"][:])

        for feat in ho["events"]:
            assert len(ho["events"][feat]) == 7


def test_basin_strategy_tap():
    """When basin strategy is "tap", features are mapped from the input"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path_out = path_orig.with_name("out.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 background_kwargs={"kernel_size": 150},
                                 basin_strategy="tap",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path_out) as h5:
        assert h5.attrs["pipeline:dcnum background"] \
            == "sparsemed:k=150^s=1^t=0^f=0.8^o=1"
        assert "image_bg" not in h5["events"]
        assert "image_bg" in h5["basin_events"]
        assert "bg_off" in h5["events"]
        assert "deform" in h5["events"]
        # the rest of the original features are basins!
        assert "time" not in h5["events"]
        assert "image" not in h5["events"]
        for feat in h5["events"]:
            assert len(h5["events"][feat]) == 275
        # The other features are accessed via basins
        hd = read.HDF5Data(h5)
        assert "image" in hd


def test_basin_strategy_tap_rollmed():
    """When basin strategy is "tap", features are mapped from the input"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path_out = path_orig.with_name("out.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 background_code="rollmed",
                                 background_kwargs={"kernel_size": 150,
                                                    "batch_size": 200},
                                 basin_strategy="tap",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(path_out) as h5:
        assert h5.attrs["pipeline:dcnum background"] \
            == "rollmed:k=150^b=200"
        assert "image_bg" in h5["events"]
        assert "basin_events" not in h5, "rollmed does not do internal basins"
        assert "bg_off" not in h5["events"], "rollmed does not do bg_off"
        assert "deform" in h5["events"]
        # the rest of the original features are basins!
        assert "time" not in h5["events"]
        assert "image" not in h5["events"]
        for feat in h5["events"]:
            assert len(h5["events"][feat]) == 280
        # The other features are accessed via basins
        hd = read.HDF5Data(h5)
        assert "image" in hd


def test_basin_relative_path():
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    path_out = path_orig.with_name("out.rtdc")
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 background_kwargs={"kernel_size": 150},
                                 basin_strategy="tap",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    dir_new = path_orig.parent / "another_directory"
    dir_new.mkdir()
    path_new = dir_new / path.name
    path_out_new = dir_new / path_out.name
    path.rename(path_new)
    path_out.rename(path_out_new)

    # Everything should just work, because we have relative paths in the basin.
    with h5py.File(path_out_new) as h5:
        assert h5.attrs["pipeline:dcnum background"] \
            == "sparsemed:k=150^s=1^t=0^f=0.8^o=1"
        assert "image_bg" not in h5["events"]
        assert "image_bg" in h5["basin_events"]
        assert "bg_off" in h5["events"]
        assert "deform" in h5["events"]
        # the rest of the original features are basins!
        assert "time" not in h5["events"]
        assert "image" not in h5["events"]
        for feat in h5["events"]:
            assert len(h5["events"][feat]) == 275
        # The other features are accessed via basins
        hd = read.HDF5Data(h5)
        assert "image" in hd


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
        assert "image" in h5["events"]
        assert "image_bg" not in h5["events"]
        assert "image_bg" in h5["basin_events"]
        for feat in h5["events"]:
            assert len(h5["events"][feat]) == 275

    # now when we do everything again, the pipeline changes
    job2 = logic.DCNumPipelineJob(path_in=path2,
                                  path_out=path2.with_name("final_out.rtdc"),
                                  background_kwargs={"kernel_size": 250},
                                  debug=True)

    with logic.DCNumJobRunner(job=job2) as runner2:
        runner2.run()

    with h5py.File(job2["path_out"]) as h5:
        assert "deform" in h5["events"]
        assert "image" in h5["events"]
        assert "image_bg" not in h5["events"]
        assert "image_bg" in h5["basin_events"]
        assert len(h5["events/deform"]) == 285
        assert h5.attrs["pipeline:dcnum background"] \
            == "sparsemed:k=250^s=1^t=0^f=0.8^o=1"
        for feat in h5["events"]:
            assert len(h5["events"][feat]) == 285


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
        # image_bg is in "events", because the background was copied
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
        # image_bg is in "events", because the background was copied
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
        # image_bg is in "events", because the background was copied
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
        # image_bg is in "events", because the background was copied
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
        # The "peter" feature is also not part of the PROTECTED_FEATURES,
        # so there should never be a "peter" feature from any input file
        # in any output file.
        assert "peter" not in h5["events"]


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
        assert h5.attrs["experiment:run identifier"] == \
            (f"d5a40aed-0b6c-0412-e87c-59789fdd28d0_"
             f"dcn-{pp_hash[:7]}")


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
        assert h5.attrs["experiment:run identifier"] == \
            (f"d5a40aed-0b6c-0412-e87c-59789fdd28d0_"
             f"dcn-{pp_hash[:7]}")


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
        if "image_bg" in h5.get("basin_events", {}):
            has_old_bg = np.all(h5["basin_events/image_bg"][:, 0, 0] == 200)
        else:
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
            image_bg = h5["basin_events/image_bg"]
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
            assert "image_bg" not in hd.h5["events"]
            assert "image_bg" in hd.h5["basin_events"]


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
            assert "image_bg" not in hd.h5["events"]
            assert "image_bg" in hd.h5["basin_events"]

        assert "image_bg" in runner.dtin.h5["basin_events"]

        assert np.all(runner.draw.h5["events/image_bg"][:, 0, 0] == 200)
        assert not np.all(
            runner.dtin.h5["basin_events/image_bg"][:, 0, 0] == 200)
