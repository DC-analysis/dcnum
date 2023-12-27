import threading

import h5py
import numpy as np
import pytest

from dcnum.feat.feat_background import bg_sparse_median

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
@pytest.mark.parametrize("event_count,kernel_size,split_time",
                         [(720, 10, 0.01),
                          (730, 10, 0.01),
                          (720, 11, 0.01),
                          (720, 11, 0.011),
                          ])  # should be independent
def test_median_sparsemend_full(tmp_path, event_count, kernel_size,
                                split_time):
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    # duration and time are hard-coded
    duration = event_count / 3600 * 1.5
    dtime = np.linspace(0, duration, event_count)

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              split_time=split_time,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert len(bic.shared_input_raw) == kernel_size * 5 * 7
        assert bic.kernel_size == kernel_size
        assert bic.duration == duration
        assert np.allclose(bic.time, dtime)
        assert np.allclose(bic.step_times[0], 0)
        assert np.allclose(bic.step_times[1], split_time)
        assert np.allclose(bic.step_times,
                           np.arange(0, duration, split_time))
        # process the data
        assert bic.get_progress() == 0
        bic.process()
        assert bic.get_progress() == 1
    assert output_path.exists()


def test_median_sparsemend_full_with_file(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    dtime = np.linspace(0, 1, 40)
    with h5py.File(path_in, "a") as h5:
        del h5["/events/image_bg"]
        del h5["/events/time"]
        h5["/events/time"] = dtime

    output_path = tmp_path / "test.h5"

    with bg_sparse_median.BackgroundSparseMed(input_data=path_in,
                                              output_path=output_path,
                                              kernel_size=7,
                                              split_time=0.11,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert len(bic.shared_input_raw) == 7 * 80 * 400
        assert bic.kernel_size == 7
        assert bic.duration == 1
        assert np.allclose(bic.time, dtime)
        assert np.allclose(bic.step_times[0], 0)
        assert np.allclose(bic.step_times[1], 0.11)
        assert np.allclose(bic.step_times, np.arange(0, 1, 0.11))
        # process the data
        bic.process()

    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert "image_bg" in h5["/events"]
        assert h5["/events/image_bg"].shape == (40, 80, 400)


def test_median_sparsemend_full_with_file_no_time(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")

    with h5py.File(path_in, "a") as h5:
        del h5["/events/image_bg"]
        del h5["/events/time"]
        del h5["/events/frame"]
        h5["/events/frame"] = np.arange(0, 40000, 1000) + 100
        h5.attrs["imaging:frame rate"] = 5000

    output_path = tmp_path / "test.h5"

    dtime = np.arange(0, 40000, 1000) / 5000

    with bg_sparse_median.BackgroundSparseMed(input_data=path_in,
                                              output_path=output_path,
                                              kernel_size=7,
                                              split_time=0.11,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert len(bic.shared_input_raw) == 7 * 80 * 400
        assert bic.kernel_size == 7
        assert np.allclose(bic.duration, dtime[-1])  # 7.8
        assert np.allclose(bic.time, dtime)
        # process the data
        bic.process()

    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert "image_bg" in h5["/events"]
        assert h5["/events/image_bg"].shape == (40, 80, 400)


def test_median_sparsemend_full_with_file_no_time_no_frame(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")

    with h5py.File(path_in, "a") as h5:
        del h5["/events/image_bg"]
        del h5["/events/time"]
        del h5["/events/frame"]
        h5.attrs["imaging:frame rate"] = 5000

    output_path = tmp_path / "test.h5"

    dtime = np.linspace(0, 40/5000*1.5, 40)

    with bg_sparse_median.BackgroundSparseMed(input_data=path_in,
                                              output_path=output_path,
                                              kernel_size=7,
                                              split_time=0.11,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert len(bic.shared_input_raw) == 7 * 80 * 400
        assert bic.kernel_size == 7
        assert np.allclose(bic.duration, dtime[-1])  # 7.8
        assert np.allclose(bic.time, dtime)
        # process the data
        bic.process()

    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert "image_bg" in h5["/events"]
        assert h5["/events/image_bg"].shape == (40, 80, 400)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_sparsemend_small_file(tmp_path):
    event_count = 34
    kernel_size = 200
    split_time = 0.01
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    # duration and time are hard-coded
    duration = event_count / 3600 * 1.5
    dtime = np.linspace(0, duration, event_count)

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              split_time=split_time,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert len(bic.shared_input_raw) == 34 * 5 * 7
        assert bic.kernel_size == 34
        assert bic.duration == duration
        assert np.allclose(bic.time, dtime)
        assert np.allclose(bic.step_times[0], 0)
        assert np.allclose(bic.step_times[1], split_time)
        assert np.allclose(bic.step_times,
                           np.arange(0, duration, split_time))
        # process the data
        bic.process()
        # even though the actual kernel size is smaller (which is properly
        # logged, the pipeline identifier should have a kernel size of 200.
        # This is good, because it helps to check for reproducibility.
        assert bic.get_ppid() == "sparsemed:k=200^s=0.01^t=0^f=0.8"

    assert output_path.exists()


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_sparsemend_worker(tmp_path):
    event_count = 34
    kernel_size = 200
    split_time = 0.01
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              split_time=split_time,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              num_cpus=1,
                                              ) as bic:
        # make all workers join
        bic.worker_counter.value = -1000
        [w.join() for w in bic.workers]
        bic.worker_counter.value = 0
        # create our own worker
        worker = bg_sparse_median.WorkerSparseMed(
            job_queue=bic.queue,
            counter=bic.worker_counter,
            shared_input=bic.shared_input_raw,
            shared_output=bic.shared_output_raw,
            kernel_size=bic.kernel_size)
        # run the worker in a thread
        thr = threading.Thread(target=worker.run)
        thr.start()
        # request the worker to do its thing
        bic.process()
        bic.worker_counter.value = -1000
        thr.join()

    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert len(h5["events/image_bg"]) == 34
