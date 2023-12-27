import threading

import h5py
import numpy as np
import pytest

from dcnum.feat.feat_background import bg_roll_median


def test_compute_median_for_slice():
    # events in shared arrays: 100
    # image shape: 5 * 7
    shared_input = np.arange(5*7).reshape(1, 5*7) * np.ones((100, 1))
    assert shared_input.size == 100 * 5 * 7
    assert np.all(shared_input[:, 0] == 0)  # pixel 0
    assert np.all(shared_input[:, 1] == 1)  # pixel 1

    shared_output = np.zeros((100, 5 * 7))

    job_slice = slice(1, 4)

    batch_size = 90
    kernel_size = 10

    bg_roll_median.compute_median_for_slice(
        shared_input=shared_input,
        shared_output=shared_output,
        job_slice=job_slice,
        output_size=batch_size,
        kernel_size=kernel_size,
    )

    # compare input and output at batch size
    assert np.all(shared_input[:90, 1:4] == shared_input[:90, 1:4])

    # sanity check with boundary values
    comp_in_b = shared_input.reshape(-1, 100)[1:4, 90:]
    comp_out_b = shared_output.reshape(-1, 100)[1:4, 90:]
    assert not np.all(comp_in_b == comp_out_b)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
@pytest.mark.parametrize("event_count", [720, 730])  # should be independent
def test_median_map_iterator(tmp_path, event_count):
    output_path = tmp_path / "test.h5"
    # batch size: 90
    # image shape: 5 * 7
    # kernel size: 10
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_roll_median.BackgroundRollMed(input_data=input_data,
                                          output_path=output_path,
                                          kernel_size=10,
                                          batch_size=90,
                                          ) as mic:
        assert len(mic.shared_input_raw) == (10 + 90) * 5 * 7

        jobs = list(mic.map_iterator())
    assert len(jobs) == 7
    assert jobs[1].start == 1 * 5
    assert jobs[1].stop == 2 * 5
    assert jobs[2].start == 2 * 5
    assert jobs[2].stop == 3 * 5
    assert jobs[6].stop == 7 * 5


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
@pytest.mark.parametrize("event_count", [720, 730])
def test_median_process_next_batch(tmp_path, event_count):
    output_path = tmp_path / "test.h5"
    # batch size: 90
    # image shape: 5 * 7
    # kernel size: 10
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    input_data = np.array(input_data, dtype=np.uint8)
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_roll_median.BackgroundRollMed(input_data=input_data,
                                          output_path=output_path,
                                          kernel_size=10,
                                          batch_size=90,
                                          ) as mic:
        assert len(mic.shared_input_raw) == (10 + 90) * 5 * 7

        assert mic.current_batch == 0
        assert mic.get_progress() == 0
        mic.process_next_batch()
        assert mic.get_progress() == 90 / event_count

        assert mic.current_batch == 1

    with h5py.File(output_path) as h5:
        ds = h5["events/image_bg"]
        assert ds.shape == (event_count, 5, 7)
        assert np.all(ds[90:] == 0), "not processed"
        assert np.all(ds[:90, 0, 0] == 0)
        assert np.all(ds[:90, 0, 1] == 1)
        assert np.all(ds[:90, 0, 2] == 2)
        assert np.all(ds[:90, 1, 0] == 7)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
@pytest.mark.parametrize("event_count, chunk_count", [[720, 8], [730, 9]])
def test_median_process_full(tmp_path, event_count, chunk_count):
    output_path = tmp_path / "test.h5"
    # batch size: 90
    # image shape: 5 * 7
    # kernel size: 10
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    input_data = np.array(input_data, dtype=np.uint8)
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_roll_median.BackgroundRollMed(input_data=input_data,
                                          output_path=output_path,
                                          kernel_size=10,
                                          batch_size=90,
                                          ) as mic:
        assert len(mic.shared_input_raw) == (10 + 90) * 5 * 7
        # output array is smaller
        assert len(mic.shared_output_raw) == 90 * 5 * 7

        assert mic.current_batch == 0
        mic.process()
        assert mic.current_batch == chunk_count

    with h5py.File(output_path) as h5:
        ds = h5["events/image_bg"]
        assert ds.shape == (event_count, 5, 7)
        assert np.all(ds[:90, 0, 0] == 0)
        assert np.all(ds[:90, 0, 1] == 1)
        assert np.all(ds[:90, 0, 2] == 2)
        assert np.all(ds[:90, 1, 0] == 7)
        assert np.all(ds[690:, 0, 0] == 0)
        assert np.all(ds[690:, 0, 1] == 1)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_rollmed_small_file(tmp_path):
    event_count = 34
    kernel_size = 200
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))
    assert np.median(np.arange(5*7)) == 17, "sanity check"

    with pytest.raises(ValueError, match="Cannot compute background"):
        with bg_roll_median.BackgroundRollMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              batch_size=1000,
                                              ):
            pass


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_rollmed_worker(tmp_path):
    event_count = 720
    kernel_size = 100
    batch_size = 200
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_roll_median.BackgroundRollMed(input_data=input_data,
                                          output_path=output_path,
                                          kernel_size=kernel_size,
                                          batch_size=batch_size,
                                          ) as bic:
        # make all workers join
        bic.worker_counter.value = -1000
        [w.join() for w in bic.workers]
        bic.worker_counter.value = 0
        # create our own worker
        worker = bg_roll_median.WorkerRollMed(
            job_queue=bic.queue,
            counter=bic.worker_counter,
            shared_input=bic.shared_input_raw,
            shared_output=bic.shared_output_raw,
            kernel_size=bic.kernel_size,
            batch_size=batch_size
        )
        # run the worker in a thread
        thr = threading.Thread(target=worker.run)
        thr.start()
        # request the worker to do its thing
        bic.process()
        bic.worker_counter.value = -1000
        thr.join()

    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert len(h5["events/image_bg"]) == 720
        assert np.median(h5["events/image_bg"][0]) == 17
