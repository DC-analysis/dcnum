import threading

import h5py
import numpy as np
import pytest

from dcnum.feat.feat_background import bg_sparse_median
from dcnum import read

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


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_sparsemend_full_bg_off(tmp_path):
    """Test computation of bg_off"""
    event_count = 720
    kernel_size = 10
    split_time = 0.01
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    # Add images that have an offset which should be corrected in bg_off
    input_data[3] += 1
    input_data[25] += 2
    input_data[101] += 3
    bg_off_exp = np.zeros(event_count)
    bg_off_exp[3] = 1
    bg_off_exp[25] = 2
    bg_off_exp[101] = 3
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              split_time=split_time,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              offset_correction=True
                                              ) as bic:
        assert len(bic.shared_input_raw) == kernel_size * 5 * 7
        assert bic.kernel_size == kernel_size
        # process the data
        bic.process()
    assert output_path.exists()
    with h5py.File(output_path) as h5:
        assert np.all(bg_off_exp == h5["events/bg_off"][:])

    with read.HDF5Data(output_path) as hd:
        # This is basically the definition of bg_off (a correction factor)
        assert np.all(
            input_data - hd["image_bg"] - hd["bg_off"].reshape(-1, 1, 1) == 0)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_median_sparsemend_full_internal_image_bg(tmp_path):
    """Test computation of internal image_bg feature"""
    event_count = 720
    kernel_size = 10
    split_time = 0.01
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    # Add images that have an offset which should be corrected in bg_off,
    # but be ignored by the median background computation.
    input_data[3] += 1
    input_data[25] += 2
    input_data[101] += 3
    # Add a few `kernel_size` images that have a higher value.
    # duration and time are hard-coded
    duration = event_count / 3600 * 1.5
    dtime = np.linspace(0, duration, event_count)
    start = np.argmin(np.abs(dtime - 0.15))
    stop = np.argmin(np.abs(dtime - 0.16))
    input_data[start:stop] += 4
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=kernel_size,
                                              split_time=split_time,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              offset_correction=True
                                              ) as bic:
        assert len(bic.shared_input_raw) == kernel_size * 5 * 7
        assert bic.kernel_size == kernel_size
        # process the data
        bic.process()
    assert output_path.exists()

    with read.HDF5Data(output_path) as hd:
        bg0 = hd["image_bg"][0]
        # sanity checks (bg_off contains the offset)
        assert np.all(bg0 == hd["image_bg"][3])
        assert np.all(bg0 == hd["image_bg"][25])
        assert np.all(bg0 == hd["image_bg"][101])
        # Real test. The interval size is really dependent on the timing,
        # the actual interval is bigger than 10.
        for ii in range(start-5, start+5):
            assert np.all(bg0 + 4 == hd["image_bg"][ii])


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
        assert "image_bg" not in h5["/events"]
        assert "image_bg" in h5["/basin_events"]
        assert h5["/basin_events/image_bg"].shape == (8, 80, 400)


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
        assert "image_bg" not in h5["/events"]
        assert "image_bg" in h5["/basin_events"]
        assert h5["/basin_events/image_bg"].shape == (57, 80, 400)


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
        assert "image_bg" not in h5["/events"]
        assert "image_bg" in h5["/basin_events"]
        assert h5["/basin_events/image_bg"].shape == (1, 80, 400)


def test_median_sparsemed_internal_basin(tmp_path):
    """Make sure the internal basin is computed correctly"""
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_in = path.with_name("concatenated.rtdc")

    dtime = np.linspace(0, 1, 400, endpoint=False)
    with read.concatenated_hdf5_data([path] * 10, path_out=path_in) as hd:
        pass
    with h5py.File(path_in, "a") as h5:
        # reset background
        del h5["events/image_bg"]
        # limit time to 1 second
        h5["events/time"] = dtime
        # rewrite the image data so that we have an artificial offset
        image = h5["events/image"][:]
        for ii in range(10):
            image[40*ii:40*(ii+1)] += ii
        del h5["events/image"]
        h5["events/image"] = image

    path_out = tmp_path / "test.h5"

    with bg_sparse_median.BackgroundSparseMed(
            input_data=path_in,
            output_path=path_out,
            # These parameters are chosen such that the background images
            # stored in the internal basin will have an offset of 1.
            kernel_size=20,
            split_time=0.1,
            thresh_cleansing=0,
            frac_cleansing=.8,
            ) as bic:
        # sanity checks
        assert len(bic.shared_input_raw) == 20 * 80 * 400
        assert bic.kernel_size == 20
        assert bic.duration == 0.9975
        assert np.allclose(bic.time, dtime)
        # process the data
        bic.process()

    assert path_out.exists()
    with h5py.File(path) as h5:
        imref = np.partition(h5["events/image"][:20], kth=10, axis=0)[10]

    with h5py.File(path_out) as h5:
        assert "image_bg" in h5["/basin_events"]
        image_bg_internal = h5["/basin_events/image_bg"][:]
        # The first image should be identical to what we expect from the
        # regular background correction.
        im0 = image_bg_internal[0]
        assert np.all(im0 == imref)
        # The other images should be the same with a constant offset.
        for ii in range(10):
            # We sometimes get spurious differences between the images,
            # that's why there is a "4" and not a "0". But it shows that
            # the background correction works in principle (If things did
            # not work, we would have differences in the order of 80*320).
            # The difference comes from the fact that the time is split
            # using `split_time` which is a floating point value and there
            # can be a small offset to the left or right when getting the
            # array indices corresponding to that time.
            assert np.sum(im0 + ii - image_bg_internal[ii]) < 4

    # Now also make sure that the image_bg feature of the resulting
    # dataset is mapped correctly.
    with read.HDF5Data(path_out) as hd:
        assert hd["basinmap0"][0] == 0
        assert np.all(hd["image_bg"][0] == im0)
        assert np.all(hd["image_bg"][10] == im0)
        for ii in range(10):
            assert np.all(hd["image_bg"][40 * ii] == image_bg_internal[ii])
            assert np.all(hd["image_bg"][40 * ii + 5] == image_bg_internal[ii])


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
        assert bic.get_ppid() == "sparsemed:k=200^s=0.01^t=0^f=0.8^o=1"

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
        assert len(h5["basin_events/image_bg"]) == 2
