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
        bic.process()
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
