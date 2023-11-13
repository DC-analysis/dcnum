import json

import h5py
import numpy as np
import pytest

from dcnum.feat.feat_background import bg_sparse_median
from dcnum.read import HDF5Data

from helper_methods import retrieve_data


def test_basic_background_output_basin_none(
        tmp_path):
    """In dcnum 0.13.0, we introduced `create_with_basins`"""
    event_count = 720
    output_path = tmp_path / "test.h5"
    # image shape: 5 * 7
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    assert np.all(input_data[0] == input_data[1])
    assert np.all(input_data[0].flatten() == np.arange(5*7))

    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=10,
                                              split_time=0.011,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        bic.process()
    # Make sure the basins exist in the input file
    with h5py.File(output_path) as h5:
        assert "basins" not in h5, "because the input is not a file"


def test_basic_background_output_basin_simple(
        tmp_path):
    """In dcnum 0.13.0, we introduced `create_with_basins`"""
    event_count = 720
    output_path = tmp_path / "test.h5"
    input_path = tmp_path / "input.h5"
    # image shape: 5 * 7
    with h5py.File(input_path, "a") as h5:
        h5["events/image"] = \
            np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))

    with bg_sparse_median.BackgroundSparseMed(input_data=input_path,
                                              output_path=output_path,
                                              kernel_size=10,
                                              split_time=0.011,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        bic.process()

    # Make sure the basins exist in the input file
    with h5py.File(output_path) as h5:
        assert "basins" in h5
        key = list(h5["basins"].keys())[0]
        bn_lines = [k.decode("utf-8") for k in h5["basins"][key]]
        bdat = json.loads(" ".join(bn_lines))
        assert bdat["paths"][0] == str(input_path)

    # Add a cherry on top (make sure everything is parseable with HDF5Data)
    with HDF5Data(output_path) as hd:
        assert "image" in hd
        assert "image_bg" in hd


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
