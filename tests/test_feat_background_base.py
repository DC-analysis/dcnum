import json

import h5py
import numpy as np

import pytest

from dcnum.feat.feat_background import bg_sparse_median
from dcnum.read import HDF5Data
from dcnum.write import create_with_basins


def test_base_background_input_has_basin(tmp_path):
    """In dcnum 0.13.0, we introduced `create_with_basins`

    This test checks whether the background class accepts a dataset
    with the image feature in a basin.
    """
    event_count = 720
    output_path = tmp_path / "test.h5"
    basin_path = tmp_path / "basin.h5"
    input_path = tmp_path / "input.h5"
    # image shape: 5 * 7
    with h5py.File(basin_path, "a") as h5:
        h5["events/image"] = \
            np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    # create a file with the basin_path as a basin
    create_with_basins(path_out=input_path, basin_paths=[basin_path])

    with bg_sparse_median.BackgroundSparseMed(input_data=input_path,
                                              output_path=output_path,
                                              kernel_size=10,
                                              split_time=0.011,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert bic.get_progress() == 0
        bic.process()
        assert bic.get_progress() == 1

    # Make sure the basins exist in the input file
    with h5py.File(output_path) as h5:
        assert len(h5["basins"].keys()) == 2
        assert "basins" in h5
        keys = list(h5["basins"].keys())
        bpaths = []
        for key in keys:
            bn_lines = [k.decode("utf-8") for k in h5["basins"][key]]
            bdat = json.loads(" ".join(bn_lines))
            bpaths.append(bdat["paths"][0])
        assert str(input_path) in bpaths
        assert str(basin_path) in bpaths

    # Add a cherry on top (make sure everything is parseable with HDF5Data)
    with HDF5Data(output_path) as hd:
        assert "image" in hd
        assert "image_bg" in hd


def test_base_background_input_is_output(tmp_path):
    event_count = 720
    input_path = tmp_path / "input.h5"
    output_path = input_path
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

    # Add a cherry on top (make sure everything is parseable with HDF5Data)
    with HDF5Data(output_path) as hd:
        assert "image" in hd
        assert "image_bg" in hd


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_base_background_output_basin_none(
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


def test_base_background_output_basin_simple(
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
        assert len(h5["basins"].keys()) == 1
        key = list(h5["basins"].keys())[0]
        bn_lines = [k.decode("utf-8") for k in h5["basins"][key]]
        bdat = json.loads(" ".join(bn_lines))
        assert bdat["paths"][0] == str(input_path)

    # Add a cherry on top (make sure everything is parseable with HDF5Data)
    with HDF5Data(output_path) as hd:
        assert "image" in hd
        assert "image_bg" in hd
