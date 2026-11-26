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
        assert len(h5["basins"].keys()) == 3
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
    """
    In dcnum 0.13.0, we introduced `create_with_basins`, and the purpose
    of this test was to check whether the data are written to the output
    file.
    In dcnum 0.24.0, we introduced internal basins, and the purpose
    of this test was to make sure that the internal basins exist.

    """
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
    # Make sure the basin data exists in the input file
    with h5py.File(output_path) as h5:
        assert len(h5["basins"]) == 1, "one internal basin"
        assert "image_bg" in h5["basin_events"]


def test_base_background_output_basin_simple(
        tmp_path):
    """
    In dcnum 0.13.0, we introduced `create_with_basins`, and the purpose
    of this test was to make sure that the image data in the input is
    referred to in the output file via a basin.
    In dcnum 0.24.0, we introduced internal basins, which means that
    the image_bg data are written to an additional, internal basin.
    """
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
        assert len(h5["basins"].keys()) == 2
        bn_keys = list(h5["basins"].keys())
        for key in bn_keys:
            bn_lines = [k.decode("utf-8") for k in h5["basins"][key]]
            bdat = json.loads(" ".join(bn_lines))
            if bdat["type"] == "file":
                assert bdat["paths"][0] == str(input_path)
            elif bdat["type"] == "internal":
                assert bdat["paths"][0] == "basin_events"

    # Add a cherry on top (make sure everything is parseable with HDF5Data)
    with HDF5Data(output_path) as hd:
        assert "image" in hd
        assert "image_bg" in hd


def test_base_background_input_without_images(tmp_path):
    """Input data without images raise a clear error"""
    path_in = tmp_path / "in.rtdc"
    with h5py.File(path_in, "w") as h5:
        h5.attrs["experiment:event count"] = 10
        h5["events/deform"] = np.linspace(0, .1, 10)
    with pytest.raises(ValueError, match="do not contain the 'image'"):
        bg_sparse_median.BackgroundSparseMed(
            input_data=path_in,
            output_path=tmp_path / "out.rtdc",
            kernel_size=5)


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_base_background_input_path_str(tmp_path):
    """The input path may be given as a string"""
    input_path = tmp_path / "input.h5"
    with h5py.File(input_path, "a") as h5:
        h5["events/image"] = \
            np.arange(5*7).reshape(1, 5, 7) * np.ones((50, 1, 1))
    with bg_sparse_median.BackgroundSparseMed(input_data=str(input_path),
                                              output_path=tmp_path / "o.h5",
                                              kernel_size=10,
                                              split_time=0.011,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        assert bic.image_count == 50


def test_base_background_get_ppid_code_base_class():
    from dcnum.feat.feat_background.base import Background
    with pytest.raises(ValueError, match="for `Background` base class"):
        Background.get_ppid_code()


def test_base_background_get_ppkw_from_ppid_invalid_code():
    from dcnum.feat.feat_background.base import Background
    with pytest.raises(ValueError,
                       match="Could not find background computation method"):
        Background.get_ppkw_from_ppid("peter:k=10")


@pytest.mark.filterwarnings(
    "ignore::dcnum.write.writer.CreatingFileWithoutBasinWarning")
def test_base_background_process_twice(tmp_path):
    """Processing again replaces the existing background data"""
    event_count = 720
    input_data = np.arange(5*7).reshape(1, 5, 7) * np.ones((event_count, 1, 1))
    output_path = tmp_path / "test.h5"
    with bg_sparse_median.BackgroundSparseMed(input_data=input_data,
                                              output_path=output_path,
                                              kernel_size=10,
                                              split_time=0.011,
                                              thresh_cleansing=0,
                                              frac_cleansing=.8,
                                              ) as bic:
        bic.process()
        bg_1 = bic.h5out["basin_events/image_bg"][:]
        bic.process()
        bg_2 = bic.h5out["basin_events/image_bg"][:]
        assert np.all(bg_1 == bg_2)
        assert bic.get_progress() == 1
        # no images -> no progress
        bic.image_count = 0
        assert bic.get_progress() == 0
