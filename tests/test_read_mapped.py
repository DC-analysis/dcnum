import h5py
import numpy as np
import pytest

from dcnum import read
from dcnum.read.mapped import (
    MappedHDF5Dataset, get_mapped_object, get_mapping_indices
)


@pytest.mark.parametrize("idx", [
    2,
    slice(1, 4),
    np.array([3, 0, 2]),
    (2, slice(1, 3)),
    (slice(1, 4), slice(0, 2), 1),
    (np.array([3, 0, 2]), slice(None), slice(1, 3)),
])
def test_mapped_dataset_getitem(tmp_path, idx):
    """Tuple indexing applies the first index to events (dcnum 0.31.2)"""
    data = np.arange(6 * 3 * 4).reshape(6, 3, 4)
    mapping = np.array([5, 1, 2, 4])
    with h5py.File(tmp_path / "test.h5", "w") as h5:
        h5["data"] = data
        mds = get_mapped_object(h5["data"], index_mapping=mapping)
        assert isinstance(mds, MappedHDF5Dataset)
        assert mds.shape == (4, 3, 4)
        assert np.array_equal(mds[idx], data[mapping][idx])


@pytest.mark.parametrize("index_mapping", [10, slice(2, 8), slice(1, 9, 2)])
def test_mapping_indices_cache_not_modifiable(index_mapping):
    """Cached mapping indices are shared and must not be modified

    In dcnum 0.31.2, modifying the returned array changed the result
    of all later calls with the same mapping.
    """
    indices = get_mapping_indices(index_mapping)
    reference = indices.copy()
    with pytest.raises(ValueError, match="read-only"):
        indices[0] = 99
    assert np.all(get_mapping_indices(index_mapping) == reference)


def test_mapped_dataset_attributes(tmp_path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as h5:
        h5["image"] = np.zeros((5, 4, 4), dtype=np.uint8)
    with h5py.File(path) as h5:
        mds = MappedHDF5Dataset(h5["image"], mapping_indices=np.arange(3))
        assert mds.name == "/image"
        assert mds.file.filename == str(path)
        assert mds.dtype == np.uint8


def test_mapped_dataset_zero_length_image_cache(tmp_path):
    """HDF5ImageCache warns about zero-length data (no AttributeError)"""
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as h5:
        h5["image"] = np.zeros((5, 4, 4), dtype=np.uint8)
    with h5py.File(path) as h5:
        mds = MappedHDF5Dataset(h5["image"],
                                mapping_indices=np.array([], dtype=int))
        with pytest.warns(read.cache.EmptyDatasetWarning,
                          match="Input image '/image' in file"):
            read.cache.HDF5ImageCache(mds)
