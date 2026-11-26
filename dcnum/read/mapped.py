"""Index-mapped access to HDF5 datasets"""

import functools
import numbers

import numpy as np

from ..common import h5py


class MappedHDF5Dataset:
    def __init__(self,
                 h5ds: h5py.Dataset,  # type: ignore
                 mapping_indices: np.ndarray):
        """An index-mapped object for accessing an HDF5 dataset

        Parameters
        ----------
        h5ds: h5py.Dataset
            HDF5 dataset from which to map data
        mapping_indices: np.ndarray
            numpy indexing array containing integer indices
        """
        self.h5ds = h5ds
        self.mapping_indices = mapping_indices
        self.shape = (mapping_indices.size,) + h5ds.shape[1:]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # Only the first index refers to events, the remaining
            # indices are applied to the event data (e.g. image ROI).
            idx, idx_data = idx[0], idx[1:]
        else:
            idx_data = ()
        if isinstance(idx, numbers.Integral):
            return self.h5ds[(self.mapping_indices[idx],) + idx_data]
        else:
            midx = self.mapping_indices[idx]
            start = np.min(midx)
            # Add one, because the final index must be included
            stop = np.max(midx) + 1
            # We have to perform mapping.
            # Since h5py is very slow at indexing with arrays,
            # we instead read the data in chunks from the input file,
            # and perform the mapping afterward using the numpy arrays.
            data_in = self.h5ds[(slice(start, stop),) + idx_data]
            # Determine the indices that we need from that chunk.
            data = data_in[midx - start]
            return data

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        """Data type of the underlying HDF5 dataset"""
        return self.h5ds.dtype

    @property
    def file(self):
        """HDF5 file of the underlying HDF5 dataset"""
        return self.h5ds.file

    @property
    def name(self):
        """Name of the underlying HDF5 dataset"""
        return self.h5ds.name


def check_index_mapping(
        index_mapping: numbers.Integral | slice | list | np.ndarray | None
        ):
    """Raise a ValueError if `index_mapping` contains negative values"""
    if index_mapping is None:
        return
    elif isinstance(index_mapping, numbers.Integral):
        values = [index_mapping]
    elif isinstance(index_mapping, slice):
        values = [index_mapping.start, index_mapping.stop,
                  index_mapping.step]
    elif isinstance(index_mapping, (np.ndarray, list)):
        indices = np.asarray(index_mapping)
        values = [np.min(indices)] if indices.size else []
    else:
        # Type is checked elsewhere
        values = []
    for val in values:
        if val is not None and val < 0:
            raise ValueError(f"Negative values in `index_mapping` are not "
                             f"allowed: {index_mapping}")


def get_mapping_indices(
        index_mapping: numbers.Integral | slice | list | np.ndarray,
        size: int | None = None,
        ):
    """Return integer numpy array with mapping indices for a range

    Parameters
    ----------
    index_mapping: numbers.Integral | slice | list | np.ndarray
        Several options you have here:

        - integer: results in ``np.arange(integer)``
        - slice: results in
          ``np.arange(slice.start, slice.stop, slice.step)``
        - list or np.ndarray: returns the input as uint32 array

        Negative values are not allowed.
    size: int
        Size of the dataset to which the mapping is applied. If given,
        integers and slices are limited to the dataset size (e.g.
        ``slice(30, None)`` or ``slice(30, 1000)`` for a dataset of size
        40 results in ``np.arange(30, 40)``), and a ``ValueError`` is
        raised if a list or array contains indices beyond the dataset.
        Slices without start or stop require ``size``.

    Returns
    -------
    indices: np.ndarray
        Integer array with the mapping indices. For integers and slices,
        this array is cached and shared with other callers, so it is
        read-only (make a copy if you need to modify it).

    Raises
    ------
    ValueError
        If ``index_mapping`` contains negative values or indices beyond
        ``size``
    TypeError
        If ``index_mapping`` has an invalid type
    NotImplementedError
        If ``index_mapping`` is a slice without start or stop and
        ``size`` is not given
    """
    check_index_mapping(index_mapping)
    if isinstance(index_mapping, numbers.Integral):
        if size is not None:
            index_mapping = min(index_mapping, size)
        return _get_mapping_indices_cached(int(index_mapping))
    elif isinstance(index_mapping, slice):
        if size is not None:
            # Limit slice to the dataset size
            return _get_mapping_indices_cached(index_mapping.indices(size))
        return _get_mapping_indices_cached(
            (index_mapping.start, index_mapping.stop, index_mapping.step))
    elif isinstance(index_mapping, (np.ndarray, list)):
        indices = np.asarray(index_mapping)
        if indices.size and size is not None and np.max(indices) >= size:
            raise ValueError(f"`index_mapping` contains indices beyond "
                             f"the dataset size {size}")
        return np.asarray(indices, dtype=np.uint32)
    else:
        raise TypeError(f"Invalid type for `index_mapping`: "
                        f"{type(index_mapping)} ({index_mapping})")


@functools.lru_cache(maxsize=100)
def _get_mapping_indices_cached(
        index_mapping: numbers.Integral | tuple
        ):
    if isinstance(index_mapping, numbers.Integral):
        indices = np.arange(index_mapping)
    elif isinstance(index_mapping, tuple):
        im_slice = slice(*index_mapping)
        if im_slice.stop is None or im_slice.start is None:
            raise NotImplementedError(
                "Slices must have start and stop defined")
        indices = np.arange(im_slice.start, im_slice.stop, im_slice.step)
    elif isinstance(index_mapping, list):
        indices = np.asarray(index_mapping, dtype=np.uint32)
    else:
        raise TypeError(f"Invalid type for cached `index_mapping`: "
                        f"{type(index_mapping)} ({index_mapping})")
    # The returned array is shared by all callers (cache), so it must
    # not be modified.
    indices.setflags(write=False)
    return indices


def get_mapped_object(obj, index_mapping=None):
    """Return an index-mapped version of an HDF5 dataset

    Parameters
    ----------
    obj: h5py.Dataset | dcnum.read.mapped.MappedHDF5Dataset
        dataset to map
    index_mapping: int | slice | list | np.ndarray | None
        index mapping (see ``get_mapping_indices``); if None,
        ``obj`` is returned unchanged

    Returns
    -------
    mapped_obj: h5py.Dataset | dcnum.read.mapped.MappedHDF5Dataset
        ``obj`` if ``index_mapping`` is None, otherwise a mapped
        dataset

    Raises
    ------
    ValueError
        If ``index_mapping`` is set and ``obj`` is not a (mapped) HDF5
        dataset

    Notes
    -----
    The size of ``obj`` is not taken into account, so slices must have
    a start and a stop (use e.g.
    :attr:`dcnum.read.hdf5_data.HDF5Data.index_mapping_indices` instead).
    """
    if index_mapping is None:
        return obj
    elif isinstance(obj, h5py.Dataset):
        return MappedHDF5Dataset(
            obj,
            mapping_indices=get_mapping_indices(index_mapping))
    elif isinstance(obj, MappedHDF5Dataset):
        # combine both mappings
        return MappedHDF5Dataset(
            obj.h5ds,
            mapping_indices=obj.mapping_indices[
                get_mapping_indices(index_mapping)])
    else:
        raise ValueError(f"No recipe to convert object of type {type(obj)} "
                         f"({obj}) to an index-mapped object")
