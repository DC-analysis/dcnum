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


def get_mapping_indices(
        index_mapping: numbers.Integral | slice | list | np.ndarray
        ):
    """Return integer numpy array with mapping indices for a range

    Parameters
    ----------
    index_mapping: numbers.Integral | slice | list | np.ndarray
        Several options you have here:
        - integer: results in np.arange(integer)
        - slice: results in np.arange(slice.start, slice.stop, slice.step)
        - list or np.ndarray: returns the input as uint32 array
    """
    if isinstance(index_mapping, numbers.Integral):
        return _get_mapping_indices_cached(index_mapping)
    elif isinstance(index_mapping, slice):
        return _get_mapping_indices_cached(
            (index_mapping.start, index_mapping.stop, index_mapping.step))
    elif isinstance(index_mapping, (np.ndarray, list)):
        return np.asarray(index_mapping, dtype=np.uint32)
    else:
        raise ValueError(f"Invalid type for `index_mapping`: "
                         f"{type(index_mapping)} ({index_mapping})")


@functools.lru_cache(maxsize=100)
def _get_mapping_indices_cached(
        index_mapping: numbers.Integral | tuple
        ):
    if isinstance(index_mapping, numbers.Integral):
        return np.arange(index_mapping)
    elif isinstance(index_mapping, tuple):
        im_slice = slice(*index_mapping)
        if im_slice.stop is None or im_slice.start is None:
            raise NotImplementedError(
                "Slices must have start and stop defined")
        return np.arange(im_slice.start, im_slice.stop, im_slice.step)
    elif isinstance(index_mapping, list):
        return np.asarray(index_mapping, dtype=np.uint32)
    else:
        raise ValueError(f"Invalid type for cached `index_mapping`: "
                         f"{type(index_mapping)} ({index_mapping})")


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
