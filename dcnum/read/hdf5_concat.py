"""Concatenation of multiple HDF5 files"""

import io
import pathlib
import tempfile
from typing import Any
import warnings

import numpy as np

from ..common import h5py

from .hdf5_data import HDF5Data


def concatenated_hdf5_data(paths: list[pathlib.Path],
                           path_out: bool | pathlib.Path | None = True,
                           compute_frame: bool = True,
                           features: list[str] | None = None):
    """Return a virtual dataset concatenating all the input paths

    Parameters
    ----------
    paths:
        Paths of the input HDF5 files that will be concatenated along
        the event axis. The metadata will be taken from the first
        file.
    path_out:
        If `None`, then the dataset is created in memory. If `True`
        (default), create a temporary file on disk (which is not deleted
        automatically). If a pathlib.Path or str is specified,
        the dataset is written to that file. Note that datasets in memory
        are likely not ``pickle``-able (so don't use them for multiprocessing).
    compute_frame:
        Whether to compute the "events/frame" feature, taking the frame
        data from the input files and properly incrementing them along
        the file index.
    features:
        List of features to take from the input files (defaults to
        all features of the first file).

    Returns
    -------
    data: dcnum.read.HDF5Data
        Instance of the concatenated dataset; The virtual datasets
        refer to the input files via absolute paths, so the input files
        must not be moved.

    Notes
    -----
    - If one of the input files does not contain a feature from the first
      input `paths`, then a `ValueError` is raised. Use the `features`
      argument to specify which features you need instead.
    - Basins are not considered.
    - The "time" feature is not included, because the time stamps of
      the individual input files cannot be concatenated (also, "frame"
      is only included if `compute_frame` is set). This is intended;
      algorithms that need a time axis (e.g. sparse median background
      computation) fall back to "frame" and "imaging:frame rate".
    """
    h5kwargs: dict[str, Any] = {"mode": "w", "libver": "latest"}
    if isinstance(path_out, (pathlib.Path, str)):
        h5kwargs["name"] = path_out
    elif path_out is True:
        with tempfile.NamedTemporaryFile(prefix="dcnum_vc_",
                                         suffix=".hdf5",
                                         delete=False) as tf:
            tf.write(b"dummy")
            h5kwargs["name"] = tf.name
    elif path_out is None:
        h5kwargs["name"] = io.BytesIO()
    else:
        raise ValueError(
            f"Invalid type for `path_out`: {type(path_out)} ({path_out}")

    if len(paths) == 0:
        raise ValueError("Please specify at least one file in `paths`!")
    elif len(paths) == 1:
        warnings.warn("Only one file passed to `concatenated_hdf5_data`; this "
                      "is equivalent to using `HDF5Data`, but slower.")

    # Use absolute paths, because relative paths in the virtual datasets
    # would be resolved relative to the current working directory when
    # the data are read.
    paths = [pathlib.Path(pp).resolve() for pp in paths]

    frames = []

    with h5py.File(**h5kwargs) as hv:
        # determine the sizes of the input files
        shapes = {}
        dtypes = {}
        size = 0
        meta = {}
        features_used: list[str] = list(features) if features else []
        for ii, pp in enumerate(paths):
            with h5py.File(pp, libver="latest") as h5:
                # get all feature keys
                featsi = sorted(h5["events"].keys())
                # get metadata
                if ii == 0:
                    meta = dict(h5.attrs)
                    if not features_used:
                        features_used = featsi
                # make sure number of features are consistent
                if not set(features_used) <= set(featsi):
                    raise ValueError(
                        f"File {pp} contains more features than {paths[0]}!")
                # populate shapes for all features
                for feat in features_used:
                    if not isinstance(h5["events"][feat], h5py.Dataset):
                        warnings.warn(
                            f"Ignoring {feat}; not implemented yet!")
                        continue
                    if feat in ["frame", "time"]:
                        # "frame" is computed below, "time" is not
                        # included (see docstring)
                        continue
                    shapes.setdefault(feat, []).append(
                        h5["events"][feat].shape)
                    if ii == 0:
                        dtypes[feat] = h5["events"][feat].dtype
                # increment size
                size += h5["events"][features_used[0]].shape[0]
                # remember the frame feature if requested
                if compute_frame:
                    frames.append(h5["events/frame"][:])

        # write metadata
        hv.attrs.update(meta)

        # Create the virtual datasets
        for feat, feat_shapes in shapes.items():
            if len(feat_shapes[0]) == 1:
                # scalar feature
                shape = (sum([sh[0] for sh in feat_shapes]))
            else:
                # non-scalar feature
                length = (sum([sh[0] for sh in feat_shapes]))
                shape = list(feat_shapes[0])
                shape[0] = length
                shape = tuple(shape)
            layout = h5py.VirtualLayout(shape=shape, dtype=dtypes[feat])
            loc = 0
            for jj, pp in enumerate(paths):
                vsource = h5py.VirtualSource(pp, f"events/{feat}",
                                             shape=feat_shapes[jj])
                cursize = feat_shapes[jj][0]
                layout[loc:loc+cursize] = vsource
                loc += cursize
            hv.create_virtual_dataset(f"/events/{feat}", layout, fillvalue=0)

        if compute_frame:
            # concatenate frames and store in dataset
            frame_concat = np.zeros(size, dtype=np.uint64)
            locf = 0  # indexing location
            prevmax = 0  # maximum frame number stored so far in array
            for fr in frames:
                offset = prevmax + 1 - fr[0]
                frame_concat[locf:locf+fr.size] = fr + offset
                locf += fr.size
                prevmax = fr[-1] + offset
            hv.create_dataset("/events/frame", data=frame_concat)

        # write metadata
        hv.attrs["experiment:event count"] = size

    data = HDF5Data(h5kwargs["name"])
    return data
