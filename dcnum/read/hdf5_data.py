from __future__ import annotations

import io
import pathlib
import tempfile
from typing import Dict, BinaryIO, List
import uuid
import warnings

import h5py
import numpy as np

from .cache import HDF5ImageCache, ImageCorrCache, md5sum
from .const import PROTECTED_FEATURES


class HDF5Data:
    """HDF5 (.rtdc) input file data instance"""
    def __init__(self,
                 path: pathlib.Path | h5py.File | BinaryIO,
                 pixel_size: float = None,
                 md5_5m: str = None,
                 meta: Dict = None,
                 logs: Dict[List[str]] = None,
                 tables: Dict[np.ndarray] = None,
                 image_cache_size: int = 5,
                 ):
        # Cached properties
        self._feats = None
        self._keys = None
        self._len = None
        # Init is in __setstate__ so we can pickle this class
        # and use it for multiprocessing.
        if isinstance(path, h5py.File):
            self.h5 = path
            path = path.filename
        self.__setstate__({"path": path,
                           "pixel_size": pixel_size,
                           "md5_5m": md5_5m,
                           "meta": meta,
                           "logs": logs,
                           "tables": tables,
                           "image_cache_size": image_cache_size,
                           })

    def __contains__(self, item):
        return item in self.keys()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, feat):
        if feat == "image":
            return self.image
        elif feat == "image_bg":
            return self.image_bg
        elif feat == "mask" and self.mask is not None:
            return self.mask
        elif feat in self._cache_scalar:
            return self._cache_scalar[feat]
        elif len(self.h5["events"][feat].shape) == 1:
            self._cache_scalar[feat] = self.h5["events"][feat][:]
            return self._cache_scalar[feat]
        else:
            # Not cached (possibly slow)
            warnings.warn(f"Feature {feat} not cached (possibly slow)")
            return self.h5["events"][feat]

    def __getstate__(self):
        return {"path": self.path,
                "pixel_size": self.pixel_size,
                "md5_5m": self.md5_5m,
                "meta": self.meta,
                "logs": self.logs,
                "tables": self.tables,
                "image_cache_size": self.image.cache_size
                }

    def __setstate__(self, state):
        # Make sure these properties exist (we rely on __init__, because
        # we want this class to be pickable and __init__ is not called by
        # `pickle.load`.
        if not hasattr(self, "_cache_scalar"):
            self._cache_scalar = {}
        if not hasattr(self, "h5"):
            self.h5 = None

        self.path = state["path"]

        self.md5_5m = state["md5_5m"]
        if self.md5_5m is None:
            if isinstance(self.path, pathlib.Path):
                # 5MB md5sum of input file
                self.md5_5m = md5sum(self.path, count=80)
            else:
                self.md5_5m = str(uuid.uuid4()).replace("-", "")
        self.logs = state["logs"]
        self.tables = state["tables"]
        self.meta = state["meta"]
        if self.meta is None or self.logs is None or self.tables is None:
            self.logs = {}
            self.tables = {}
            # get dataset configuration
            with h5py.File(self.path,
                           libver="latest",
                           locking=False,
                           ) as h5:
                self.meta = dict(h5.attrs)
                for key in self.meta:
                    if isinstance(self.meta[key], bytes):
                        self.meta[key] = self.meta[key].decode("utf-8")
                for key in h5.get("logs", []):
                    alog = list(h5["logs"][key])
                    if alog:
                        if isinstance(alog[0], bytes):
                            alog = [ll.decode("utf") for ll in alog]
                        self.logs[key] = alog
                for tab in h5.get("tables", []):
                    tabdict = {}
                    for tkey in h5["tables"][tab].dtype.fields.keys():
                        tabdict[tkey] = \
                            np.array(h5["tables"][tab][tkey]).reshape(-1)
                    self.tables[tab] = tabdict

        if state["pixel_size"] is not None:
            self.pixel_size = state["pixel_size"]
        else:
            # Set known pixel size if possible
            did = self.meta.get("setup:identifier", "EMPTY")
            if (did.startswith("RC-")
                    and (self.pixel_size < 0.255 or self.pixel_size > 0.275)):
                warnings.warn(
                    f"Correcting for invalid pixel size in '{self.path}'!")
                # Set default pixel size for Rivercyte devices
                self.pixel_size = 0.2645

        if self.h5 is None:
            self.h5 = h5py.File(self.path, libver="latest")
        self.image = HDF5ImageCache(
            self.h5["events/image"],
            cache_size=state["image_cache_size"])

        if "events/image_bg" in self.h5:
            self.image_bg = HDF5ImageCache(
                self.h5["events/image_bg"],
                cache_size=state["image_cache_size"])
        else:
            self.image_bg = None

        if "events/mask" in self.h5:
            self.mask = HDF5ImageCache(
                self.h5["events/mask"],
                cache_size=state["image_cache_size"],
                boolean=True)
        else:
            self.mask = None

        self.image_corr = ImageCorrCache(self.image, self.image_bg)

    def __len__(self):
        if self._len is None:
            self._len = self.h5.attrs["experiment:event count"]
        return self._len

    @property
    def meta_nest(self):
        """Return `self.meta` as nested dicitonary

        This gets very close to the dclab `config` property of datasets.
        """
        md = {}
        for key in self.meta:
            sec, var = key.split(":")
            md.setdefault(sec, {})[var] = self.meta[key]
        return md

    @property
    def pixel_size(self):
        return self.meta.get("imaging:pixel size", 0)

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        self.meta["imaging:pixel size"] = pixel_size

    def close(self):
        """Close the underlying HDF5 file"""
        self.h5.close()

    def keys(self):
        if self._keys is None:
            self._keys = sorted(self.h5["/events"].keys())
        return self._keys

    @property
    def features_scalar_frame(self):
        """Scalar features that apply to all events in a frame

        This is a convenience function for copying scalar features
        over to new processed datasets. Return a list of all features
        that describe a frame (e.g. temperature or time).
        """
        if self._feats is None:
            feats = []
            for feat in self.h5["events"]:
                if feat in PROTECTED_FEATURES:
                    feats.append(feat)
            self._feats = feats
        return self._feats


def concatenated_hdf5_data(paths: List[pathlib.Path],
                           path_out: True | pathlib.Path | None = True,
                           compute_frame: bool = True,
                           features: List[str] | None = None):
    """Return a virtual dataset concatenating all the input paths

    Parameters
    ----------
    paths:
        Path of the input HDF5 files that will be concatenated along
        the feature axis. The metadata will be taken from the first
        file.
    path_out:
        If `None`, then the dataset is created in memory. If `True`
        (default), create a file on disk. If a pathlib.Path is specified,
        the dataset is written to that file. Note that files in memory
        are liekely not pickable (so don't use for multiprocessing).
    compute_frame:
        Whether to compute the "events/frame" feature, taking the frame
        data from the input files and properly incrementing them along
        the file index.
    features:
        List of features to take from the input files.

    Notes
    -----
    - If one of the input files does not contain a feature from the first
      input `paths`, then a `ValueError` is raised. Use the `features`
      argument to specify which features you need instead.
    """
    h5kwargs = {"mode": "w", "libver": "latest"}
    if isinstance(path_out, (pathlib.Path, str)):
        h5kwargs["name"] = path_out
    elif path_out is True:
        tf = tempfile.NamedTemporaryFile(prefix="dcnum_vc_",
                                         suffix=".hdf5",
                                         delete=False)
        tf.write(b"dummy")
        h5kwargs["name"] = tf.name
        tf.close()
    elif path_out is None:
        h5kwargs["name"] = io.BytesIO()
    else:
        raise ValueError(
            f"Invalid type for `path_out`: {type(path_out)} ({path_out}")

    if len(paths) <= 1:
        raise ValueError("Please specify at least two files in `paths`!")

    frames = []

    with h5py.File(**h5kwargs) as hv:
        # determine the sizes of the input files
        shapes = {}
        dtypes = {}
        size = 0
        for ii, pp in enumerate(paths):
            pp = pathlib.Path(pp).resolve()
            with h5py.File(pp, libver="latest") as h5:
                # get all feature keys
                featsi = sorted(h5["events"].keys())
                # get metadata
                if ii == 0:
                    meta = dict(h5.attrs)
                    features = featsi
                # make sure number of features are consistent
                if not set(features) <= set(featsi):
                    raise ValueError(
                        f"File {pp} contains more features than {paths[0]}!")
                # populate shapes for all features
                for feat in features:
                    if not isinstance(h5["events"][feat], h5py.Dataset):
                        warnings.warn(
                            f"Ignoring {feat}; not implemented yet!")
                    if feat in ["frame", "time"]:
                        continue
                    shapes.setdefault(feat, []).append(
                        h5["events"][feat].shape)
                    if ii == 0:
                        dtypes[feat] = h5["events"][feat].dtype
                # increment size
                size += h5["events"][features[0]].shape[0]
                # remember the frame feature if requested
                if compute_frame:
                    frames.append(h5["events/frame"][:])

        # write metadata
        hv.attrs.update(meta)

        # Create the virtual datasets
        for feat in shapes:
            if len(shapes[feat][0]) == 1:
                # scalar feature
                shape = (sum([sh[0] for sh in shapes[feat]]))
            else:
                # non-scalar feature
                length = (sum([sh[0] for sh in shapes[feat]]))
                shape = list(shapes[feat][0])
                shape[0] = length
                shape = tuple(shape)
            layout = h5py.VirtualLayout(shape=shape, dtype=dtypes[feat])
            loc = 0
            for jj, pp in enumerate(paths):
                vsource = h5py.VirtualSource(pp, f"events/{feat}",
                                             shape=shapes[feat][jj])
                cursize = shapes[feat][jj][0]
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
