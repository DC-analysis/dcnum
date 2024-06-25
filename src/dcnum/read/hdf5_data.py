from __future__ import annotations

import hashlib
import io
import json
import numbers
import pathlib
import tempfile
from typing import Dict, BinaryIO, List
import uuid
import warnings

import h5py
import numpy as np

from .cache import HDF5ImageCache, ImageCorrCache, md5sum
from .const import PROTECTED_FEATURES
from .mapped import get_mapped_object, get_mapping_indices


class HDF5Data:
    """HDF5 (.rtdc) input file data instance"""
    def __init__(self,
                 path: pathlib.Path | h5py.File | BinaryIO,
                 pixel_size: float = None,
                 md5_5m: str = None,
                 meta: Dict = None,
                 basins: List[Dict[List[str] | str]] = None,
                 logs: Dict[List[str]] = None,
                 tables: Dict[np.ndarray] = None,
                 image_cache_size: int = 2,
                 index_mapping: int | slice | List | np.ndarray = None,
                 ):
        """

        Parameters
        ----------
        path:
            path to data file
        pixel_size:
            pixel size in µm
        md5_5m:
            MD5 sum of the first 5 MiB; computed if not provided
        meta:
            metadata dictionary; extracted from HDF5 attributes
            if not provided
        basins:
            list of basin dictionaries; extracted from HDF5 attributes
            if not provided
        logs:
            dictionary of logs; extracted from HDF5 attributes
            if not provided
        tables:
            dictionary of tables; extracted from HDF5 attributes
            if not provided
        image_cache_size:
            size of the image cache to use when accessing image data
        index_mapping:
            select only a subset of input events, transparently reducing the
            size of the dataset, possible data types are
            - int `N`: use the first `N` events
            - slice: use the events defined by a slice
            - list: list of integers specifying the event indices to use
            Numpy indexing rules apply. E.g. to only process the first
            100 events, set this to `100` or `slice(0, 100)`.
        """
        # Init is in __setstate__ so we can pickle this class
        # and use it for multiprocessing.
        if isinstance(path, h5py.File):
            self.h5 = path
            path = path.filename

        self.__setstate__({"path": path,
                           "pixel_size": pixel_size,
                           "md5_5m": md5_5m,
                           "meta": meta,
                           "basins": basins,
                           "logs": logs,
                           "tables": tables,
                           "image_cache_size": image_cache_size,
                           "index_mapping": index_mapping,
                           })

    def __contains__(self, item):
        return item in self.keys()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, feat):
        if feat in ["image", "image_bg", "mask"]:
            data = self.get_image_cache(feat)  # already index-mapped
            if data is None:
                raise KeyError(f"Feature '{feat}' not found in {self}!")
            else:
                return data
        elif feat in self._cache_scalar:  # check for scalar cached
            return self._cache_scalar[feat]
        elif (feat in self.h5["events"]
              and len(self.h5["events"][feat].shape) == 1):  # cache scalar
            if self.index_mapping is None:
                # no mapping indices, just slice
                dat_sc = self.h5["events"][feat][:]
            else:
                dat_sc = get_mapped_object(self.h5["events"][feat],
                                           index_mapping=self.index_mapping)[:]
            self._cache_scalar[feat] = dat_sc
            return self._cache_scalar[feat]
        else:
            if feat in self.h5["events"]:
                # Not cached (possibly slow)
                warnings.warn(f"Feature {feat} not cached (possibly slow)")
                return get_mapped_object(
                    obj=self.h5["events"][feat],
                    index_mapping=self.index_mapping)
            else:
                # Check the basins
                for idx in range(len(self.basins)):
                    bn_grp, bn_feats, bn_map = self.get_basin_data(idx)
                    if bn_feats and feat in bn_feats:
                        mapped_ds = get_mapped_object(obj=bn_grp[feat],
                                                      index_mapping=bn_map)
                        return mapped_ds
        # If we got here, then the feature data does not exist.
        raise KeyError(f"Feature '{feat}' not found in {self}!")

    def __getstate__(self):
        return {"path": self.path,
                "pixel_size": self.pixel_size,
                "md5_5m": self.md5_5m,
                "meta": self.meta,
                "logs": self.logs,
                "tables": self.tables,
                "basins": self.basins,
                "image_cache_size": self.image.cache_size,
                "index_mapping": self.index_mapping,
                }

    def __setstate__(self, state):
        # Make sure these properties exist (we rely on __init__, because
        # we want this class to be pickable and __init__ is not called by
        # `pickle.load`).
        # Cached properties
        self._feats = None
        self._keys = None
        self._len = None
        self.image_cache_size = state["image_cache_size"]
        # Image cache
        if not hasattr(self, "_image_cache"):
            self._image_cache = {}
        # Basin data
        if not hasattr(self, "_basin_data"):
            self._basin_data = {}
        # Scalar feature cache
        if not hasattr(self, "_cache_scalar"):
            self._cache_scalar = {}
        if not hasattr(self, "h5"):
            self.h5 = None

        self.path = state["path"]

        self.md5_5m = state["md5_5m"]
        if self.md5_5m is None:
            if isinstance(self.path, pathlib.Path):
                # 5MB md5sum of input file
                self.md5_5m = md5sum(self.path, blocksize=65536, count=80)
            else:
                self.md5_5m = str(uuid.uuid4()).replace("-", "")
        self.meta = state["meta"]
        self.logs = state["logs"]
        self.tables = state["tables"]
        self.basins = state["basins"]
        if (self.meta is None
                or self.logs is None
                or self.tables is None
                or self.basins is None):
            self.logs = {}
            self.tables = {}
            self.basins = []
            # get dataset configuration
            with h5py.File(self.path,
                           libver="latest",
                           ) as h5:
                # meta
                self.meta = dict(h5.attrs)
                for key in self.meta:
                    if isinstance(self.meta[key], bytes):
                        self.meta[key] = self.meta[key].decode("utf-8")
                # logs
                for key in sorted(h5.get("logs", {}).keys()):
                    alog = list(h5["logs"][key])
                    if alog:
                        if isinstance(alog[0], bytes):
                            alog = [ll.decode("utf") for ll in alog]
                        self.logs[key] = alog
                # tables
                for tab in sorted(h5.get("tables", {}).keys()):
                    tabdict = {}
                    for tkey in h5["tables"][tab].dtype.fields.keys():
                        tabdict[tkey] = \
                            np.array(h5["tables"][tab][tkey]).reshape(-1)
                    self.tables[tab] = tabdict
                # basins
                basins = self.extract_basin_dicts(h5)
                self.basins = sorted(basins, key=lambda x: x["name"])

        if state["pixel_size"] is not None:
            self.pixel_size = state["pixel_size"]

        self.image_cache_size = state["image_cache_size"]

        self.index_mapping = state["index_mapping"]

        if self.h5 is None:
            self.h5 = h5py.File(self.path, libver="latest")

    def __len__(self):
        if self._len is None:
            if self.index_mapping is not None:
                self._len = get_mapping_indices(self.index_mapping).size
            else:
                self._len = self.h5.attrs["experiment:event count"]
        return self._len

    @property
    def image(self):
        return self.get_image_cache("image")

    @property
    def image_bg(self):
        return self.get_image_cache("image_bg")

    @property
    def image_corr(self):
        if "image_corr" not in self._image_cache:
            if self.image is not None and self.image_bg is not None:
                image_corr = ImageCorrCache(self.image, self.image_bg)
            else:
                image_corr = None
            self._image_cache["image_corr"] = image_corr
        return self._image_cache["image_corr"]

    @property
    def mask(self):
        return self.get_image_cache("mask")

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
    def pixel_size(self, pixel_size: float):
        # Reduce pixel_size accuracy to 8 digits after the point to
        # enforce pipeline reproducibility (see get_ppid_from_ppkw).
        pixel_size = float(f"{pixel_size:.8f}")
        self.meta["imaging:pixel size"] = pixel_size

    @staticmethod
    def extract_basin_dicts(h5, check=True):
        """Return list of basin dictionaries"""
        # TODO:
        #  - support iterative mapped basins and catch
        #    circular basin definitions.
        basins = []
        for bnkey in h5.get("basins", {}).keys():
            bn_data = "\n".join(
                [s.decode() for s in h5["basins"][bnkey][:].tolist()])
            bn_dict = json.loads(bn_data)
            if check:
                if bn_dict["type"] not in ["internal", "file"]:
                    # we only support file-based and internal basins
                    continue
                basinmap = bn_dict.get("mapping")
                if basinmap is not None and basinmap not in h5["events"]:
                    # basinmap feature is missing
                    continue
            # Add the basin
            basins.append(bn_dict)

        return basins

    @property
    def features_scalar_frame(self):
        """Scalar features that apply to all events in a frame

        This is a convenience function for copying scalar features
        over to new processed datasets. Return a list of all features
        that describe a frame (e.g. temperature or time).
        """
        if self._feats is None:
            feats = []
            for feat in self.keys():
                if feat in PROTECTED_FEATURES:
                    feats.append(feat)
            self._feats = feats
        return self._feats

    def close(self):
        """Close the underlying HDF5 file"""
        for bn_group, _, _ in self._basin_data.values():
            if bn_group is not None:
                if bn_group.id.valid:
                    bn_group.file.close()
        self._image_cache.clear()
        self._basin_data.clear()
        self.h5.close()

    def get_ppid(self):
        return self.get_ppid_from_ppkw(
            {"pixel_size": self.pixel_size,
             "index_mapping": self.index_mapping})

    @classmethod
    def get_ppid_code(cls):
        return "hdf"

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs):
        # Data does not really fit into the PPID scheme we use for the rest
        # of the pipeline. This implementation here is custom.
        code = cls.get_ppid_code()
        # pixel size
        ppid_ps = f"{kwargs['pixel_size']:.8f}".rstrip("0")
        # index mapping
        ppid_im = cls.get_ppid_index_mapping(kwargs.get("index_mapping", None))
        kwid = "^".join([f"p={ppid_ps}", f"i={ppid_im}"])
        return ":".join([code, kwid])

    @staticmethod
    def get_ppid_index_mapping(index_mapping):
        """Return the pipeline identifier part for index mapping"""
        im = index_mapping
        if im is None:
            dim = "0"
        elif isinstance(im, numbers.Integral):
            dim = f"{im}"
        elif isinstance(im, slice):
            dim = (f"{im.start if im.start is not None else 'n'}"
                   + f"-{im.stop if im.stop is not None else 'n'}"
                   + f"-{im.step if im.step is not None else 'n'}"
                   )
        elif isinstance(im, (list, np.ndarray)):
            idhash = hashlib.md5(
                np.array(im, dtype=np.uint32).tobytes()).hexdigest()
            dim = f"h-{idhash[:8]}"
        else:
            dim = "unknown"
        return dim

    @staticmethod
    def get_ppkw_from_ppid(dat_ppid):
        # Data does not fit in the PPID scheme we use, but we still
        # would like to pass pixel_size to __init__ if we need it.
        code, kwargs_str = dat_ppid.split(":")
        if code != HDF5Data.get_ppid_code():
            raise ValueError(f"Could not find data method '{code}'!")
        kwitems = kwargs_str.split("^")
        kwargs = {}
        for item in kwitems:
            var, val = item.split("=")
            if var == "p":
                kwargs["pixel_size"] = float(val)
            elif var == "i":
                if val.startswith("h-") or val == "unknown":
                    raise ValueError(f"Cannot invert index mapping {val}")
                elif val == "0":
                    kwargs["index_mapping"] = None
                elif val.count("-"):
                    start, stop, step = val.split("-")
                    kwargs["index_mapping"] = slice(
                        None if start == "n" else int(start),
                        None if stop == "n" else int(stop),
                        None if step == "n" else int(step)
                    )
                else:
                    kwargs["index_mapping"] = int(val)
            else:
                raise ValueError(f"Invalid parameter '{var}'!")
        return kwargs

    def get_basin_data(self, index: int) -> (
            h5py.Group,
            List,
            int | slice | List | np.ndarray,
            ):
        """Return HDF5Data info for a basin index in `self.basins`

        Parameters
        ----------
        index: int
            index of the basin from which to get data

        Returns
        -------
        group: h5py.Group
            HDF5 group containing HDF5 Datasets with the names
            listed in `features`
        features: list of str
            list of features made available by this basin
        index_mapping:
            a mapping (see `__init__`) that defines mapping from
            the basin dataset to the referring dataset
        """
        if index not in self._basin_data:
            bn_dict = self.basins[index]

            # HDF5 group containing the feature data
            if bn_dict["type"] == "file":
                h5group, features = self._get_basin_data_file(bn_dict)
            elif bn_dict["type"] == "internal":
                h5group, features = self._get_basin_data_internal(bn_dict)
            else:
                raise ValueError(f"Invalid basin type '{bn_dict['type']}'")

            # index mapping
            feat_basinmap = bn_dict.get("mapping", None)
            if feat_basinmap is None:
                # This is NOT a mapped basin.
                index_mapping = self.index_mapping
            else:
                # This is a mapped basin. Create an indexing list.
                if self.index_mapping is None:
                    # The current dataset is not mapped.
                    basinmap_idx = slice(None)
                else:
                    # The current dataset is also mapped.
                    basinmap_idx = get_mapping_indices(self.index_mapping)
                basinmap = self.h5[f"events/{feat_basinmap}"]
                index_mapping = basinmap[basinmap_idx]

            self._basin_data[index] = (h5group, features, index_mapping)
        return self._basin_data[index]

    def _get_basin_data_file(self, bn_dict):
        for ff in bn_dict["paths"]:
            pp = pathlib.Path(ff)
            if pp.is_absolute() and pp.exists():
                path = pp
                break
            else:
                # try relative path
                prel = pathlib.Path(self.path).parent / pp
                if prel.exists():
                    path = prel
                    break
        else:
            path = None
        if path is None:
            # Cannot get data from this basin / cannot find file
            h5group = None
            features = []
        else:
            h5 = h5py.File(path, "r")
            h5group = h5["events"]
            # features defined in the basin
            features = bn_dict.get("features")
            if features is None:
                # Only get the features from the actual HDF5 file.
                # If this file has basins as well, the basin metadata
                # should have been copied over to the parent file. This
                # makes things a little cleaner, because basins are not
                # nested, but all basins are available in the top file.
                # See :func:`write.store_metadata` for copying metadata
                # between files.
                # The writer can still specify "features" in the basin
                # metadata, then these basins are indeed nested, and
                # we consider that ok as well.
                features = sorted(h5group.keys())
        return h5group, features

    def _get_basin_data_internal(self, bn_dict):
        # The group name is normally "basin_events"
        group_name = bn_dict["paths"][0]
        if group_name != "basin_events":
            warnings.warn(
                f"Uncommon group name for basin features: {group_name}")
        h5group = self.h5[group_name]
        features = bn_dict.get("features")
        if features is None:
            raise ValueError(
                f"Encountered invalid internal basin '{bn_dict}': "
                f"'features' must be defined")
        return h5group, features

    def get_image_cache(self, feat):
        """Create an HDF5ImageCache object for the current dataset

        This method also tries to find image data in `self.basins`.
        """
        if feat not in self._image_cache:
            if f"events/{feat}" in self.h5:
                ds = self.h5[f"events/{feat}"]
                idx_map = self.index_mapping
            else:
                idx_map = None
                # search all basins
                for idx in range(len(self.basins)):
                    bn_grp, bn_feats, bn_map = self.get_basin_data(idx)
                    if bn_feats is not None:
                        if feat in bn_feats:
                            # HDF5 dataset
                            ds = bn_grp[feat]
                            # Index mapping (taken from the basins which
                            # already includes the mapping from the current
                            # instance).
                            idx_map = bn_map
                            break
                else:
                    ds = None

            if ds is not None:
                image = HDF5ImageCache(
                    h5ds=get_mapped_object(obj=ds, index_mapping=idx_map),
                    cache_size=self.image_cache_size,
                    boolean=feat == "mask")
            else:
                image = None
            self._image_cache[feat] = image

        return self._image_cache[feat]

    def keys(self):
        if self._keys is None:
            features = sorted(self.h5["/events"].keys())
            # add basin features
            for ii in range(len(self.basins)):
                _, bn_feats, _ = self.get_basin_data(ii)
                if bn_feats:
                    features += bn_feats
            self._keys = sorted(set(features))
        return self._keys


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
        the dataset is written to that file. Note that datasets in memory
        are likely not pickable (so don't use them for multiprocessing).
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
    - Basins are not considered.
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
                    if not features:
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
                        continue
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
