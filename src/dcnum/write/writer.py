import hashlib
import json
import pathlib
from typing import List
import warnings

import h5py
import hdf5plugin
import numpy as np

from .._version import version


class CreatingFileWithoutBasinWarning(UserWarning):
    """Issued when creating a basin-based dataset without basins"""
    pass


class HDF5Writer:
    def __init__(self, path, mode="a", ds_kwds=None):
        """Write deformability cytometry HDF5 data"""
        self.h5 = h5py.File(path, mode=mode, libver="latest")
        self.events = self.h5.require_group("events")
        ds_kwds = set_default_filter_kwargs(ds_kwds)
        self.ds_kwds = ds_kwds

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.h5.close()

    @staticmethod
    def get_best_nd_chunks(item_shape, feat_dtype=np.float64):
        """Return best chunks for HDF5 datasets

        Chunking has performance implications. It’s recommended to keep the
        total size of dataset chunks between 10 KiB and 1 MiB. This number
        defines the maximum chunk size as well as half the maximum cache
        size for each dataset.
        """
        # set image feature chunk size to approximately 1MiB
        num_bytes = 1024 ** 2
        # Note that `np.prod(()) == 1`
        event_size = np.prod(item_shape) * np.dtype(feat_dtype).itemsize
        chunk_size = num_bytes / event_size
        # Set minimum chunk size to 10 so that we can have at least some
        # compression performance.
        chunk_size_int = max(10, int(np.floor(chunk_size)))
        return tuple([chunk_size_int] + list(item_shape))

    def require_feature(self, feat, item_shape, feat_dtype, ds_kwds=None):
        """Create a new feature in the "events" group"""
        if feat not in self.events:
            if ds_kwds is None:
                ds_kwds = {}
            for key in self.ds_kwds:
                ds_kwds.setdefault(key, self.ds_kwds[key])
            dset = self.events.create_dataset(
                feat,
                shape=tuple([0] + list(item_shape)),
                dtype=feat_dtype,
                maxshape=tuple([None] + list(item_shape)),
                chunks=self.get_best_nd_chunks(item_shape,
                                               feat_dtype=feat_dtype),
                **ds_kwds)
            if len(item_shape) == 2:
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS',
                                  np.string_('IMAGE_GRAYSCALE'))
            offset = 0
        else:
            dset = self.events[feat]
            offset = dset.shape[0]
        return dset, offset

    def store_basin(self,
                    name: str,
                    paths: List[str | pathlib.Path],
                    features: List[str] = None,
                    description: str | None = None,
                    ):
        """Write an HDF5-based file basin

        Parameters
        ----------
        name: str
            basin name; Names do not have to be unique.
        paths: list of str or pathlib.Path
            location(s) of the basin
        features: list of str
            list of features provided by `paths`
        description: str
            optional string describing the basin
        """
        bdat = {
            "description": description,
            "format": "hdf5",
            "name": name,
            "paths": [str(pp) for pp in paths],
            "type": "file",
        }
        if features is not None and len(features):
            bdat["features"] = features
        bstring = json.dumps(bdat, indent=2)
        # basin key is its hash
        key = hashlib.md5(bstring.encode("utf-8",
                                         errors="ignore")).hexdigest()
        # write json-encoded basin to "basins" group
        basins = self.h5.require_group("basins")
        if key not in basins:
            blines = bstring.split("\n")
            basins.create_dataset(
                name=key,
                data=blines,
                shape=(len(blines),),
                # maximum line length
                dtype=f"S{max([len(b) for b in blines])}",
                chunks=True,
                **self.ds_kwds)

    def store_feature_chunk(self, feat, data):
        """Store feature data

        The "chunk" implies that always chunks of data are stored,
        never single events.
        """
        if feat == "mask" and data.dtype == bool:
            data = 255 * np.array(data, dtype=np.uint8)
        ds, offset = self.require_feature(feat=feat,
                                          item_shape=data.shape[1:],
                                          feat_dtype=data.dtype)
        dsize = data.shape[0]
        ds.resize(offset + dsize, axis=0)
        ds[offset:offset + dsize] = data

    def store_log(self,
                  log: str,
                  data: List[str],
                  override: bool = False):
        """Store log data

        Store the log data under the key `log`. The `data`
        kwarg must be a list of strings. If the log entry
        already exists, `ValueError` is raised unless
        `override` is set to True.
        """
        logs = self.h5.require_group("logs")
        if log in logs:
            if override:
                del logs[log]
            else:
                raise ValueError(
                    f"Log '{log}' already exists in {self.h5.filename}!")
        logs.create_dataset(
            name=log,
            data=data,
            shape=(len(data),),
            # maximum line length
            dtype=f"S{max([len(ll) for ll in data])}",
            chunks=True,
            **self.ds_kwds)


def create_with_basins(
        path_out: str | pathlib.Path,
        basin_paths: List[str | pathlib.Path] | List[List[str | pathlib.Path]]
        ):
    """Create an .rtdc file with basins

    Parameters
    ----------
    path_out:
        The output .rtdc file where basins are written to
    basin_paths:
        The paths to the basins written to `path_out`. This can be
        either a list of paths (to different basins) or a list of
        lists for paths (for basins containing the same information,
        commonly used for relative and absolute paths).
    """
    path_out = pathlib.Path(path_out)
    if not basin_paths:
        warnings.warn(f"Creating basin-based file '{path_out}' without any "
                      f"basins, since the list `basin_paths' is empty!",
                      CreatingFileWithoutBasinWarning)
    with HDF5Writer(path_out, mode="w") as hw:
        # Get the metadata from the first available basin path

        for bp in basin_paths:
            if isinstance(bp, (str, pathlib.Path)):
                # We have a single basin file
                bps = [bp]
            else:  # list or tuple
                bps = bp

            # We need to make sure that we are not resolving a relative
            # path to the working directory when we copy over data. Get
            # a representative path for metadata extraction.
            for pp in bps:
                pp = pathlib.Path(pp)
                if pp.is_absolute() and pp.exists():
                    prep = pp
                    break
                else:
                    # try relative path
                    prel = pathlib.Path(path_out).parent / pp
                    if prel.exists():
                        prep = prel
                        break
            else:
                prep = None

            # Copy the metadata from the representative path.
            if prep is not None:
                # copy metadata
                with h5py.File(prep, libver="latest") as h5:
                    copy_metadata(h5_src=h5, h5_dst=hw.h5)
                    # extract features
                    features = sorted(h5["events"].keys())
                name = prep.name
            else:
                features = None
                name = bps[0]

            # Finally, write the basin.
            hw.store_basin(name=name,
                           paths=bps,
                           features=features,
                           description=f"Created by dcnum {version}",
                           )


def copy_metadata(h5_src: h5py.File,
                  h5_dst: h5py.File,
                  copy_basins=True):
    """Copy attributes, tables, logs, and basins from one H5File to another

    Notes
    -----
    Metadata in `h5_dst` are never overridden, only metadata that
    are not defined already are added.
    """
    # compress data
    ds_kwds = set_default_filter_kwargs()
    # set attributes
    src_attrs = dict(h5_src.attrs)
    for kk in src_attrs:
        h5_dst.attrs.setdefault(kk, src_attrs[kk])
    copy_data = ["logs", "tables"]
    if copy_basins:
        copy_data.append("basins")
    # copy other metadata
    for topic in copy_data:
        if topic in h5_src:
            for key in h5_src[topic]:
                h5_dst.require_group(topic)
                if key not in h5_dst[topic]:
                    data = h5_src[topic][key][:]
                    if data.size:  # ignore empty datasets
                        if data.dtype == np.dtype("O"):
                            # convert variable-length strings to fixed-length
                            max_length = max([len(line) for line in data])
                            data = np.asarray(data, dtype=f"S{max_length}")
                        ds = h5_dst[topic].create_dataset(
                            name=key,
                            data=data,
                            **ds_kwds
                        )
                        # help with debugging and add some meta-metadata
                        ds.attrs.update(h5_src[topic][key].attrs)
                        soft_strgs = [ds.attrs.get("software"),
                                      f"dcnum {version}"]
                        soft_strgs = [s for s in soft_strgs if s is not None]
                        ds.attrs["software"] = " | ".join(soft_strgs)


def set_default_filter_kwargs(ds_kwds=None, compression=True):
    if ds_kwds is None:
        ds_kwds = {}
    if compression:
        # compression
        for key, val in dict(hdf5plugin.Zstd(clevel=5)).items():
            ds_kwds.setdefault(key, val)
    # checksums
    ds_kwds.setdefault("fletcher32", True)
    return ds_kwds
