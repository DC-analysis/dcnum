import hashlib
import json
import pathlib
from typing import Dict, List, Tuple
import warnings

import h5py
import hdf5plugin
import numpy as np

from ..read import HDF5Data
from .._version import version


class CreatingFileWithoutBasinWarning(UserWarning):
    """Issued when creating a basin-based dataset without basins"""
    pass


class IgnoringBasinTypeWarning(UserWarning):
    """Issued when a specific basin type is ignored"""
    pass


class HDF5Writer:
    def __init__(self,
                 # TODO: make this a mandatory argument when `path` is
                 #       properly removed
                 obj: h5py.File | pathlib.Path | str = None,
                 mode: str = "a",
                 ds_kwds: Dict = None,
                 path: pathlib.Path | str = None,
                 ):
        """Write deformability cytometry HDF5 data

        Parameters
        ----------
        obj: h5py.File | pathlib.Path | str
            object to instantiate the writer from; If this is already
            a :class:`h5py.File` object, then it is used, otherwise the
            argument is passed to :class:`h5py.File`
        mode: str
            opening mode when using :class:`h5py.File`
        ds_kwds: Dict
            keyword arguments with which to initialize new Datasets
            (e.g. compression)
        """
        if path is not None:
            obj = path
            warnings.warn("The `path` keyword argument is deprecated, use "
                          "`obj` instead",
                          DeprecationWarning)
        if isinstance(obj, h5py.File):
            self.h5 = obj
            self.h5_owned = False
        else:
            self.h5 = h5py.File(obj,
                                mode=mode,
                                libver="latest",
                                # Set chunk cache size to 3 MiB for each
                                # dataset to allow partial writes.
                                rdcc_nbytes=3145728,
                                )
            self.h5_owned = True
        self.events = self.h5.require_group("events")
        ds_kwds = set_default_filter_kwargs(ds_kwds)
        self.ds_kwds = ds_kwds

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.h5.flush()
        if self.h5_owned:
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

    def require_feature(self,
                        feat: str,
                        item_shape: Tuple[int],
                        feat_dtype: np.dtype,
                        ds_kwds: Dict = None,
                        group_name: str = "events"):
        """Create a new feature in the "events" group

        Parameters
        ----------
        feat: str
            name of the feature
        item_shape: Tuple[int]
            shape for one event of this feature, e.g. for a scalar
            event, the shape would be `(1,)` and for an image, the
            shape could be `(80, 300)`.
        feat_dtype: np.dtype
            dtype of the feature
        ds_kwds: Dict
            HDF5 Dataset keyword arguments (e.g. compression, fletcher32)
        group_name: str
            name of the HDF5 group where the feature should be written to;
            defaults to the "events" group, but a different group can be
            specified for storing e.g. internal  basin features.
        """
        if group_name == "events":
            egroup = self.events
        else:
            egroup = self.h5.require_group(group_name)

        if feat not in egroup:
            if ds_kwds is None:
                ds_kwds = {}
            for key in self.ds_kwds:
                ds_kwds.setdefault(key, self.ds_kwds[key])
            dset = egroup.create_dataset(
                feat,
                shape=tuple([0] + list(item_shape)),
                dtype=feat_dtype,
                maxshape=tuple([None] + list(item_shape)),
                chunks=self.get_best_nd_chunks(item_shape,
                                               feat_dtype=feat_dtype),
                **ds_kwds)
            if len(item_shape) == 2:
                dset.attrs.create('CLASS', np.bytes_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.bytes_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS',
                                  np.bytes_('IMAGE_GRAYSCALE'))
            offset = 0
        else:
            dset = egroup[feat]
            offset = dset.shape[0]
        return dset, offset

    def store_basin(self,
                    name: str,
                    paths: List[str | pathlib.Path] | None = None,
                    features: List[str] = None,
                    description: str | None = None,
                    mapping: np.ndarray = None,
                    internal_data: Dict | None = None,
                    ):
        """Write an HDF5-based file basin

        Parameters
        ----------
        name: str
            basin name; Names do not have to be unique.
        paths: list of str or pathlib.Path or None
            location(s) of the basin; must be None when storing internal
            data, a list of paths otherwise
        features: list of str
            list of features provided by `paths`
        description: str
            optional string describing the basin
        mapping: 1D array
            integer array with indices that map the basin dataset
            to this dataset
        internal_data: dict of ndarrays
            internal basin data to store; If this is set, then `features`
            and `paths` must be set to `None`.
        """
        bdat = {
            "description": description,
            "name": name,
        }

        if internal_data:
            if features is not None:
                raise ValueError("`features` must be set to None when storing "
                                 "internal basin features")
            if paths is not None:
                raise ValueError("`paths` must be set to None when storing "
                                 "internal basin features")
            # store the internal basin information
            for feat in internal_data:
                if feat in self.h5.require_group("basin_events"):
                    raise ValueError(f"Feature '{feat}' is already defined "
                                     f"as an internal basin feature")
                self.store_feature_chunk(feat=feat,
                                         data=internal_data[feat],
                                         group_name="basin_events")
            features = sorted(internal_data.keys())
            bdat["format"] = "h5dataset"
            bdat["paths"] = ["basin_events"]
            bdat["type"] = "internal"
        else:
            bdat["format"] = "hdf5"
            bdat["paths"] = [str(pp) for pp in paths]
            bdat["type"] = "file"

        # Explicit features stored in basin file
        if features is not None and len(features):
            bdat["features"] = features
        # Mapped basin information
        if mapping is not None:
            events = self.h5.require_group("events")
            # Reserve a mapping feature for this dataset
            for ii in range(10):  # basinmap0 to basinmap9
                bm_cand = f"basinmap{ii}"
                if bm_cand in events:
                    # There is a basin mapping defined in the file. Check
                    # whether it is identical to ours.
                    if np.all(events[bm_cand] == mapping):
                        # Great, we are done here.
                        feat_basinmap = bm_cand
                        break
                    else:
                        # This mapping belongs to a different basin,
                        # try the next mapping.
                        continue
                else:
                    # The mapping is not defined in the dataset, and we may
                    # write it to a new feature.
                    feat_basinmap = bm_cand
                    self.store_feature_chunk(feat=feat_basinmap, data=mapping)
                    break
            else:
                raise ValueError(
                    "You have exhausted the usage of mapped basins for "
                    "the current dataset. Please revise your analysis "
                    "pipeline.")
            bdat["mapping"] = feat_basinmap
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

    def store_feature_chunk(self, feat, data, group_name="events"):
        """Store feature data

        The "chunk" implies that always chunks of data are stored,
        never single events.
        """
        if feat == "mask" and data.dtype == bool:
            data = 255 * np.array(data, dtype=np.uint8)
        ds, offset = self.require_feature(feat=feat,
                                          item_shape=data.shape[1:],
                                          feat_dtype=data.dtype,
                                          group_name=group_name)
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
                    copy_basins(h5_src=h5, h5_dst=hw.h5)
                    # extract features
                    features = sorted(h5["events"].keys())
                    features = [f for f in features if
                                not f.startswith("basinmap")]
                name = prep.name
            else:
                features = None
                name = bps[0]

            # Write the basin data
            hw.store_basin(name=name,
                           paths=bps,
                           features=features,
                           description=f"Created by dcnum {version}",
                           )


def copy_basins(h5_src: h5py.File,
                h5_dst: h5py.File,
                internal_basins: bool = True
                ):
    """Reassemble basin data in the output file

    This does not just copy the datasets defined in the "basins"
    group, but it also loads the "basinmap?" features and stores
    them as new "basinmap?" features in the output file.
    """
    basins = HDF5Data.extract_basin_dicts(h5_src, check=False)
    hw = HDF5Writer(h5_dst)
    for bn_dict in basins:
        if bn_dict["type"] == "internal" and internal_basins:
            internal_data = {}
            for feat in bn_dict["features"]:
                internal_data[feat] = h5_src["basin_events"][feat]
            hw.store_basin(name=bn_dict["name"],
                           description=bn_dict["description"],
                           mapping=h5_src["events"][bn_dict["mapping"]][:],
                           internal_data=internal_data,
                           )
        elif bn_dict["type"] == "file":
            if bn_dict.get("mapping") is not None:
                mapping = h5_src["events"][bn_dict["mapping"]][:]
            else:
                mapping = None
            hw.store_basin(name=bn_dict["name"],
                           description=bn_dict["description"],
                           paths=bn_dict["paths"],
                           features=bn_dict["features"],
                           mapping=mapping,
                           )
        else:
            warnings.warn(f"Ignored basin of type '{bn_dict['type']}'",
                          IgnoringBasinTypeWarning)


def copy_features(h5_src: h5py.File,
                  h5_dst: h5py.File,
                  features: List[str],
                  mapping: np.ndarray = None,
                  ):
    """Copy feature data from one HDF5 file to another

    The feature must not exist in the destination file.

    Parameters
    ----------
    h5_src: h5py.File
        Input HDF5File containing `features` in the "events" group
    h5_dst: h5py.File
        Output HDF5File opened in write mode not containing `features`
    features: List[str]
        List of features to copy from source to destination
    mapping: 1D array
        If given, contains indices in the input file that should be
        written to the output file. If set to None, all features are written.
    """
    ei = h5_src["events"]
    eo = h5_dst.require_group("events")
    hw = HDF5Writer(h5_dst)
    for feat in features:
        if feat in eo:
            raise ValueError(f"Output file {h5_dst.filename} already contains "
                             f"the feature {feat}.")
        if not isinstance(ei[feat], h5py.Dataset):
            raise NotImplementedError(
                f"Only dataset-based features are supported here, not {feat}")
        if mapping is None:
            # Just copy the data as-is.
            h5py.h5o.copy(src_loc=ei.id,
                          src_name=feat.encode(),
                          dst_loc=eo.id,
                          dst_name=feat.encode(),
                          )
        else:
            # We have to perform mapping.
            # Since h5py is very slow at indexing with arrays,
            # we instead read the data in chunks from the input file,
            # and perform the mapping afterward using the numpy arrays.
            dsi = ei[feat]
            chunk_size = hw.get_best_nd_chunks(dsi[0].shape, dsi.dtype)[0]
            size_in = dsi.shape[0]
            start = 0
            while start < size_in:
                # Get a big chunk of data
                big_chunk = 10 * chunk_size
                stop = start + big_chunk
                data_in = dsi[start:stop]
                # Determine the indices that we need from that chunk.
                mapping_idx = (start <= mapping) * (mapping < stop)
                mapping_chunk = mapping[mapping_idx] - start
                data = data_in[mapping_chunk]
                # Note that HDF5 does its own caching, properly handling
                # partial chunk writes.
                hw.store_feature_chunk(feat, data)
                # increment start
                start = stop


def copy_metadata(h5_src: h5py.File,
                  h5_dst: h5py.File
                  ):
    """Copy attributes, tables, and logs from one H5File to another

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
