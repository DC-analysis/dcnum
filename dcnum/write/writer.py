import hashlib
import json
import pathlib
from typing import List

import h5py
import hdf5plugin
import numpy as np


class HDF5Writer:
    def __init__(self, path, mode="a", ds_kwds=None):
        """Write deformability cytometry HDF5 data"""
        self.h5 = h5py.File(path, mode=mode, libver="latest")
        self.events = self.h5.require_group("events")
        if ds_kwds is None:
            ds_kwds = {}
        for key, val in dict(hdf5plugin.Zstd(clevel=5)).items():
            ds_kwds.setdefault(key, val)
        ds_kwds.setdefault("fletcher32", True)
        self.ds_kwds = ds_kwds

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.h5.close()

    @staticmethod
    def get_best_nd_chunks(item_shape):
        """Return best chunks for image data

        Chunking has performance implications. It’s recommended to keep the
        total size of your chunks between 10 KiB and 1 MiB. This number defines
        the maximum chunk size as well as half the maximum cache size for each
        dataset.
        """
        num_bytes = 1024**2  # between 10KiB and 1 MiB
        if len(item_shape) == 0:
            # scalar feature
            chunk_size_int = 10000
        else:
            event_size = np.prod(item_shape) * np.dtype(np.uint8).itemsize
            chunk_size = num_bytes / event_size
            chunk_size_int = max(1, int(np.floor(chunk_size)))
        return tuple([chunk_size_int] + list(item_shape))

    def require_feature(self, feat, item_shape, dtype, ds_kwds=None):
        """Create a new feature in the "events" group"""

        if ds_kwds is None:
            ds_kwds = {}
        for key in self.ds_kwds:
            ds_kwds.setdefault(key, self.ds_kwds[key])
        if feat not in self.events:
            dset = self.events.create_dataset(
                feat,
                shape=tuple([0] + list(item_shape)),
                dtype=dtype,
                maxshape=tuple([None] + list(item_shape)),
                chunks=self.get_best_nd_chunks(item_shape),
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
                    description: str | None = None,
                    ):
        """Write an HDF5-based file basin

        Parameters
        ----------
        name: str
            basin name; Names do not have to be unique.
        paths: list of str or pathlib.Path
            location(s) of the basin
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
                                          dtype=data.dtype)
        dsize = data.shape[0]
        ds.resize(offset + dsize, axis=0)
        ds[offset:offset + dsize] = data
