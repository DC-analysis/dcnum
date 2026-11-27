"""Lossless compression of .rtdc files with optional image cropping"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
import copy
import json
import logging
import multiprocessing as mp
import pathlib
from typing import Literal
import uuid

import numpy as np

from .._version import version
from ..common import cpu_count, h5py
from ..read import HDF5Data
from ..read.hdf5_data import get_measurement_identifier
from . import chop
from .writer import HDF5Writer, copy_metadata, get_compression_kwargs


logger = logging.getLogger(__name__)

#: Features required for cropping the "image" feature
CROP_REQUIRED_FEATURES = ["frame", "image_bg", "mask", "size_x", "size_y"]

#: Approximate size of the uncompressed data processed in one worker job
JOB_SIZE_BYTES = 16 * 1024 ** 2

#: HDF5Data instance of the input file in the worker processes
_worker_hd: HDF5Data | None = None


def compress(path_in: pathlib.Path | str,
             path_out: pathlib.Path | str,
             basin_paths: list[pathlib.Path | str] | None = None,
             crop_event_images: bool = False,
             crop_method: Literal["box", "dilation"] = "dilation",
             pad_um: float = 1.5,
             compression: str = "zstd-5",
             num_procs: int | None = None,
             progress_callback: Callable[[float], None] | None = None,
             ) -> pathlib.Path:
    """Write a compressed copy of an .rtdc file

    All features of the input file, including the features from its
    file basins, are written to the output file, such that the output
    file contains all data of the measurement. Internal basins (e.g. a
    mapped "image_bg" feature) are kept as internal basins and the basin
    definitions of the input file are copied.

    Parameters
    ----------
    path_in:
        input .rtdc file
    path_out:
        output .rtdc file; must not exist
    basin_paths:
        locations of file basins of the input file (e.g. if the basin
        file was moved); Each path must match one of the file basins of
        the input file (by its measurement identifier or, for basins
        without identifier, by its path or file name).
    crop_event_images:
        store the "image" feature as chopped data (internal
        "h5datasetchop" basin, as in ``dclab-compress
        --crop-event-images``); this requires the features "frame",
        "image_bg", "mask", "size_x", and "size_y"
    crop_method:
        how the event images are cropped; "box" keeps the bounding box
        of each event (plus padding), "dilation" (default) only keeps the
        pixels within `pad_um` of the event mask
    pad_um:
        padding around the events for cropping [µm]
    compression:
        compression algorithm, either "none" or one of "zstd-1" (fast)
        to "zstd-9" (small files)
    num_procs:
        number of worker processes; defaults to the number of physical
        CPUs (see :func:`dcnum.common.cpu_count`); if
        set to 1, all data are processed in the current process
    progress_callback:
        function that is called with the progress (between 0 and 1)

    Returns
    -------
    path_out:
        the output path

    Raises
    ------
    FileExistsError
        if `path_out` exists
    ValueError
        if the "image" feature is not available (also not in the file
        basins), if a path in `basin_paths` is not a file basin of the
        input file, or if features required for cropping are missing
    """
    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)
    if path_out.exists():
        raise FileExistsError(f"Output file '{path_out}' exists")
    ds_kwds = get_compression_kwargs(compression)
    # check this early
    chop.get_dilation_structure(crop_method, pixel_size=1, pad_um=pad_um)

    hd = get_input_data(path_in, basin_paths)
    try:
        check_input_data(hd, crop_event_images=crop_event_images)
    except BaseException:
        hd.close()
        raise

    if num_procs is None:
        num_procs = cpu_count()

    # The workers only need the feature data. Logs and tables can be
    # large, and passing them to the workers slows down their start.
    hd_worker = HDF5Data(hd.path,
                         md5_5m=hd.md5_5m,
                         meta=copy.deepcopy(hd.meta),
                         logs={},
                         tables={},
                         basins=copy.deepcopy(hd.basins))

    path_temp = path_out.with_name(path_out.name + "~")
    path_temp.unlink(missing_ok=True)
    try:
        if num_procs == 1:
            executor = InlineExecutor(initializer=_init_worker,
                                      initargs=(hd_worker,))
        else:
            executor = ProcessPoolExecutor(
                max_workers=num_procs,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(hd_worker,))
        with executor, h5py.File(path_temp, "w", libver="latest") as h5out:
            Compressor(hd=hd,
                       h5out=h5out,
                       executor=executor,
                       num_procs=num_procs,
                       ds_kwds=ds_kwds,
                       crop_event_images=crop_event_images,
                       crop_method=crop_method,
                       pad_um=pad_um,
                       progress_callback=progress_callback,
                       ).run()
            store_compress_log(
                hd=hd,
                h5out=h5out,
                ds_kwds=ds_kwds,
                params={"compression": compression,
                        "crop_event_images": crop_event_images,
                        "crop_method": crop_method,
                        "pad_um": pad_um,
                        "basin_paths": [str(p) for p in basin_paths or []],
                        })
    except BaseException:
        path_temp.unlink(missing_ok=True)
        raise
    finally:
        hd.close()
        hd_worker.close()
    path_temp.rename(path_out)
    return path_out


class Compressor:
    def __init__(self,
                 hd: HDF5Data,
                 h5out: h5py.File,  # type: ignore
                 executor,
                 num_procs: int,
                 ds_kwds: dict,
                 crop_event_images: bool,
                 crop_method: Literal["box", "dilation"],
                 pad_um: float,
                 progress_callback: Callable[[float], None] | None,
                 ):
        """Write the compressed data of `hd` to `h5out`

        The data are compressed in chunks by the worker processes of
        `executor` and written to `h5out` as raw chunks in this process.
        """
        self.hd = hd
        self.h5in = hd.h5
        self.h5out = h5out
        self.executor = executor
        self.ds_kwds = ds_kwds
        self.crop_event_images = crop_event_images
        self.crop_method = crop_method
        self.pad_um = pad_um
        self.progress_callback = progress_callback
        #: maximum number of pending jobs
        self.max_pending = 2 * num_procs
        #: pending compression jobs (future, dataset, offset, number of
        #: events, whether to account the events for the progress)
        self.pending_writes: deque[tuple] = deque()
        self.bytes_total = 0
        self.bytes_done = 0

    def run(self):
        """Compress all data"""
        h5in = self.h5in
        h5out = self.h5out
        copy_metadata(h5_src=h5in, h5_dst=h5out)

        # Basins of the input file
        internal_features = set()
        for key in h5in.get("basins", {}):
            bn_dict = json.loads("\n".join(
                ln.decode("utf-8") if isinstance(ln, bytes) else ln
                for ln in h5in["basins"][key][:]))
            if bn_dict["type"] == "internal":
                if bn_dict.get("format") == "h5datasetchop":
                    # The chopped features are written anew.
                    continue
                # Keep internal basins (e.g. "image_bg" mapped with a
                # "basinmap" feature).
                for feat in bn_dict.get("features") or []:
                    if feat not in internal_features:
                        self.copy_object(f"{bn_dict['paths'][0]}/{feat}")
                        internal_features.add(feat)
            self.copy_object(f"basins/{key}")

        # Features written to the "events" group
        features = []
        # (HDF5Data does not support iteration over its features)
        features_all = self.hd.keys()
        for feat in features_all:
            if feat in h5in["events"]:
                features.append(feat)
            elif feat.startswith("basinmap"):
                # A basin mapping feature of a basin file is only
                # meaningful in the basin file.
                continue
            elif feat not in internal_features:
                features.append(feat)
        if self.crop_event_images and "image" in features:
            features.remove("image")
        groups = [ft for ft in features
                  if isinstance(h5in["events"].get(ft), h5py.Group)]
        features = [ft for ft in features if ft not in groups]
        for grp in groups:
            # e.g. fluorescence traces in the "trace" group
            self.copy_object(f"events/{grp}")

        # progress
        sizes = {}
        for feat in features:
            sizes[feat] = get_data_size(get_feature_data(self.hd, feat))
        self.bytes_total = sum(sizes.values())
        if self.crop_event_images:
            self.bytes_total += get_data_size(
                get_feature_data(self.hd, "image"))

        for feat in features:
            self.compress_feature(feat)
        if self.crop_event_images:
            self.compress_chopped_image()
        self.wait_for_writes(0)

    def compress_chopped_image(self):
        """Write the "image" feature as chopped data"""
        hd = self.hd
        image = get_feature_data(hd, "image")
        roi_shape = image.shape[1:]
        num_events = len(image)
        dtype = image.dtype
        shapes = chop.get_event_shapes(size_x=hd["size_x"][:],
                                       size_y=hd["size_y"][:],
                                       pixel_size=hd.pixel_size,
                                       roi_shape=roi_shape,
                                       pad_um=self.pad_um)
        clusters, num_clusters = chop.get_clusters(shapes=shapes,
                                                   dtype=dtype)
        index = np.zeros((num_events, 5), dtype=np.int64)
        index[:, 4] = chop.get_sibling_links(hd["frame"][:])

        grp = self.h5out.require_group("basin_events/image")
        grp.attrs["inverse_background_feature"] = "image_bg"

        # datasets for the clusters
        datasets = {}
        chunk_sizes = {}
        written = {}
        caches = {}
        for cid in range(1, num_clusters + 1):
            cidx = clusters == cid
            ccount = int(np.sum(cidx))
            shy, shx = (int(s) for s in np.max(shapes[cidx], axis=0))
            chunk_size = min(ccount, chop.get_chunk_size((shy, shx), dtype))
            ds = grp.create_dataset(name=f"{cid - 1}",
                                    shape=(ccount, shy, shx),
                                    dtype=dtype,
                                    chunks=(chunk_size, shy, shx),
                                    **self.ds_kwds)
            set_image_attrs(ds)
            datasets[cid] = ds
            chunk_sizes[cid] = chunk_size
            written[cid] = 0
            caches[cid] = []
            index[cidx, 0] = cid - 1
            index[cidx, 1] = np.arange(ccount)

        def flush(cid, final=False):
            """Compress full chunks (all data if `final`)"""
            ds = datasets[cid]
            cache = caches[cid]
            size = chunk_sizes[cid]
            num = len(cache) if final else len(cache) // size * size
            if num:
                data = np.zeros((num,) + ds.shape[1:], dtype=dtype)
                for jj in range(num):
                    chopped = cache[jj]
                    data[jj, :chopped.shape[0], :chopped.shape[1]] = chopped
                del cache[:num]
                self.submit_write(ds, written[cid], data, account=False)
                written[cid] += num

        # chop the events in blocks
        event_bytes = int(np.prod(roi_shape)) * dtype.itemsize
        block_size = max(1, JOB_SIZE_BYTES // event_bytes)
        pending_chops = deque()
        starts = list(range(0, num_events, block_size))
        while starts or pending_chops:
            while starts and len(pending_chops) < self.max_pending:
                start = starts.pop(0)
                stop = min(start + block_size, num_events)
                pending_chops.append((start, stop, self.executor.submit(
                    _worker_chop_image,
                    start=start,
                    stop=stop,
                    shapes=shapes[start:stop],
                    pad_um=self.pad_um,
                    crop_method=self.crop_method)))
            start, stop, future = pending_chops.popleft()
            offsets, chopped = future.result()
            index[start:stop, 2:4] = offsets
            for ii, data in zip(range(start, stop), chopped):
                caches[clusters[ii]].append(data)
            for cid in caches:
                flush(cid)
            self.add_progress((stop - start) * event_bytes)
        for cid in caches:
            flush(cid, final=True)

        grp.create_dataset(
            name="index",
            data=index,
            maxshape=(None, 5),
            chunks=HDF5Writer.get_best_nd_chunks((5,), index.dtype),
            **self.ds_kwds)
        chop.store_chopped_basin(self.h5out, "image")

    def compress_feature(self, feat):
        """Write the feature `feat` to the "events" group"""
        data = get_feature_data(self.hd, feat)
        item_shape = data.shape[1:]
        dtype = data.dtype
        num_events = len(data)
        chunks = HDF5Writer.get_best_nd_chunks(item_shape, feat_dtype=dtype)
        ds = self.h5out.require_group("events").create_dataset(
            name=feat,
            shape=data.shape,
            dtype=dtype,
            maxshape=(None,) + item_shape,
            chunks=chunks,
            **self.ds_kwds)
        if len(item_shape) == 2:
            set_image_attrs(ds)
        chunk_bytes = chunks[0] * get_data_size(data) // max(1, num_events)
        chunks_per_job = max(1, JOB_SIZE_BYTES // max(1, chunk_bytes))
        job_size = chunks[0] * chunks_per_job
        for start in range(0, num_events, job_size):
            stop = min(start + job_size, num_events)
            self.wait_for_writes(self.max_pending)
            future = self.executor.submit(_worker_compress_feature,
                                          feat=feat,
                                          start=start,
                                          stop=stop,
                                          chunk_size=chunks[0],
                                          ds_kwds=self.ds_kwds)
            self.pending_writes.append((future, ds, start, stop - start,
                                        True))

    def copy_object(self, name):
        """Copy an HDF5 object (if it does not exist in the output)"""
        if name not in self.h5out:
            group, obj = name.rsplit("/", 1)
            h5py.h5o.copy(src_loc=self.h5in[group].id,
                          src_name=obj.encode(),
                          dst_loc=self.h5out.require_group(group).id,
                          dst_name=obj.encode())

    def add_progress(self, num_bytes):
        self.bytes_done += num_bytes
        if self.progress_callback is not None and self.bytes_total:
            self.progress_callback(min(1, self.bytes_done / self.bytes_total))

    def submit_write(self, ds, offset, data, account=True):
        """Compress `data` in a worker and write it to `ds` at `offset`"""
        self.wait_for_writes(self.max_pending)
        future = self.executor.submit(compress_chunks,
                                      data=data,
                                      chunk_size=ds.chunks[0],
                                      ds_kwds=self.ds_kwds)
        self.pending_writes.append((future, ds, offset, len(data), account))

    def wait_for_writes(self, max_pending):
        """Write the compressed data until `max_pending` jobs are left"""
        while len(self.pending_writes) > max_pending:
            future, ds, offset, num, account = self.pending_writes.popleft()
            write_raw_chunks(ds, offset, future.result())
            if account:
                self.add_progress(num * get_data_size(ds) // len(ds))


class InlineExecutor:
    def __init__(self, initializer=None, initargs=()):
        """Executor that runs the jobs immediately in this process"""
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, *args):
        _init_worker(None)

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def check_input_data(hd: HDF5Data, crop_event_images: bool):
    """Make sure that the input data can be compressed"""
    features = hd.keys()
    if "image" not in features:
        missing = [bn.get("paths") for ii, bn in enumerate(hd.basins)
                   if (bn["type"] == "file"
                       and hd.get_basin_data(ii)[0] is None)]
        msg = f"No image data found in '{hd.path}'"
        if missing:
            msg += (f"; the following file basins are not available "
                    f"(specify their locations as basin paths): {missing}")
        raise ValueError(msg)
    if crop_event_images:
        missing = [f for f in CROP_REQUIRED_FEATURES if f not in features]
        if missing:
            raise ValueError(
                f"Cropping event images requires the features "
                f"{CROP_REQUIRED_FEATURES}, but {missing} are not available "
                f"in '{hd.path}' (e.g. raw data that were not processed "
                f"yet)")


def compress_chunks(data: np.ndarray,
                    chunk_size: int,
                    ds_kwds: dict) -> list[tuple[int, int, bytes]]:
    """Compress data in chunks with HDF5 filters

    The data are written to an in-memory HDF5 dataset with the given
    chunk size and filters. The raw chunks can be written to a dataset
    with the same chunk size and filters using
    :func:`write_raw_chunks`.

    Returns
    -------
    raw_chunks:
        list of the event offset in `data`, the HDF5 filter mask, and
        the raw (compressed) chunk data for each chunk
    """
    num_chunks = int(np.ceil(len(data) / chunk_size))
    item_shape = data.shape[1:]
    with h5py.File(f"{uuid.uuid4()}.h5", "w",
                   driver="core", backing_store=False) as h5:
        ds = h5.create_dataset(name="data",
                               shape=(num_chunks * chunk_size,) + item_shape,
                               dtype=data.dtype,
                               chunks=(chunk_size,) + item_shape,
                               **ds_kwds)
        ds[:len(data)] = data
        raw_chunks = []
        for ii in range(num_chunks):
            offset = (ii * chunk_size,) + (0,) * len(item_shape)
            filter_mask, raw = ds.id.read_direct_chunk(offset)
            raw_chunks.append((ii * chunk_size, filter_mask, raw))
    return raw_chunks


def get_data_size(data) -> int:
    """Return the size of the uncompressed data in bytes"""
    return int(np.prod(data.shape)) * np.dtype(data.dtype).itemsize


def get_feature_data(hd: HDF5Data, feat: str):
    """Return an object from which the feature data can be sliced"""
    if feat in hd.h5["events"]:
        return hd.h5["events"][feat]
    elif feat in ["image", "image_bg", "mask"]:
        return hd.get_image_cache(feat).h5ds
    else:
        return hd[feat]


def get_input_data(path_in: pathlib.Path,
                   basin_paths: list[pathlib.Path | str] | None = None,
                   ) -> HDF5Data:
    """Return HDF5Data of the input file, using the given basin paths

    Each path in `basin_paths` is prepended to the paths of the file
    basins of the input file that it matches. File basins with a
    measurement identifier must have the same identifier, file basins
    without an identifier must refer to the same file or to a file
    with the same name.
    """
    hd0 = HDF5Data(path_in)
    if not basin_paths:
        return hd0
    basins = copy.deepcopy(hd0.basins)
    for pb in basin_paths:
        pb = pathlib.Path(pb).resolve()
        with h5py.File(pb, "r") as h5:
            mid = get_measurement_identifier(h5)
        matched = False
        for bn_dict in basins:
            if bn_dict["type"] != "file":
                continue
            bn_id = bn_dict.get("identifier")
            if bn_id is not None:
                # The basin is identified by its measurement identifier.
                is_match = bn_id == mid
            else:
                # The basin is identified by its location or file name.
                is_match = any(
                    is_same_file(path_in.parent / pp, pb)
                    or pathlib.Path(pp).name == pb.name
                    for pp in bn_dict["paths"])
            if is_match:
                bn_dict["paths"] = [str(pb)] + list(bn_dict["paths"])
                matched = True
        if not matched:
            raise ValueError(f"The file '{pb}' is not a file basin of "
                             f"'{path_in}'")
    # Pass copies, because `HDF5Data.close` clears these dictionaries.
    hd = HDF5Data(path_in,
                  md5_5m=hd0.md5_5m,
                  meta=copy.deepcopy(hd0.meta),
                  logs=copy.deepcopy(hd0.logs),
                  tables=copy.deepcopy(hd0.tables),
                  basins=basins)
    hd0.close()
    return hd


def is_same_file(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    """Whether two paths point to the same existing file"""
    try:
        return path_a.samefile(path_b)
    except OSError:
        return False


def set_image_attrs(ds):
    """Set the attributes for image data (e.g. for HDFView)"""
    ds.attrs.create('CLASS', np.bytes_('IMAGE'))
    ds.attrs.create('IMAGE_VERSION', np.bytes_('1.2'))
    ds.attrs.create('IMAGE_SUBCLASS', np.bytes_('IMAGE_GRAYSCALE'))


def store_compress_log(hd, h5out, ds_kwds, params):
    """Store the "dcnum-compress" log"""
    logs = h5out.require_group("logs")
    if "dcnum-compress" in logs:
        # The input file was compressed with dcnum before.
        logs.move("dcnum-compress", f"dcnum-compress_{hd.md5_5m}")
    data = {"dcnum version": version,
            "input file": hd.path.name,
            "input md5_5m": hd.md5_5m,
            } | params
    hw = HDF5Writer(h5out, ds_kwds=ds_kwds)
    hw.store_log("dcnum-compress", json.dumps(data, indent=2).split("\n"))


def write_raw_chunks(ds: h5py.Dataset,  # type: ignore
                     offset: int,
                     raw_chunks: list[tuple[int, int, bytes]]):
    """Write chunks from :func:`compress_chunks` to `ds` at `offset`"""
    zeros = (0,) * (len(ds.shape) - 1)
    for chunk_offset, filter_mask, raw in raw_chunks:
        ds.id.write_direct_chunk((offset + chunk_offset,) + zeros,
                                 raw,
                                 filter_mask)


def _init_worker(hd):
    global _worker_hd
    _worker_hd = hd


def _get_worker_hd() -> HDF5Data:
    """Return the input data of the worker process"""
    if _worker_hd is None:
        raise RuntimeError("The worker was not initialized")
    return _worker_hd


def _worker_chop_image(start, stop, shapes, pad_um, crop_method):
    """Chop the "image" data of the events from `start` to `stop`"""
    hd = _get_worker_hd()
    mask = get_feature_data(hd, "mask")[start:stop]
    offsets = chop.get_event_offsets(mask=mask,
                                     shapes=shapes,
                                     pixel_size=hd.pixel_size,
                                     pad_um=pad_um)
    structure = chop.get_dilation_structure(crop_method,
                                            pixel_size=hd.pixel_size,
                                            pad_um=pad_um)
    chopped = chop.chop_events(
        data=get_feature_data(hd, "image")[start:stop],
        data_bg=get_feature_data(hd, "image_bg")[start:stop],
        shapes=shapes,
        offsets=offsets,
        mask=mask,
        dilation_structure=structure)
    return offsets, chopped


def _worker_compress_feature(feat, start, stop, chunk_size, ds_kwds):
    """Compress the data of `feat` from `start` to `stop`"""
    data = get_feature_data(_get_worker_hd(), feat)[start:stop]
    return compress_chunks(data, chunk_size=chunk_size, ds_kwds=ds_kwds)
