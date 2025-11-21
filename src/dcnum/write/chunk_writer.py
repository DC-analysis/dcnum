import collections
import logging
import multiprocessing as mp
import pathlib
import threading
import time

import h5py

from .writer import HDF5Writer


class ChunkWriter(threading.Thread):
    def __init__(self,
                 path_out: pathlib.Path | h5py.File,
                 dq: collections.deque,
                 write_queue_size: mp.Value,
                 ds_kwds: dict = None,
                 mode: str = "a",
                 parent_logger: logging.Logger = None,
                 *args, **kwargs):
        """Convenience class for writing to data outside the main loop

        Data are numpy arrays collected from a `dequeue` object

        Parameters
        ----------
        path_out:
            Path to the output HDF5 file
        dq: collections.deque
            `collections.deque` object from which data are taken
            using `popleft()`.
        write_queue_size:
            Multiprocessing value to which the size of `dq` is written
            periodically
        ds_kwds:
            keyword arguments for dataset creation,
            passed to :class:`.HDF5Writer`
        mode:
            HDF5 file opening mode, passed to :class:`.HDF5Writer`
        """
        super(ChunkWriter, self).__init__(*args, **kwargs)
        if parent_logger is None:
            self.logger = logging.getLogger("dcnum.write.ChunkWriter")
        else:
            self.logger = parent_logger.getChild("ChunkWriter")
        if mode == "w":
            path_out.unlink(missing_ok=True)
        self.writer = HDF5Writer(path_out, mode=mode, ds_kwds=ds_kwds)
        self.dq = dq
        self.may_stop_loop = False
        self.must_stop_loop = False
        self.write_queue_size = write_queue_size

    def abort_loop(self):
        """Force aborting the loop as soon as possible"""
        self.must_stop_loop = True

    def finished_when_queue_empty(self):
        """Stop the loop as soon as `self.dq` is empty"""
        self.may_stop_loop = True

    def run(self):
        time_tot = 0
        while True:
            ldq = len(self.dq)
            self.write_queue_size.value = ldq
            if self.must_stop_loop:
                break
            elif ldq:
                t0 = time.perf_counter()
                for _ in range(ldq):
                    feat, data = self.dq.popleft()
                    self.writer.store_feature_chunk(feat=feat, data=data)
                    self.write_queue_size.value -= 1
                time_tot += time.perf_counter() - t0
            elif self.may_stop_loop:
                break
            else:
                # wait for the next item to arrive
                time.sleep(.1)
        self.logger.info(f"Disk time: {time_tot:.1f}s")
        self.writer.close()
