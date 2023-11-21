import collections
import pathlib
import threading
import time

from .writer import HDF5Writer


class DequeWriterThread(threading.Thread):
    def __init__(self,
                 path_out: pathlib.Path,
                 dq: collections.deque,
                 ds_kwds: dict = None,
                 mode: str = "a",
                 *args, **kwargs):
        """Convenience class for writing to data outside the main loop

        Parameters
        ----------
        path_out:
            Path to the output HDF5 file
        dq:
            `collections.deque` object from which data are taken
            using `popleft()`.
        """
        super(DequeWriterThread, self).__init__(*args, **kwargs)
        if mode == "w":
            path_out.unlink(missing_ok=True)
        self.writer = HDF5Writer(path_out, mode=mode, ds_kwds=ds_kwds)
        self.dq = dq
        self.may_stop_loop = False
        self.must_stop_loop = False

    def abort_loop(self):
        """Force aborting the loop as soon as possible"""
        self.must_stop_loop = True

    def finished_when_queue_empty(self):
        """Stop the loop as soon as `self.dq` is empty"""
        self.may_stop_loop = True

    def run(self):
        while True:
            if self.must_stop_loop:
                break
            elif len(self.dq):
                feat, data = self.dq.popleft()
                self.writer.store_feature_chunk(feat=feat, data=data)
            elif self.may_stop_loop:
                break
            else:
                # wait for the next item to arrive
                time.sleep(.1)
        self.writer.close()
