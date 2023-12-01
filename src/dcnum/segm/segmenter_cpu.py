import abc
import multiprocessing as mp
import time
import threading
from typing import Dict

import numpy as np

from .segmenter import Segmenter


# All subprocesses should use 'spawn' to avoid issues with threads
# and 'fork' on POSIX systems.
mp_spawn = mp.get_context('spawn')


class CPUSegmenter(Segmenter, abc.ABC):
    hardware_processor = "cpu"

    def __init__(self,
                 *,
                 num_workers: int = None,
                 kwargs_mask: Dict = None,
                 debug: bool = False,
                 **kwargs):
        """CPU base segmenter

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for `segment_approach`
            defined in the subclass.
        """
        super(CPUSegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)
        self.num_workers = num_workers or mp.cpu_count()
        self.mp_image_raw = None
        self._mp_image_np = None
        self.mp_labels_raw = None
        self._mp_labels_np = None
        self._mp_workers = []
        # Image shape of the input array
        self.image_shape = None
        # Processing control values
        # The batch worker number helps to communicate with workers.
        # <-1: exit
        # -1: idle
        # 0: start
        # >0: this number of workers finished a batch
        self.mp_batch_worker = mp_spawn.Value("i", 0)
        # The iteration of the process (increment to wake workers)
        # (raw value, because only this thread changes it)
        self.mp_batch_index = mp_spawn.RawValue("i", -1)
        # Tells the workers to stop
        # (raw value, because only this thread changes it)
        self.mp_shutdown = mp_spawn.RawValue("i", 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join_workers()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        # This is important when using "spawn" to create new processes,
        # because then the entire object gets pickled and some things
        # cannot be pickled!
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["logger"]
        del state["_mp_image_np"]
        del state["_mp_labels_np"]
        del state["_mp_workers"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

    @staticmethod
    def _create_shared_array(image_shape, batch_size, dtype):
        """Return raw and numpy-view on shared array

        Parameters
        ----------
        image_shape: tuple of int
            Shape of one single image in the array
        batch_size: int
            Number of images in the array
        dtype:
            ctype, e.g. `np.ctypeslib.ctypes.c_uint8`
            or `np.ctypeslib.ctypes.c_bool`
        """
        sx, sy = image_shape
        sa_raw = mp_spawn.RawArray(dtype, int(sx * sy * batch_size))
        # Convert the RawArray to something we can write to fast
        # (similar to memory view, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        sa_np = np.ctypeslib.as_array(sa_raw).reshape(batch_size, sx, sy)
        return sa_raw, sa_np

    @property
    def image_array(self):
        return self._mp_image_np

    @property
    def labels_array(self):
        return self._mp_labels_np

    @property
    def mask_array(self):
        return np.array(self._mp_labels_np, dtype=bool)

    def join_workers(self):
        """Ask all workers to stop and join them"""
        if self._mp_workers:
            self.mp_shutdown.value = 1
            [w.join() for w in self._mp_workers]

    def segment_batch(self,
                      image_data: np.ndarray,
                      start: int = None,
                      stop: int = None):
        """Perform batch segmentation of `image_data`

        Parameters
        ----------
        image_data: 3d np.ndarray
            The time-series image data. First axis is time.
        start: int
            First index to analyze in `image_data`
        stop: int
            Index after the last index to analyze in `image_data`

        Notes
        -----
        - If the segmentation algorithm only accepts background-corrected
          images, then `image_data` must already be background-corrected.
        """
        if stop is None or start is None:
            start = 0
            stop = len(image_data)

        batch_size = stop - start
        size = np.prod(image_data.shape[1:]) * batch_size

        if self.image_shape is None:
            self.image_shape = image_data[0].shape

        if self._mp_image_np is not None and self._mp_image_np.size != size:
            # reset image data
            self._mp_image_np = None
            self._mp_labels_np = None
            # TODO: If only the batch_size changes, don't
            #  reinitialize the workers. Otherwise, the final rest of
            #  analyzing a dataset would always take a little longer.
            self.join_workers()
            self._mp_workers = []
            self.mp_batch_index.value = -1
            self.mp_shutdown.value = 0

        if self._mp_image_np is None:
            self.mp_image_raw, self._mp_image_np = self._create_shared_array(
                image_shape=self.image_shape,
                batch_size=batch_size,
                dtype=np.ctypeslib.ctypes.c_int32,
            )

        if self._mp_labels_np is None:
            self.mp_labels_raw, self._mp_labels_np = self._create_shared_array(
                image_shape=self.image_shape,
                batch_size=batch_size,
                dtype=np.ctypeslib.ctypes.c_uint16,
            )

        # populate image data
        self._mp_image_np[:] = image_data[start:stop]

        # Create the workers
        if self.debug:
            worker_cls = CPUSegmenterWorkerThread
            num_workers = 1
        else:
            worker_cls = CPUSegmenterWorkerProcess
            num_workers = min(self.num_workers, image_data.shape[0])

        if not self._mp_workers:
            step_size = batch_size // num_workers
            rest = batch_size % num_workers
            wstart = 0
            for ii in range(num_workers):
                # Give every worker the same-sized workload and add one
                # from the rest until there is no more.
                wstop = wstart + step_size
                if rest:
                    wstop += 1
                    rest -= 1
                args = [self, wstart, wstop]
                w = worker_cls(*args)
                w.start()
                self._mp_workers.append(w)
                wstart = wstop

        # Match iteration number with workers
        self.mp_batch_index.value += 1

        # Tell workers to get going
        self.mp_batch_worker.value = 0

        # Wait for all workers to complete
        while self.mp_batch_worker.value != num_workers:
            time.sleep(.01)

        return self._mp_labels_np


class CPUSegmenterWorker:
    def __init__(self,
                 segmenter,
                 sl_start: int,
                 sl_stop: int,
                 ):
        """Worker process for CPU-based segmentation

        Parameters
        ----------
        segmenter: CPUSegmenter
            The segmentation instance
        sl_start: int
            Start of slice of input array to process
        sl_stop: int
            Stop of slice of input array to process
        """
        # Must call super init, otherwise Thread or Process are not initialized
        super(CPUSegmenterWorker, self).__init__()
        self.segmenter = segmenter
        # Value incrementing the batch index. Starts with 0 and is
        # incremented every time :func:`Segmenter.segment_batch` is
        # called.
        self.batch_index = segmenter.mp_batch_index
        # Value incrementing the number of workers that have finished
        # their part of the batch.
        self.batch_worker = segmenter.mp_batch_worker
        # Shutdown bit tells workers to stop when set to != 0
        self.shutdown = segmenter.mp_shutdown
        # The image data for segmentation
        self.image_data_raw = segmenter.mp_image_raw
        # Boolean mask array
        self.labels_data_raw = segmenter.mp_labels_raw
        # The shape of one image
        self.image_shape = segmenter.image_shape
        self.sl_start = sl_start
        self.sl_stop = sl_stop

    def run(self):
        # print(f"Running {self} in PID {os.getpid()}")
        # We have to create the numpy-versions of the mp.RawArrays here,
        # otherwise we only get some kind of copy in the new process
        # when we use "spawn" instead of "fork".
        labels_data = np.ctypeslib.as_array(self.labels_data_raw).reshape(
            -1, self.image_shape[0], self.image_shape[1])
        image_data = np.ctypeslib.as_array(self.image_data_raw).reshape(
            -1, self.image_shape[0], self.image_shape[1])

        idx = self.sl_start
        itr = 0  # current iteration (incremented when we reach self.sl_stop)
        while True:
            correct_iter = self.batch_index.value == itr
            if correct_iter:
                if idx == self.sl_stop:
                    # We finished processing everything.
                    itr += 1  # prevent running same things again
                    idx = self.sl_start  # reset counter for next batch
                    with self.batch_worker:
                        self.batch_worker.value += 1
                else:
                    labels_data[idx, :, :] = self.segmenter.segment_frame(
                        image_data[idx])
                    idx += 1
            elif self.shutdown.value:
                break
            else:
                # Wait for more data to arrive
                time.sleep(.01)


class CPUSegmenterWorkerProcess(CPUSegmenterWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super(CPUSegmenterWorkerProcess, self).__init__(*args, **kwargs)


class CPUSegmenterWorkerThread(CPUSegmenterWorker, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(CPUSegmenterWorkerThread, self).__init__(*args, **kwargs)
