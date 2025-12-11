import abc
import multiprocessing as mp
import time
import threading
from typing import Dict

import numpy as np

from ..common import cpu_count
from ..os_env_st import RequestSingleThreaded, confirm_single_threaded

from .segmenter import Segmenter, assert_labels


# All subprocesses should use 'spawn' to avoid issues with threads
# and 'fork' on POSIX systems.
mp_spawn = mp.get_context('spawn')


class MPOSegmenter(Segmenter, abc.ABC):
    hardware_processor = "cpu"

    def __init__(self,
                 *,
                 num_workers: int = None,
                 kwargs_mask: Dict = None,
                 debug: bool = False,
                 **kwargs):
        """Segmenter with multiprocessing operation

        Parameters
        ----------
        num_workers
            Number of workers (processes) to spawn
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_labels`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for ``segment_algorithm``
            defined in the subclass.
        """
        super(MPOSegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)
        self.num_workers = num_workers or cpu_count()

        self.slot_list = None
        """List of ChunkSlot instances"""

        self.mp_slot_index = mp_spawn.Value("I", 0)
        """The slot that is currently being worked on"""

        self.mp_active = mp_spawn.Event()
        """Event that defines whether the workers are allowed to do work"""

        self.mp_num_workers_done = mp_spawn.Value("I", 0)
        """Number of workers that are done processing the slot"""

        self.mp_shutdown = mp_spawn.Event()
        """Shutdown event tells workers to stop when set to != 0"""

        # workers
        self._workers = []

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
        del state["_workers"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

    def join_workers(self):
        """Ask all workers to stop and join them"""
        if self._workers:
            self.mp_shutdown.set()
            for w in self._workers:
                w.join()
                if hasattr(w, "close"):
                    w.close()
        self._workers.clear()

    def reinitialize_workers(self, slot_list):
        self.join_workers()
        self.slot_list = slot_list
        self.mp_shutdown.clear()

        if self.debug:
            worker_cls = MPOSegmenterWorkerThread
            num_workers = 1
            self.logger.debug("Running with one worker in main thread")
        else:
            worker_cls = MPOSegmenterWorkerProcess
            num_workers = self.num_workers
            self.logger.debug(f"Running with {num_workers} workers")

        chunk_size = self.slot_list[0].length
        self.num_workers = min(num_workers, chunk_size)
        step_size = chunk_size // self.num_workers
        rest = chunk_size % self.num_workers
        w_start = 0
        for ii in range(self.num_workers):
            # Give every worker the same-sized workload and add one
            # from the rest until there is no more.
            w_stop = w_start + step_size
            if rest:
                w_stop += 1
                rest -= 1
            args = [self, w_start, w_stop]
            w = worker_cls(*args)
            w.start()
            self._workers.append(w)
            w_start = w_stop

    def segment_batch(self,
                      images: np.ndarray,
                      bg_off: np.ndarray = None,
                      ):
        """Perform batch segmentation of `images`

        Before segmentation, an optional background offset correction with
        ``bg_off`` is performed. After segmentation, mask postprocessing is
        performed according to the segmenter class definition.

        Parameters
        ----------
        images: 3d np.ndarray of shape (N, Y, X)
            The time-series image data. First axis is time.
        bg_off: 1D np.ndarray of length N
            Optional 1D numpy array with background offset

        Notes
        -----
        - If the segmentation algorithm only accepts background-corrected
          images, then `images` must already be background-corrected,
          except for the optional `bg_off`.
        """
        from ..logic.chunk_slot_data import ChunkSlotData
        available_features = ["image_bg"]

        if bg_off is not None:
            if not self.requires_background_correction:
                raise ValueError(f"The segmenter {self.__class__.__name__} "
                                 f"does not employ background correction, "
                                 f"but the `bg_off` keyword argument was "
                                 f"passed to `segment_batch`. Please check "
                                 f"your analysis pipeline.")

        if bg_off is not None:
            available_features.append("bg_off")

        cs = ChunkSlotData(shape=images.shape,
                           available_features=available_features)

        if self.requires_background_correction:
            cs.image_corr[:] = images
            if bg_off is not None:
                cs.bg_off[:] = bg_off
        else:
            cs.image[:] = images

        cs.chunk = 0

        self.segment_chunk(0, [cs])
        return cs.labels[:]

    def segment_chunk(self,
                      chunk: int,  # noqa: F821
                      slot_list: list,
                      ):
        """Segment the image data of one `ChunkSlot`

        Parameters
        ----------
        chunk:
            The data chunk index to perform segmentation on
        slot_list:
            List of `ChunkSlotData` instances (e.g. `SlotRegister.slots`)

        Returns
        -------
        labels: np.array
            The `chunk_slot.labels` numpy view on the shared labels array.
        """
        self.mp_active.clear()

        # Find the slot that we are supposed to be working on.
        for cs in slot_list:
            if cs.chunk == chunk:
                break
        else:
            raise ValueError(f"Could not find slot for {chunk=}")

        # Prepare everything for the workers, so they can already start
        # segmenting when they are created.
        slot_index = slot_list.index(cs)
        self.mp_slot_index.value = slot_index

        self.mp_num_workers_done.value = 0
        self.mp_active.set()

        if self.slot_list is not None:
            for cs1, cs2 in zip(slot_list, self.slot_list):
                if cs1 is not cs2:
                    # Something changed. We have to respawn the workers.
                    self.slot_list = slot_list
                    self.reinitialize_workers(slot_list)
                    break
        else:
            self.slot_list = slot_list
            self.reinitialize_workers(slot_list)

        while self.mp_num_workers_done.value != self.num_workers:
            time.sleep(.01)

        self.mp_active.clear()
        return cs.labels

    def segment_single(self, image, bg_off: float = None):
        """Return the integer label image for an input image

        Before segmentation, an optional background offset correction with
        ``bg_off`` is performed. After segmentation, mask postprocessing is
        performed according to the class definition.
        """
        segm_wrap = self.segment_algorithm_wrapper()

        # optional subtraction of background offset
        if bg_off is not None:
            image = image - bg_off

        # obtain mask or label
        mol = segm_wrap(image)

        # optional mask/label postprocessing
        if self.mask_postprocessing:
            labels = self.process_labels(mol, **self.kwargs_mask)
        else:
            labels = assert_labels(mol)
        return labels

    def close(self):
        self.join_workers()


class MPOSegmenterWorker:
    def __init__(self,
                 segmenter,
                 sl_start: int,
                 sl_stop: int,
                 ):
        """Worker process for CPU-based segmentation

        Parameters
        ----------
        segmenter: .segmenter_mpo.MPOSegmenter
            The segmentation instance
        sl_start: int
            Start of slice of input array to process
        sl_stop: int
            Stop of slice of input array to process
        """
        # Must call super init, otherwise Thread or Process are not initialized
        super(MPOSegmenterWorker, self).__init__()
        self.segmenter = segmenter

        self.slot_list = segmenter.slot_list
        """List of ChunkSlot instances"""

        self.mp_slot_index = segmenter.mp_slot_index
        """The slot that is currently being worked on"""

        self.mp_active = segmenter.mp_active
        """Whether the workers are allowed to do work"""

        self.mp_num_workers_done = segmenter.mp_num_workers_done
        """Number of workers that are done processing the slot"""

        self.mp_shutdown = segmenter.mp_shutdown
        """Shutdown bit tells workers to stop when set to != 0"""

        self.sl_start = sl_start
        self.sl_stop = sl_stop

    def run(self):
        # confirm single-threadedness (prints to log)
        confirm_single_threaded()
        last_chunk = -1

        while True:
            if self.mp_shutdown.is_set():
                break
            if self.mp_active.wait(timeout=1):
                # Get the current slot
                cs = self.slot_list[self.mp_slot_index.value]
                if cs.chunk == last_chunk:
                    # We processed this chunk already
                    time.sleep(.01)
                    continue
                elif self.sl_start >= cs.length:
                    # We have no data to process
                    pass
                else:
                    if self.segmenter.requires_background_correction:
                        images = cs.image_corr
                        bg_offs = cs.bg_off
                    else:
                        images = cs.image
                        bg_offs = None

                    # Iterate over the chunks in that slot
                    for idx in range(self.sl_start,
                                     min(self.sl_stop, cs.length)):
                        cs.labels[idx] = self.segmenter.segment_single(
                            image=images[idx],
                            bg_off=None if bg_offs is None else bg_offs[idx],
                        )

                with self.mp_num_workers_done:
                    self.mp_num_workers_done.value += 1

                last_chunk = cs.chunk


class MPOSegmenterWorkerProcess(MPOSegmenterWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super(MPOSegmenterWorkerProcess, self).__init__(*args, **kwargs)

    def start(self):
        # Set all relevant os environment variables such libraries in the
        # new process only use single-threaded computation.
        with RequestSingleThreaded():
            mp_spawn.Process.start(self)


class MPOSegmenterWorkerThread(MPOSegmenterWorker, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(MPOSegmenterWorkerThread, self).__init__(*args, **kwargs)
