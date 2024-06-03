import abc
import multiprocessing as mp
import time
import threading
from typing import Dict

import numpy as np
import scipy.ndimage as ndi

from .segmenter import Segmenter


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
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for `segment_algorithm`
            defined in the subclass.
        """
        super(MPOSegmenter, self).__init__(kwargs_mask=kwargs_mask,
                                           debug=debug,
                                           **kwargs)
        self.num_workers = num_workers or mp.cpu_count()
        # batch input image data
        self.mp_image_raw = None
        self._mp_image_np = None
        # batch output image data
        self.mp_labels_raw = None
        self._mp_labels_np = None
        # batch image background offset
        self.mp_bg_off_raw = None
        self._mp_bg_off_np = None
        # workers
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
        del state["_mp_bg_off_np"]
        del state["_mp_workers"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

    @staticmethod
    def _create_shared_array(array_shape, batch_size, dtype):
        """Return raw and numpy-view on shared array

        Parameters
        ----------
        array_shape: tuple of int
            Shape of one single image in the array
        batch_size: int
            Number of images in the array
        dtype:
            numpy dtype
        """
        ctype = np.ctypeslib.as_ctypes_type(dtype)
        sa_raw = mp_spawn.RawArray(ctype,
                                   int(np.prod(array_shape) * batch_size))
        # Convert the RawArray to something we can write to fast
        # (similar to memory view, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        sa_np = np.ctypeslib.as_array(sa_raw).reshape(batch_size, *array_shape)
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
                      images: np.ndarray,
                      start: int = None,
                      stop: int = None,
                      bg_off: np.ndarray = None,
                      ):
        """Perform batch segmentation of `images`

        Before segmentation, an optional background offset correction with
        `bg_off` is performed. After segmentation, mask postprocessing is
        performed according to the class definition.

        Parameters
        ----------
        images: 3d np.ndarray
            The time-series image data. First axis is time.
        start: int
            First index to analyze in `images`
        stop: int
            Index after the last index to analyze in `images`
        bg_off: 1D np.ndarray
            Optional 1D numpy array with background offset

        Notes
        -----
        - If the segmentation algorithm only accepts background-corrected
          images, then `images` must already be background-corrected,
          except for the optional `bg_off`.
        """
        if stop is None or start is None:
            start = 0
            stop = len(images)

        batch_size = stop - start
        size = np.prod(images.shape[1:]) * batch_size

        if self.image_shape is None:
            self.image_shape = images[0].shape

        if self._mp_image_np is not None and self._mp_image_np.size != size:
            # reset image data
            self._mp_image_np = None
            self._mp_labels_np = None
            self._mp_bg_off_np = None
            # TODO: If only the batch_size changes, don't
            #  reinitialize the workers. Otherwise, the final rest of
            #  analyzing a dataset would always take a little longer.
            self.join_workers()
            self._mp_workers = []
            self.mp_batch_index.value = -1
            self.mp_shutdown.value = 0

        if bg_off is not None:
            if not self.requires_background_correction:
                raise ValueError(f"The segmenter {self.__class__.__name__} "
                                 f"does not employ background correction, "
                                 f"but the `bg_off` keyword argument was "
                                 f"passed to `segment_chunk`. Please check "
                                 f"your analysis pipeline.")
            # background offset
            if self._mp_bg_off_np is None:
                self.mp_bg_off_raw, self._mp_bg_off_np = \
                    self._create_shared_array(
                        array_shape=(stop - start,),
                        batch_size=batch_size,
                        dtype=np.float64)
            self._mp_bg_off_np[:] = bg_off[start:stop]

        # input images
        if self._mp_image_np is None:
            self.mp_image_raw, self._mp_image_np = self._create_shared_array(
                array_shape=self.image_shape,
                batch_size=batch_size,
                dtype=images.dtype,
            )
        self._mp_image_np[:] = images[start:stop]

        # output labels
        if self._mp_labels_np is None:
            self.mp_labels_raw, self._mp_labels_np = self._create_shared_array(
                array_shape=self.image_shape,
                batch_size=batch_size,
                dtype=np.uint16,
            )

        # Create the workers
        if self.debug:
            worker_cls = MPOSegmenterWorkerThread
            num_workers = 1
            self.logger.debug("Running with one worker in main thread")
        else:
            worker_cls = MPOSegmenterWorkerProcess
            num_workers = min(self.num_workers, images.shape[0])
            self.logger.debug(f"Running with {num_workers} workers")

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

    def segment_single(self, image, bg_off: float = None):
        """Return the integer label image for an input image

        Before segmentation, an optional background offset correction with
        `bg_off` is performed. After segmentation, mask postprocessing is
        performed according to the class definition.
        """
        segm_wrap = self.segment_algorithm_wrapper()
        # optional subtraction of background offset
        if bg_off is not None:
            image = image - bg_off
        # obtain mask or label
        mol = segm_wrap(image)
        if mol.dtype == bool:
            # convert mask to labels
            labels, _ = ndi.label(
                input=mol,
                structure=ndi.generate_binary_structure(2, 2))
        else:
            labels = mol
        # optional mask/label postprocessing
        if self.mask_postprocessing:
            labels = self.process_mask(labels, **self.kwargs_mask)
        return labels


class MPOSegmenterWorker:
    def __init__(self,
                 segmenter,
                 sl_start: int,
                 sl_stop: int,
                 ):
        """Worker process for CPU-based segmentation

        Parameters
        ----------
        segmenter: MPOSegmenter
            The segmentation instance
        sl_start: int
            Start of slice of input array to process
        sl_stop: int
            Stop of slice of input array to process
        """
        # Must call super init, otherwise Thread or Process are not initialized
        super(MPOSegmenterWorker, self).__init__()
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
        self.image_arr_raw = segmenter.mp_image_raw
        # Background data offset
        self.bg_off = segmenter.mp_bg_off_raw
        # Integer output label array
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
        labels_arr = np.ctypeslib.as_array(self.labels_data_raw).reshape(
            -1, self.image_shape[0], self.image_shape[1])
        image_arr = np.ctypeslib.as_array(self.image_arr_raw).reshape(
            -1, self.image_shape[0], self.image_shape[1])
        if self.bg_off is not None:
            bg_off_data = np.ctypeslib.as_array(self.bg_off)
        else:
            bg_off_data = None

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
                    if bg_off_data is None:
                        bg_off = None
                    else:
                        bg_off = bg_off_data[idx]

                    labels_arr[idx, :, :] = self.segmenter.segment_single(
                        image=image_arr[idx], bg_off=bg_off)
                    idx += 1
            elif self.shutdown.value:
                break
            else:
                # Wait for more data to arrive
                time.sleep(.01)


class MPOSegmenterWorkerProcess(MPOSegmenterWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super(MPOSegmenterWorkerProcess, self).__init__(*args, **kwargs)


class MPOSegmenterWorkerThread(MPOSegmenterWorker, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(MPOSegmenterWorkerThread, self).__init__(*args, **kwargs)
