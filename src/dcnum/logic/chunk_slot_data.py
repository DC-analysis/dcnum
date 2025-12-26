import ctypes
import multiprocessing as mp

import numpy as np


mp_spawn = mp.get_context("spawn")


class ChunkSlotData:
    def __init__(self, shape, available_features=None):
        self.shape = shape
        """3D shape of the chunk in this slot"""

        available_features = available_features or []
        self.length = shape[0]

        self._task_reserve_array = mp_spawn.RawArray(ctypes.c_bool,
                                                     self.length)
        self._task_reserve_lock = mp_spawn.Lock()

        self._task_progress_array = mp_spawn.RawArray(ctypes.c_bool,
                                                      self.length)

        self._state = mp_spawn.RawValue("u", "0")

        # Initialize with negative value to avoid ambiguities with first chunk.
        self._chunk = mp_spawn.Value(ctypes.c_long, -1, lock=False)

        # Initialize all shared arrays
        if self.length:
            array_length = int(np.prod(self.shape))

            # Image data
            self.mp_image = mp_spawn.RawArray(ctypes.c_uint8, array_length)

            if "image_bg" in available_features:
                self.mp_image_bg = mp_spawn.RawArray(ctypes.c_uint8,
                                                     array_length)

                self.mp_image_corr = mp_spawn.RawArray(ctypes.c_int16,
                                                       array_length)
            else:
                self.mp_image_bg = None
                self.mp_image_corr = None

            if "bg_off" in available_features:
                # background offset data
                self.mp_bg_off = mp_spawn.RawArray(ctypes.c_double,
                                                   self.length)
            else:
                self.mp_bg_off = None

            # Mask data
            self.mp_mask = mp_spawn.RawArray(ctypes.c_bool, array_length)

            # Label data
            self.mp_labels = mp_spawn.RawArray(ctypes.c_uint16, array_length)

        self._state.value = "i"

    @property
    def chunk(self):
        """Current chunk being analyzed"""
        return self._chunk.value

    @chunk.setter
    def chunk(self, value):
        self._chunk.value = value

    @property
    def state(self):
        """Current state of the slot

        Valid values are:

        - "0": construction of instance
        - "i": image loading (populates image, image_bg, image_corr, bg_off)
        - "s": segmentation (populates mask or labels)
        - "m": mask processing (takes data from mask and populates labels)
        - "l": label processing (modifies labels in-place)
        - "e": feature extraction (requires labels)
        - "w": writing
        - "d": done (slot can be repurposed for next chunk)
        - "n": not specified

        The pipeline workflow is:

            "0" -> "i" -> "s" -> "m" or "l" -> "e" -> "w" -> "d" -> "i" ...
        """
        return self._state.value

    @state.setter
    def state(self, value):
        with self._task_reserve_lock:
            if self._state.value != value:
                self._state.value = value
                # reset the progress
                progress_arr = np.ctypeslib.as_array(self._task_progress_array)
                progress_arr[:] = False

    @property
    def bg_off(self):
        """Brightness offset correction for the current chunk"""
        if self.mp_bg_off is not None:
            return np.ctypeslib.as_array(self.mp_bg_off)
        else:
            return None

    @property
    def image(self):
        """Return numpy view on image data"""
        # Convert the RawArray to something we can write to fast
        # (similar to memory view, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        return np.ctypeslib.as_array(self.mp_image).reshape(self.shape)

    @property
    def image_bg(self):
        """Return numpy view on background image data"""
        if self.mp_image_bg is not None:
            return np.ctypeslib.as_array(self.mp_image_bg).reshape(self.shape)
        else:
            return None

    @property
    def image_corr(self):
        """Return numpy view on background-corrected image data"""
        if self.mp_image_corr is not None:
            return np.ctypeslib.as_array(
                self.mp_image_corr).reshape(self.shape)
        else:
            return None

    @property
    def labels(self):
        return np.ctypeslib.as_array(
            self.mp_labels).reshape(self.shape)

    def acquire_task_lock(self,
                          req_state: str,
                          batch_size: int = None
                          ) -> tuple[int, int]:
        """Acquire the lock for performing a task

        Return the start and stop indices for which the lock was acquired.
        If no lock could be acquired, return `(0, 0)`. The returned
        indices might not match the input batch size: Locks for
        contiguous indices are returned based on availability.
        """
        # array for reserving a new batch
        reserve_array = np.ctypeslib.as_array(self._task_reserve_array)
        # array that monitors the progress of the current state
        progress_array = np.ctypeslib.as_array(self._task_progress_array)
        with self._task_reserve_lock:
            # combined array with frames are completed or are in use
            used_array = np.logical_or(reserve_array, progress_array)
            if self._state.value != req_state:
                # wrong state
                start = stop = 0
            else:
                # determine how many frames are locked
                num_locked = np.sum(used_array)
                if num_locked == self.length:
                    # all frames are locked
                    start = stop = 0
                else:
                    # We have at least one zero value
                    first_zero = np.where(~used_array)[0][0]
                    # Do we have a non-zero value after that?
                    pot_end = np.where(used_array[first_zero:])[0]
                    if pot_end.size:
                        # limit the array lock up until this value
                        max_size = first_zero + pot_end[0]
                    else:
                        # we may use the entire chunk
                        max_size = self.length

                    start = first_zero

                    if batch_size is None:
                        # reserve the rest of the chunk or this batch
                        stop = max_size
                    else:
                        # stop at the next non-zero element or the batch size
                        stop = min(start + batch_size, max_size)
                    reserve_array[start:stop] = True
        return start, stop

    def release_task_lock(self, start, stop, task_done=True):
        """Release the task lock for a batch range

        Releasing the task lock is done after completing the
        task for which a lock was required. This method also updates the
        progress (see `get_progress`) for the current task. Only release
        the task lock if you acquired it before.
        """
        if task_done:
            progress_array = np.ctypeslib.as_array(self._task_progress_array)
            progress_array[start:stop] = True

        reserve_array = np.ctypeslib.as_array(self._task_reserve_array)
        with self._task_reserve_lock:
            reserve_array[start:stop] = False

    def get_progress(self):
        """Return the progress of the current task

        Return a value between 0 and 1. If processing is done in batches
        (`batch_size` set in `acquire_task_lock`), this returns the
        fraction of frames for which `release_task_lock` was called
        with `task_done=True`.
        """
        progress_array = np.ctypeslib.as_array(self._task_progress_array)
        return float(np.sum(progress_array)) / self.length
