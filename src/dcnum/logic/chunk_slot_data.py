import multiprocessing as mp

import numpy as np


mp_spawn = mp.get_context("spawn")


class ChunkSlotData:
    def __init__(self, shape, available_features=None):
        self.shape = shape
        """3D shape of the chunk in this slot"""

        self.task_lock = mp_spawn.Lock()
        """Lock for synchronizing chunk data modification"""

        available_features = available_features or []
        self.length = shape[0]

        self._state = mp_spawn.Value("u", "0", lock=False)

        # Initialize with negative value to avoid ambiguities with first chunk.
        self._chunk = mp_spawn.Value("q", -1, lock=False)

        # Initialize all shared arrays
        if self.length:
            array_length = int(np.prod(self.shape))

            # Image data
            self.mp_image = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.uint8), array_length)

            if "image_bg" in available_features:
                self.mp_image_bg = mp_spawn.RawArray(
                    np.ctypeslib.as_ctypes_type(np.uint8), array_length)

                self.mp_image_corr = mp_spawn.RawArray(
                    np.ctypeslib.as_ctypes_type(np.int16), array_length)
            else:
                self.mp_image_bg = None
                self.mp_image_corr = None

            if "bg_off" in available_features:
                # background offset data
                self.mp_bg_off = mp_spawn.RawArray(
                    np.ctypeslib.as_ctypes_type(np.float64), self.length)
            else:
                self.mp_bg_off = None

            # Mask data
            self.mp_mask = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.bool), array_length)

            # Label data
            self.mp_labels = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.uint16), array_length)

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
        self._state.value = value

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

    def acquire_task_lock(self) -> bool:
        """Acquire the lock for performing a task

        Return True if the lock is acquired, False if the
        lock has been acquired beforehand.
        """
        return self.task_lock.acquire(block=False)

    def release_task_lock(self) -> bool:
        """Release the task lock

        Releasing the task lock is done after completing the
        task for which a lock was required. Only release the
        task lock if you acquired it before.

        Return True if the lock has been released, False
        if the lock wasn't acquired beforehand (and thus not released).
        """
        try:
            self.task_lock.release()
        except ValueError:
            released = False
        else:
            released = True
        return released
