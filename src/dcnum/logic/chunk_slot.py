import functools
import multiprocessing as mp
import time

import numpy as np

from ..segm import get_available_segmenters

mp_spawn = mp.get_context("spawn")


class with_state_change:
    def __init__(self, before, after):
        """Decorator for enforcing a state change in ChunkSlot class"""
        self.before = before
        self.after = after

    def __call__(self, func):
        @functools.wraps(func)
        def method(inst, *args, **kwargs):
            if inst.state != self.before:
                raise ValueError(
                    f"Incorrect state: {inst.state=}, {inst.before=}")
            t0 = time.perf_counter()
            data = func(inst, *args, **kwargs)
            inst.state = self.after
            # update the time counter for this method
            fn = func.__name__
            if fn in inst.timers:
                with inst.timers[fn].get_lock():
                    inst.timers[fn].value += time.perf_counter() - t0
            return data

        return method


class ChunkSlotBase:
    def __init__(self, shape, available_features=None):
        self.shape = shape
        available_features = available_features or []
        self.length = shape[0]

        self._state = mp_spawn.Value("u", "0", lock=False)

        self._chunk = mp_spawn.Value("i", 0, lock=False)

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


class ChunkSlot(ChunkSlotBase):
    _instance_counter = 0

    def __init__(self, job, data, is_remainder=False):
        self._instance_counter += 1
        self.index = self._instance_counter

        self.timers = {
            "load": mp_spawn.Value("d", 0.0)
        }

        self.job = job
        """Job information object"""

        self.data = data
        """Input data object"""

        self.is_remainder = is_remainder
        """Whether this slot only applies to the last chunk"""

        # Determine the dtype of the input data
        self.seg_cls = get_available_segmenters()[self.job["segmenter_code"]]
        """Segmentation class"""

        if self.is_remainder:
            try:
                length = self.data.image.get_chunk_size(
                    chunk_index=self.data.image.num_chunks - 1)
            except IndexError:
                length = 0
        else:
            length = self.data.image.chunk_size

        super(ChunkSlot, self).__init__(
            shape=(length,) + self.data.image.shape[1:],
            available_features=self.data.keys(),
        )

    def __str__(self):
        return f"SC-{self.index}"

    @with_state_change(before="i", after="s")
    def load(self, idx):
        """Load chunk `idx` into `self.mp_image` and return a numpy view"""
        # create views on image arrays
        image = self.image
        image[:] = self.data.image.get_chunk(idx)

        if self.mp_image_bg is not None:
            image_bg = self.image_bg
            image_bg[:] = self.data.image_bg.get_chunk(idx)
            image_corr = self.image_corr
            image_corr[:] = np.asarray(image, dtype=np.int16) - image_bg
        else:
            image_bg = None
            image_corr = None

        if self.mp_bg_off is not None:
            bg_off = self.bg_off
            chunk_slice = self.data.image.get_chunk_slice(idx)
            bg_off[:] = self.data["bg_off"][chunk_slice]
        else:
            bg_off = None

        # TODO: Check for duplicate, consecutive images while loading data
        #  and store that information in a boolean array. This can speed-up
        #  segmentation and feature extraction.

        self.chunk = idx
        return image, image_bg, image_corr, bg_off
