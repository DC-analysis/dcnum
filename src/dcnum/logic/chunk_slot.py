import functools
import multiprocessing as mp

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
            assert inst.state == self.before
            data = func(inst, *args, **kwargs)
            inst.state = self.after
            return data

        return method


class ChunkSlot:
    _instance_counter = 0

    def __init__(self, job, data, is_remainder=False):
        self._instance_counter += 1
        self.index = self._instance_counter

        self.job = job
        """Job information object"""

        self.data = data
        """Input data object"""

        self._state = mp_spawn.Value("u", "0", lock=False)

        self._chunk = mp_spawn.Value("i", 0, lock=False)

        self.is_remainder = is_remainder
        """Whether this slot only applies to the last chunk"""

        # Determine the dtype of the input data
        self.seg_cls = get_available_segmenters()[self.job["segmenter_code"]]
        """Segmentation class"""

        if self.is_remainder:
            try:
                self.length = self.data.image.get_chunk_size(
                    chunk_index=self.data.image.num_chunks - 1)
            except IndexError:
                self.length = 0
        else:
            self.length = self.data.image.chunk_size

        self.shape = (self.length,) + self.data.image.shape[1:]

        # Initialize all shared arrays
        if self.length:
            array_length = int(np.prod(self.shape))

            # Image data
            self.mp_image = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.uint8), array_length)

            if "image_bg" in self.data:
                self.mp_image_bg = mp_spawn.RawArray(
                    np.ctypeslib.as_ctypes_type(np.uint8), array_length)

                self.mp_image_corr = mp_spawn.RawArray(
                    np.ctypeslib.as_ctypes_type(np.int16), array_length)
            else:
                self.mp_image_bg = None
                self.mp_image_corr = None

            if "bg_off" in self.data:
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
                np.ctypeslib.as_ctypes_type(np.int16), array_length)

        # TODO: Track timing information for e.g. the `load` method.

        self._state.value = "i"

    def __str__(self):
        return f"SC-{self.index}"

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
        - "i": image loading
        - "s": segmentation
        - "w": writing
        - "d": done
        - "n": not specified
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
            image_corr[:] = np.array(image, dtype=np.int16) - image_bg
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
