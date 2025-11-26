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

            self.mp_image_corr = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.int16), array_length)

            self.mp_image_bg = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.uint8), array_length)

            # TODO: implement `segment` method.
            # Mask data
            self.mp_mask = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.bool), array_length)

            # Label data
            self.mp_labels = mp_spawn.RawArray(
                np.ctypeslib.as_ctypes_type(np.int16), array_length)

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

    @with_state_change(before="i", after="s")
    def load(self, idx):
        """Load chunk `idx` into `self.mp_image` and return a numpy view"""
        # create view on images array
        # Convert the RawArray to something we can write to fast
        # (similar to memory view, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        image = np.ctypeslib.as_array(self.mp_image).reshape(self.shape)
        image[:] = self.data.image.get_chunk(idx)

        if self.data.image_bg:
            image_bg = np.ctypeslib.as_array(
                self.mp_image_bg).reshape(self.shape)
            image_bg[:] = self.data.image_bg.get_chunk(idx)

            image_corr = np.ctypeslib.as_array(
                self.mp_image_corr).reshape(self.shape)
            image_corr[:] = np.array(image, dtype=np.int16) - image_bg
        else:
            image_bg = None
            image_corr = None

        self.chunk = idx
        return image, image_bg, image_corr
