import functools
import multiprocessing as mp
import time

import numpy as np

from ..segm import get_available_segmenters

from .chunk_slot_data import ChunkSlotData


mp_spawn = mp.get_context("spawn")


class count_time:
    """Decorator for counting execution time"""

    def __call__(self, func):
        @functools.wraps(func)
        def method(inst, *args, **kwargs):
            t0 = time.perf_counter()
            retval = func(inst, *args, **kwargs)
            # update the time counter for this method
            fn = func.__name__
            if fn in inst.timers:
                with inst.timers[fn].get_lock():
                    inst.timers[fn].value += time.perf_counter() - t0
            return retval

        return method


class ChunkSlot(ChunkSlotData):
    _instance_counter = 0

    def __init__(self, job, data, is_remainder=False):
        ChunkSlot._instance_counter += 1
        self.index = ChunkSlot._instance_counter

        self.timers = {
            "load": mp_spawn.Value("d", 0.0)
        }

        self.job = job
        """Job information object"""

        self.data = data
        """Input data object"""

        self.is_remainder = is_remainder
        """Whether this slot only applies to the last chunk"""

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

    def __repr__(self):
        return (f"<dcnum ChunkSlot {self.index} (state {self.state}) "
                f"at {hex(id(self))}>")

    @count_time()
    def load(self, idx):
        """Load chunk `idx` into `self.mp_image` and return numpy views"""
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
