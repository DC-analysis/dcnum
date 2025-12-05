import multiprocessing as mp
import traceback

import numpy as np

from ..read import HDF5Data

from .chunk_slot import ChunkSlot
from .job import DCNumPipelineJob


mp_spawn = mp.get_context("spawn")


class SlotRegister:
    def __init__(self,
                 job: DCNumPipelineJob,
                 data: HDF5Data,
                 num_slots: int = 3):
        """A register for `ChunkSlot`s for shared memory access

        The `SlotRegister` manages all `ChunkSlot` instances and
        implements methods to interact with individual `ChunkSlot`s.
        """
        self.data = data
        self.chunk_size = data.image.chunk_size
        self.num_chunks = data.image.num_chunks
        self._slots = []

        self._chunks_loaded = mp_spawn.Value("Q", 0)

        self._state = mp_spawn.Value("u", "w")

        self.num_frames = len(self.data)
        """Total number of frames to process"""

        # generate all slots
        for ii in range(num_slots):
            self._slots.append(ChunkSlot(job=job, data=data))
        # we might need a slot for the remainder
        chunk_slot_remainder = ChunkSlot(job=job, data=data, is_remainder=True)
        if chunk_slot_remainder.length != 0:
            self._slots.append(chunk_slot_remainder)

    def __getitem__(self, idx):
        return self.slots[idx]

    def __iter__(self):
        """Iterate over slots, sorted by current chunk number"""
        slots_indices = np.argsort([sc.chunk for sc in self.slots])
        for idx in slots_indices:
            yield self.slots[idx]

    def __len__(self):
        return len(self.slots)

    @property
    def chunks_loaded(self):
        """A multiprocessing value counting the number of chunks loaded

        This number increments as `ChunkSlot.task_load_all` is called.
        """
        return self._chunks_loaded.value

    @chunks_loaded.setter
    def chunks_loaded(self, value):
        self._chunks_loaded.value = value

    @property
    def slots(self):
        """A list of all `ChunkSlots`"""
        return [s for s in self._slots]

    @property
    def state(self):
        """State of the `SlotRegister`, used for communication with workers

         - "w": initialized (workers work)
         - "p": paused (all workers pause)
         - "q": quit (all workers stop)
         """
        return self._state.value

    @state.setter
    def state(self, value):
        self._state.value = value

    def close(self):
        # Let everyone know we are closing
        self._state.value = "q"

    def find_slot(self, state: str, chunk: int = None) -> ChunkSlot | None:
        """Return the first `ChunkSlot` that has the given state

        We sort the slots according to the slot chunks so that we
        always process the slot with the smallest slot chunk number
        first. Initially, the slot_chunks array is filled with
        zeros, but we populate it here.

        Return None if no matching slot exists
        """
        for sc in self:
            if sc.state == state:
                if chunk is None:
                    return sc
                elif sc.chunk == chunk:
                    return sc

        # fallback to nothing found
        return None

    def get_lock(self, name):
        if name == "chunks_loaded":
            return self._chunks_loaded.get_lock()
        else:
            raise KeyError(f"No lock defined for {name}")

    def get_time(self, method_name):
        """Return accumulative time for the given method

        The times are extracted from each slot's `timers` values.
        """
        time_count = 0.0
        for sc in self._slots:
            time_count += sc.timers[method_name].value
        return time_count

    def task_load_all(self) -> bool:
        """Load chunk data into memory for as many slots as possible

        Returns
        -------
        did_something : bool
            Whether data were loaded into memory
        """
        did_something = False
        lock = self.get_lock("chunks_loaded")
        has_lock = lock.acquire(block=False)
        if has_lock and self.chunks_loaded < self.num_chunks:
            try:
                for sc in self:
                    # The number of sr.chunks_loaded is identical to the
                    # chunk index we want to load next.
                    if sc.state == "i" and sc.chunk <= self.chunks_loaded:
                        if ((self.chunks_loaded < self.num_chunks - 1
                             and not sc.is_remainder)
                                or (self.chunks_loaded == self.num_chunks - 1
                                    and sc.is_remainder)):
                            sc.load(self.chunks_loaded)
                            self.chunks_loaded += 1
                            did_something = True
            except BaseException:
                print(traceback.format_exc())
            finally:
                lock.release()
        return did_something
