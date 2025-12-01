import multiprocessing as mp

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
        """A register fo `ChunkSlot`s for shared memory access"""
        self.data = data
        self.chunk_size = data.image.chunk_size
        self.num_chunks = data.image.num_chunks
        self._slots = []

        self._chunks_loaded = mp_spawn.Value("L", 0)

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
        return self._chunks_loaded.value

    @chunks_loaded.setter
    def chunks_loaded(self, value):
        self._chunks_loaded.value = value

    @property
    def slots(self):
        return [s for s in self._slots]

    @property
    def state(self):
        """State of the slot manager, used for communication with workers

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
