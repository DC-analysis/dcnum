import multiprocessing as mp
import traceback

import numpy as np

from ..read import HDF5Data

from .chunk_slot import ChunkSlot, ChunkSlotData
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

    def reserve_slot_for_task(self,
                              current_state: str,
                              next_state: str,
                              chunk_slot: ChunkSlot = None,
                              ) -> "StateWarden | None":
        """Return slot with the specified state and lowest chunk index

        Returns
        -------
        state_warden
            Context manager that enforces setting the next state or
            None if no `ChunkSlot` could be reserved.
            Usage:

                if state_warden is not None:
                    with state_warden as chunk_slot:
                        perform_task(chunk_slot)

            This context manager will automatically set the slot
            state to `next_state` when the context is exits
            without exceptions.
        """
        if chunk_slot is None:
            for sc in self:
                if sc.state == current_state and sc.acquire_task_lock():
                    return StateWarden(sc, current_state, next_state)

            # fallback to nothing found
            return None
        else:
            return StateWarden(chunk_slot, current_state, next_state)

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
                for cs in self:
                    # The number of sr.chunks_loaded is identical to the
                    # chunk index we want to load next.
                    if cs.state == "i" and cs.chunk <= self.chunks_loaded:
                        if ((self.chunks_loaded < self.num_chunks - 1
                             and not cs.is_remainder)
                                or (self.chunks_loaded == self.num_chunks - 1
                                    and cs.is_remainder)):
                            with self.reserve_slot_for_task(current_state="i",
                                                            next_state="s",
                                                            chunk_slot=cs):
                                cs.load(self.chunks_loaded)
                            self.chunks_loaded += 1
                            did_something = True
            except BaseException:
                print(traceback.format_exc())
            finally:
                lock.release()
        return did_something


class StateWarden:
    """Context manager for changing the state a `SlotChunk`"""
    def __init__(self,
                 chunk_slot: ChunkSlot | ChunkSlotData,
                 current_state: str,
                 next_state: str,
                 batch_size: int = None,
                 ):
        # Make sure the state is correct
        if chunk_slot.state != current_state:
            raise ValueError(
                f"Current state of slot {chunk_slot} ({chunk_slot.state}) "
                f"does not match expected state {current_state}.")
        # Make sure the task lock is acquired.
        chunk_slot.acquire_task_lock()

        self.chunk_slot = chunk_slot
        self.current_state = current_state
        self.next_state = next_state

    def __enter__(self):
        # Make sure the state is still correct
        # release the lock, because somebody else might need it
        if self.chunk_slot.state != self.current_state:
            self.chunk_slot.release_task_lock()
            raise ValueError(
                f"Current state of slot {self.chunk_slot} "
                f"({self.chunk_slot.state}) does not match "
                f"expected state {self.current_state}.")
        return self.chunk_slot

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.chunk_slot.state = self.next_state
        self.chunk_slot.release_task_lock()

    def __repr__(self):
        return (f"<StateWarden {self.current_state}->{self.next_state} "
                f"at {hex(id(self))}>")
