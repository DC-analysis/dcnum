import logging
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

        # Counters are created with recursive locks, which means that the
        # same process may acquire multiple locks on the object, and only
        # after releasing all of them, may the lock be acquired by another
        # process.
        self.counters = {
            "chunks_loaded": mp_spawn.Value("Q", 0),
            "masks_dropped": mp_spawn.Value("Q", 0),
            "write_queue_size": mp_spawn.Value("Q", 0),
        }

        self._state = mp_spawn.Value("u", "w")

        self.num_frames = len(self.data)
        """Total number of frames to process"""

        self.feat_nevents = mp_spawn.RawArray("l", self.num_frames)
        """Number of events per frame
        Shared RawArray of length `len(data)` into which the number of
        events per frame is written.
        """
        # Initialize feat_nevents with -1
        self.feat_nevents[:] = np.full(self.num_frames, -1)

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
        """A process-safe counter for the number of chunks loaded

        This number increments as `SlotRegister.task_load_all` is called.
        """
        return self.counters["chunks_loaded"].value

    @chunks_loaded.setter
    def chunks_loaded(self, value):
        self.counters["chunks_loaded"].value = value

    @property
    def masks_dropped(self):
        """A process-safe counter for the number of masks dropped

        Segmentation may drop invalid masks/events.
        """
        return self.counters["masks_dropped"].value

    @masks_dropped.setter
    def masks_dropped(self, value):
        self.counters["masks_dropped"].value = value

    @property
    def write_queue_size(self):
        """A process-safe counter for the number of chunks in the writer queue

        A large number indicates a slow writer which can be
        a result of a slow hard disk or a slow CPU (since
        is used compression). Used for preventing
        OOM events by stalling data processing when the writer is slow
        """
        return self.counters["write_queue_size"].value

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

    def get_counter_lock(self, name):
        if name in self.counters:
            return self.counters[name].get_lock()
        else:
            raise KeyError(f"No counter lock defined for {name}")

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
                              batch_size: int = None,
                              ) -> "StateWarden | None":
        """Return slot with the specified state and lowest chunk index

        Parameters
        ----------
        current_state:
            State requried for the task to start
        next_state:
            State that will be set after the task is done
        chunk_slot:
            Optional `ChunkSlot` to operate on; if set to None, search
            for a matching one, and if none can be found, return None
        batch_size:
            Number of frames to reserve for performing the task. Defaults
            to the entire chunk.

        Returns
        -------
        state_warden
            Context manager that enforces setting the next state or
            None if no `ChunkSlot` could be reserved.
            Usage:

                if state_warden is not None:
                    with state_warden as (chunk_slot, batch_range):
                        perform_task(chunk_slot,
                                     start_index=batch_range[0],
                                     stop_index=batch_range[1]
                                     )

            The `batch_range` indices are defined by the `batch_size`
            parameter.

            This context manager will automatically set the slot
            state to `next_state` when the context is exits
            without exceptions.
        """
        if chunk_slot is None:
            for sc in self:
                if sc.state == current_state:
                    sw = StateWarden(sc,
                                     current_state=current_state,
                                     next_state=next_state,
                                     batch_size=batch_size)
                    if sw.batch_size:
                        return sw
                    else:
                        # nothing could be reserved
                        return None

            # fallback to nothing found
            return None
        else:
            return StateWarden(chunk_slot,
                               current_state=current_state,
                               next_state=next_state,
                               batch_size=batch_size)

    def task_load_all(self, logger: logging.Logger = None) -> bool:
        """Load chunk data into memory for as many slots as possible

        Returns
        -------
        did_something : bool
            Whether data were loaded into memory
        """
        did_something = False
        lock = self.get_counter_lock("chunks_loaded")
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
                if logger is not None:
                    logger.error(traceback.format_exc())
            finally:
                lock.release()
        return did_something


class StateWarden:
    """Context manager for changing the state of a `ChunkSlot`"""
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
        self.batch_range = chunk_slot.acquire_task_lock(batch_size=batch_size)
        self.batch_size = self.batch_range[1] - self.batch_range[0]

        self.chunk_slot = chunk_slot
        self.current_state = current_state
        self.next_state = next_state

    def __enter__(self):
        # Make sure the state is still correct
        # release the lock, because somebody else might need it
        if self.chunk_slot.state != self.current_state:
            self.chunk_slot.release_task_lock(*self.batch_range,
                                              task_done=False)
            raise ValueError(
                f"Current state of slot {self.chunk_slot} "
                f"({self.chunk_slot.state}) does not match "
                f"expected state {self.current_state}.")
        return self.chunk_slot, self.batch_range

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.chunk_slot.release_task_lock(
            *self.batch_range,
            # only set batch to done if no exception occurred
            task_done=exc_type is None)
        if self.chunk_slot.get_progress() == 1:
            self.chunk_slot.state = self.next_state

    def __repr__(self):
        return (f"<StateWarden {self.current_state}->{self.next_state} "
                f"at {hex(id(self))}>")
