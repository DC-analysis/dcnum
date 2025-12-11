"""Feature computation: managing event extraction threads"""
import logging
import multiprocessing as mp
import threading
import time
from typing import Dict

import numpy as np

from .queue_event_extractor import EventExtractorThread, EventExtractorProcess


class EventExtractorManagerThread(threading.Thread):
    def __init__(self,
                 slot_register: "SlotRegister",  # noqa: F821
                 fe_kwargs: Dict,
                 num_workers: int,
                 write_queue_size: mp.Value,
                 debug: bool = False,
                 *args, **kwargs):
        """Manage event extraction threads or precesses

        Parameters
        ----------
        slot_register:
            Manages a list of `ChunkSlots`, shared arrays on which
            to operate
        fe_kwargs:
            Feature extraction keyword arguments. See
            :func:`.EventExtractor.get_init_kwargs` for more information.
        num_workers:
            Number of child threads or worker processes to use.
        write_queue_size:
            Multiprocessing value holding the number of event chunks
            waiting to be written to the output file; used for preventing
            OOM events by stalling data processing when the writer is slow
        debug:
            Whether to run in debugging mode which means only one
            event extraction thread (``num_workers`` has no effect).
        """
        super(EventExtractorManagerThread, self).__init__(
              name="EventExtractorManager", *args, **kwargs)
        self.logger = logging.getLogger(
            "dcnum.feat.EventExtractorManagerThread")

        self.fe_kwargs = fe_kwargs
        """Keyword arguments
        for :class:`event_extractor_manager_thread.py.QueueEventExtractor`
        instances"""

        self.slot_register = slot_register
        """Slot manager"""

        self.num_workers = 1 if debug else num_workers
        """Number of workers"""

        self.raw_queue = self.fe_kwargs["raw_queue"]
        """Queue for sending chunks and label indices to the workers"""

        self.write_queue_size = write_queue_size
        """Number of event chunks waiting to be written to the output file"""

        self.t_extract = 0
        """Feature extraction time counter"""

        self.t_wait = 0
        """Wait time counter"""

        self.debug = debug
        """Whether debugging is enabled"""

    def run(self):
        # Initialize all workers
        ta = time.perf_counter()
        if self.debug:
            worker_cls = EventExtractorThread
        else:
            worker_cls = EventExtractorProcess
        workers = [worker_cls(*list(self.fe_kwargs.values()), worker_index=ii)
                   for ii in range(self.num_workers)]
        [w.start() for w in workers]
        worker_monitor = self.fe_kwargs["worker_monitor"]

        self.logger.info(
            f"Initialization time: {time.perf_counter() - ta:.1f}s")

        chunks_processed = 0
        frames_processed = 0
        while True:
            t0 = time.perf_counter()

            # If the writer_dq starts filling up, then this could lead to
            # an oom-kill signal. Stall for the writer to prevent this.
            if (ldq := self.write_queue_size.value) > 1000:
                stalled_sec = 0.
                for ii in range(60):
                    if self.write_queue_size.value > 200:
                        time.sleep(.5)
                        stalled_sec += .5
                self.logger.warning(
                    f"Stalled {stalled_sec:.1f}s due to slow writer "
                    f"({ldq} chunks queued)")

            # Check all slots for segmented labels
            while True:
                state_warden = self.slot_register.reserve_slot_for_task(
                    current_state="e",
                    next_state="i")
                if state_warden is None:
                    time.sleep(.01)
                else:
                    break

            # We have a chunk, process it!
            t1 = time.perf_counter()
            self.t_wait += t1 - t0

            with state_warden as cs:
                # Let the workers know there is work
                [self.raw_queue.put((cs.chunk, ii)) for ii in range(cs.length)]

                # Make sure the entire chunk has been processed.
                while np.sum(worker_monitor) != frames_processed + cs.length:
                    time.sleep(.01)

            self.logger.debug(f"Extracted chunk {cs.chunk} in slot {cs}")

            self.t_extract += time.perf_counter() - t1

            chunks_processed += 1
            frames_processed += cs.length

            if chunks_processed == self.slot_register.num_chunks:
                break

        inv_masks = self.fe_kwargs["invalid_mask_counter"].value
        if inv_masks:
            self.logger.info(f"Encountered {inv_masks} invalid masks")
            inv_frac = inv_masks / self.slot_register.num_frames
            if inv_frac > 0.005:  # warn above one half percent
                self.logger.warning(f"Discarded {inv_frac:.1%} of the masks, "
                                    f"please check segmenter applicability")

        self.logger.debug("Requesting extraction workers to join")
        self.fe_kwargs["finalize_extraction"].value = True
        [w.join() for w in workers]

        self.logger.debug("Finished extraction")
        self.logger.info(f"Extraction time: {self.t_extract:.1f}s")
        self.logger.info(f"Waiting time: {self.t_wait:.1f}s")
