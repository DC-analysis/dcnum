from __future__ import annotations

import copy
import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
import os
import threading
import time
import traceback
import typing

from ..common import DCNUMHaltInterrupt
from ..os_env_st import RequestSingleThreaded, confirm_single_threaded
from ..segm import UNISegmenter, get_segmenters


if typing.TYPE_CHECKING:
    from ..logic import DCNumPipelineJob
    from .slot_register import SlotRegister

mp_spawn = mp.get_context("spawn")

default_dedications = [
    "load_all",
    "segment_images",
    "label_masks",
    "process_labels",
    "extract_features",
]


class UniversalWorker:
    def __init__(self,
                 slot_register: SlotRegister,
                 log_queue: mp.Queue,
                 dedications: list[str] | None = None,
                 log_level: int = logging.INFO,
                 *args, **kwargs):
        # Must call super init, otherwise Thread or Process is not initialized
        super().__init__(*args, **kwargs)

        if dedications is None:
            dedications = copy.copy(default_dedications)
        self.dedications = dedications

        self.slot_register = slot_register
        """Chunk slot register"""

        self.log_queue = log_queue
        """queue for logging"""

        # Logging needs to be set up after `start` is called, otherwise
        # it looks like we have the same PID as the parent process. We
        # are setting up logging in `run`.
        self.log_level = log_level or logging.getLogger("dcnum").level

    @staticmethod
    def get_worker_dedications(job: DCNumPipelineJob,
                               num_universal: int,
                               ) -> list[list[str]]:
        """Return a worker dedications for each UniversalWorker"""
        # Start with: all workers do everything.
        dcs = [copy.copy(default_dedications) for _ in range(num_universal)]

        # If the segmenter is not the UNISegmenter, then workers
        # should not segment at all.
        seg_cls = get_segmenters()[job["segmenter_code"]]
        if not issubclass(seg_cls, UNISegmenter):
            for ii in range(num_universal):
                dcs[ii].remove("segment_images")

        # Loading image data only needs to be done by one worker.
        if num_universal > 1:
            # Only the first worker should load image data.
            for ii in range(1, num_universal):
                dcs[ii].remove("load_all")

        if issubclass(seg_cls, UNISegmenter):
            # The UNISegmenter may modify the dedications.
            dcs = seg_cls.update_worker_dedications(job, dcs)

        return dcs

    def run(self):
        # If multiprocessing is used, we now live in our own process.
        confirm_single_threaded()

        # Connect the logger
        logger = logging.getLogger(
            f"dcnum.logic.UniversalWorker.{os.getpid()}")
        """logger that sends all logs to `self.log_queue`"""
        logger.setLevel(self.log_level)
        # Clear any handlers that might be set for this logger. This is
        # important for the case when we are an instance of
        # UniversalWorkerThread, because then all handlers from the main
        # thread are inherited (as opposed to no handlers in the case
        # of UniversalWorkerProcess).
        logger.handlers.clear()
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(self.log_level)
        logger.addHandler(queue_handler)
        logger.debug("Ready")

        # Only close queues when we have created them ourselves.
        close_queues = isinstance(self, mp_spawn.Process)
        wait_time_writer = 0

        # If we are responsible for segmentation, set everything up.
        try:
            self.slot_register.segmenter.log_info(logger)
        except ValueError:
            # Not a UNISegmenter
            pass

        sr = self.slot_register
        try:
            while sr.state != "q":
                did_something = False

                if sr.state == "p":
                    time.sleep(0.5)
                    continue

                # Check whether the writer is overloaded
                if (ldq := self.slot_register.write_queue_size) > 1000:
                    stalled_sec = 0.
                    for ii in range(60):
                        if self.slot_register.write_queue_size > 200:
                            time.sleep(.5)
                            stalled_sec += .5
                    wait_time_writer += stalled_sec
                    logger.debug(
                        f"Stalled {stalled_sec:.1f}s due to slow writer "
                        f"({ldq} chunks queued)")

                if "load_all" in self.dedications:
                    # Load data into memory for all available slots
                    did_something |= sr.task_load_all(logger=logger)

                if "segment_images" in self.dedications:
                    # Segmentation is only done for `UNISegmenter` subclasses
                    did_something |= sr.task_segment_images(logger=logger)

                if "label_masks" in self.dedications:
                    # After segmentation, perform mask to label conversion
                    did_something |= sr.task_label_masks(logger=logger)

                if "process_labels" in self.dedications:
                    # After labeling, perform label processing
                    did_something |= sr.task_process_labels(logger=logger)

                if "extract_features" in self.dedications:
                    # Finally, perform feature extraction
                    did_something |= sr.task_extract_features(logger=logger)

                if not did_something:
                    time.sleep(.01)

        except (KeyboardInterrupt, DCNUMHaltInterrupt):
            self.log_queue.cancel_join_thread()
            return
        except BaseException:
            logger.error(traceback.format_exc())

        if wait_time_writer > 10:
            logger.warning(f"Waited a total of {wait_time_writer:.1f}s "
                           f"due to slow writer")
        logger.debug("Finalizing")

        # Make sure everything gets written to the queue.
        queue_handler.flush()
        queue_handler.close()

        if close_queues:
            # Also close the logging queue. Note that not all messages might
            # arrive in the logging queue, since we called `cancel_join_thread`
            # earlier.
            self.log_queue.close()
            self.log_queue.join_thread()


class UniversalWorkerThread(UniversalWorker, threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="UniversalWorkerThread", **kwargs)


class UniversalWorkerProcess(UniversalWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="UniversalWorkerProcess", **kwargs)

    def start(self):
        # Set all relevant os environment variables such libraries in the
        # new process only use single-threaded computation.
        with RequestSingleThreaded():
            mp_spawn.Process.start(self)
