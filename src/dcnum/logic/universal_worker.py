import logging
import traceback
from logging.handlers import QueueHandler
import multiprocessing as mp
import os
import threading
import time

from ..os_env_st import confirm_single_threaded


mp_spawn = mp.get_context("spawn")


class UniversalWorker:
    def __init__(self,
                 slot_register: "SlotRegister",  # noqa: F821
                 log_queue: "mp.Queue",
                 log_level: int = logging.INFO,
                 *args, **kwargs):
        # Must call super init, otherwise Thread or Process are not initialized
        super(UniversalWorker, self).__init__(*args, **kwargs)

        self.slot_register = slot_register
        """Chunk slot register"""

        self.log_queue = log_queue
        """queue for logging"""

        # Logging needs to be set up after `start` is called, otherwise
        # it looks like we have the same PID as the parent process. We
        # are setting up logging in `run`.
        self.log_level = log_level or logging.getLogger("dcnum").level

    def run(self):
        confirm_single_threaded()

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

        # only close queues when we have created them ourselves.
        close_queues = isinstance(self, mp_spawn.Process)
        wait_time_writer = 0

        sr = self.slot_register
        while sr.state != "q":
            did_something = False

            if sr.state == "p":
                time.sleep(0.5)
                continue

            try:
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

                # Load data into memory for all available slots
                did_something |= sr.task_load_all(logger=logger)

                # Perform feature extraction
                did_something |= sr.task_extract_features(logger=logger)
            except BaseException:
                logger.error(traceback.format_exc())

            if not did_something:
                time.sleep(.01)

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
        super(UniversalWorkerThread, self).__init__(
            name="UniversalWorkerThread", *args, **kwargs)


class UniversalWorkerProcess(UniversalWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super(UniversalWorkerProcess, self).__init__(
            name="UniversalWorkerProcess", *args, **kwargs)
