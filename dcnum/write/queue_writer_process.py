"""Process-based queue writer"""
import traceback
from logging.handlers import QueueHandler
import multiprocessing as mp

from .queue_writer_base import QueueWriterBase

mp_spawn = mp.get_context('spawn')


class QueueWriterProcess(QueueWriterBase, mp_spawn.Process):
    def __init__(self,
                 log_queue: mp.Queue,
                 *args, **kwargs):
        """Write events from a queue to an .rtdc file in a subprocess

        This is :class:`~dcnum.write.queue_writer_base.QueueWriterBase`
        running in a spawned :class:`multiprocessing.Process`.

        Parameters
        ----------
        log_queue: multiprocessing.Queue
            queue to which log records are sent, since logging handlers
            cannot be shared with the parent process
        *args:
            positional arguments passed to
            :class:`~dcnum.write.queue_writer_base.QueueWriterBase`
        **kwargs:
            keyword arguments passed to
            :class:`~dcnum.write.queue_writer_base.QueueWriterBase`
        """
        # Since we are running in a process, we cannot initialize the logger
        # during init. We must initialize the logger in `run`.
        self.log_queue = log_queue
        """queue for logging"""

        super().__init__(*args, name="QueueWriterProcess", **kwargs)

    def run(self, **kwargs):
        """Collect and write events (executed in the subprocess)

        Sets up a logger that forwards all records to ``log_queue`` and
        then calls
        :meth:`~dcnum.write.queue_writer_base.QueueWriterBase.run`.
        Keyword arguments are ignored (a warning is logged for each one
        that is not None).
        """
        # Clear any handlers that might be set for this logger. This is
        # important for the case when we are an instance of
        # EventExtractorThread, because then all handlers from the main
        # thread are inherited (as opposed to no handlers in the case
        # of EventExtractorProcess).
        logger = self.get_logger()
        logger.handlers.clear()
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(self.log_level)
        logger.addHandler(queue_handler)
        logger.info("Ready")

        # some checks to avoid confusion
        for kw, val in kwargs.items():
            if val is not None:
                logger.warning(f"Specifying `{kw}` in "
                               f"`QueueWriterProcess.run` has no effect.")

        try:
            super().run(logger=logger)
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            self.log_queue.cancel_join_thread()
            self.event_queue.cancel_join_thread()
        except BaseException:
            logger.error(traceback.format_exc())

        # Make sure everything gets written to the queue.
        queue_handler.flush()
        queue_handler.close()

        # Close the logging queue.
        self.log_queue.close()
        self.log_queue.join_thread()
