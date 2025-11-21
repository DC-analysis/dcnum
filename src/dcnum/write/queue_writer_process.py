from logging.handlers import QueueHandler
import multiprocessing as mp

from .queue_writer_base import QueueWriterBase

mp_spawn = mp.get_context('spawn')


class QueueWriterProcess(QueueWriterBase, mp_spawn.Process):
    def __init__(self,
                 log_queue: mp.Queue,
                 *args, **kwargs):
        # Since we are running in a process, we cannot initialize the logger
        # during init. We must initialize the logger in `run`.
        self.log_queue = log_queue
        """queue for logging"""

        super(QueueWriterProcess, self).__init__(
              name="QueueWriterProcess", *args, **kwargs)

    def run(self, **kwargs):
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

        super(QueueWriterProcess, self).run(logger=logger)

        # Make sure everything gets written to the queue.
        queue_handler.flush()

        # Close the logging queue.
        self.log_queue.close()
        self.log_queue.join_thread()
