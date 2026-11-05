"""Thread-based queue writer"""
import threading

from .queue_writer_base import QueueWriterBase


class QueueWriterThread(QueueWriterBase, threading.Thread):
    def __init__(self, *args, **kwargs):
        """Write events from a queue to an .rtdc file in a thread

        This is :class:`~dcnum.write.queue_writer_base.QueueWriterBase`
        running in a :class:`threading.Thread` (used in debugging mode).
        All arguments are passed to
        :class:`~dcnum.write.queue_writer_base.QueueWriterBase`.
        """
        super().__init__(*args, name="QueueWriterThread", **kwargs)
