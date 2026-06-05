import threading

from .queue_writer_base import QueueWriterBase


class QueueWriterThread(QueueWriterBase, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(QueueWriterThread, self).__init__(
              name="QueueWriterThread", *args, **kwargs)
