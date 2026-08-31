import threading

from .queue_writer_base import QueueWriterBase


class QueueWriterThread(QueueWriterBase, threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="QueueWriterThread", **kwargs)
