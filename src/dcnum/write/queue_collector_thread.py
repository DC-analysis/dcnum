import threading

from .queue_collector_base import QueueCollectorBase


class QueueCollectorThread(QueueCollectorBase, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(QueueCollectorThread, self).__init__(
              name="QueueCollectorThread", *args, **kwargs)
