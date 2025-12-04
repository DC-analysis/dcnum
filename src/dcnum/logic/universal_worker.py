import multiprocessing as mp
import threading
import time

mp_spawn = mp.get_context("spawn")


class UniversalWorker:
    def __init__(self, slot_register, *args, **kwargs):
        self.slot_register = slot_register
        # Must call super init, otherwise Thread or Process are not initialized
        super(UniversalWorker, self).__init__(*args, **kwargs)

    def run(self):
        sr = self.slot_register
        while sr.state != "q":
            did_something = False

            if sr.state == "p":
                time.sleep(0.5)
                continue

            # Load data into memory for all available slots
            did_something |= sr.task_load_all()

            if not did_something:
                time.sleep(.01)


class UniversalWorkerThread(UniversalWorker, threading.Thread):
    def __init__(self, *args, **kwargs):
        super(UniversalWorkerThread, self).__init__(
            name="UniversalWorkerThread", *args, **kwargs)


class UniversalWorkerProcess(UniversalWorker, mp_spawn.Process):
    def __init__(self, *args, **kwargs):
        super(UniversalWorkerProcess, self).__init__(
            name="UniversalWorkerProcess", *args, **kwargs)
