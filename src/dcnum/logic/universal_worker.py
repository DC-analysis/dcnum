import multiprocessing as mp
import threading
import time
import traceback

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

            # Load data into memory for all slots
            lock = sr.get_lock("chunks_loaded")
            has_lock = lock.acquire(block=False)
            if has_lock and sr.chunks_loaded < sr.num_chunks:
                try:
                    for sc in sr:
                        # The number of sr.chunks_loaded is identical to the
                        # chunk index we want to load next.
                        if sc.state == "i" and sc.chunk <= sr.chunks_loaded:
                            if ((sr.chunks_loaded < sr.num_chunks - 1
                                 and not sc.is_remainder)
                                or (sr.chunks_loaded == sr.num_chunks - 1
                                    and sc.is_remainder)):
                                sc.load(sr.chunks_loaded)
                                sr.chunks_loaded += 1
                                did_something = True
                except BaseException:
                    print(traceback.format_exc())
                finally:
                    lock.release()

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
