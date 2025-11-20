import shutil
from collections import deque
import multiprocessing as mp
import pathlib
import tempfile

import h5py
import numpy as np

from dcnum import write


mp_spawn = mp.get_context('spawn')


class Benchmark:
    def __init__(self):
        self.tmp_path = tempfile.mkdtemp(prefix=pathlib.Path(__file__).name)
        self.path_out = pathlib.Path(self.tmp_path) / "benchmark.rtdc"
        self.event_queue = mp_spawn.Queue()
        self.writer_dq = deque()

        batch_size = 500
        num_batches = 6
        num_events = batch_size * num_batches

        self.feat_nevents = mp_spawn.Array("i", num_events)
        # every frame contains one event
        self.feat_nevents[:] = [1] * num_events

        # Create 1000 events with at most two repetitions in a frame
        np.random.seed(42)
        rng = np.random.default_rng()
        number_order = rng.choice(batch_size, size=batch_size, replace=False)

        # create a sample event
        for ii in range(num_batches):
            for idx in number_order:
                event = {
                    "temp": np.atleast_1d(rng.normal(23)),
                    "mask": rng.random((1, 80, 320)) > .5,
                }
                self.event_queue.put((ii*batch_size + idx, event))

    def benchmark(self):
        thr_coll = write.QueueWriterThread(
            event_queue=self.event_queue,
            write_queue_size=mp_spawn.Value("L", 0),
            feat_nevents=self.feat_nevents,
            path_out=self.path_out,
            write_threshold=500,
        )
        thr_coll.run()

    def verify(self):
        with h5py.File(self.path_out) as h5:
            assert h5["events/mask"].shape == (3000, 80, 320)
            assert h5["events/temp"].shape == (3000,)

    def teardown(self):
        shutil.rmtree(self.tmp_path, ignore_errors=True)
