from collections import deque
import pathlib
import shutil
import tempfile

import multiprocessing as mp

import h5py
import numpy as np

from dcnum import write


mp_spawn = mp.get_context('spawn')


class Benchmark:
    def __init__(self):
        total_frames = 3000
        batch_size = 500
        num_batches = 6
        assert batch_size * num_batches == total_frames

        self.writer_dq = deque()
        # Create 1000 events with at most two repetitions in a frame
        np.random.seed(42)
        rng = np.random.default_rng()

        # create a sample event
        for ii in range(num_batches):
            self.writer_dq.append(("mask",
                                   rng.random((batch_size, 80, 320)) > .5))
            self.writer_dq.append(("temp",
                                   rng.normal(23, size=batch_size)))

        self.tmp_path = tempfile.mkdtemp(prefix=pathlib.Path(__file__).name)
        self.path_out = pathlib.Path(self.tmp_path) / "out.rtdc"

    def benchmark(self):
        thr_drw = write.ChunkWriter(
            path_out=self.path_out,
            dq=self.writer_dq,
            write_queue_size=mp_spawn.Value("L", 0)
        )
        thr_drw.may_stop_loop = True
        thr_drw.run()

    def verify(self):
        with h5py.File(self.path_out) as h5:
            assert h5["events/mask"].shape == (3000, 80, 320)
            assert h5["events/temp"].shape == (3000,)

    def teardown(self):
        shutil.rmtree(self.tmp_path, ignore_errors=True)
