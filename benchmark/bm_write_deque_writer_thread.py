import atexit
from collections import deque
import pathlib
import shutil
import tempfile

import multiprocessing as mp

import numpy as np

from dcnum import write


mp_spawn = mp.get_context('spawn')


def setup():
    global path_out
    global writer_dq
    total_frames = 3000
    batch_size = 500
    num_batches = 6
    assert batch_size * num_batches == total_frames

    writer_dq = deque()
    # Create 1000 events with at most two repetitions in a frame
    np.random.seed(42)
    rng = np.random.default_rng()

    # create a sample event
    for ii in range(num_batches):
        writer_dq.append(("mask", rng.random((batch_size, 80, 320)) > .5))
        writer_dq.append(("temp", rng.normal(23, size=batch_size)))

    temp_dir = tempfile.mkdtemp(prefix=pathlib.Path(__file__).name)
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True, onerror=None)
    path_out = pathlib.Path(temp_dir) / "out.rtdc"


def main():
    thr_drw = write.DequeWriterThread(
        path_out=path_out,
        dq=writer_dq,
    )
    thr_drw.may_stop_loop = True
    thr_drw.run()
