import atexit
import shutil
from collections import deque
import multiprocessing as mp
import pathlib
import tempfile

import h5py
import numpy as np

from dcnum import write


mp_spawn = mp.get_context('spawn')


def setup():
    global event_queue
    global writer_dq
    global feat_nevents
    batch_size = 500
    num_batches = 6
    num_events = batch_size * num_batches
    event_queue = mp_spawn.Queue()
    writer_dq = deque()
    feat_nevents = mp_spawn.Array("i", num_events)
    # every frame contains one event
    feat_nevents[:] = [1] * num_events

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
            event_queue.put((ii*batch_size + idx, event))


def main():
    tmp_dir = tempfile.mkdtemp(prefix=pathlib.Path(__file__).name)
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    path_out = pathlib.Path(tmp_dir) / "benchmark.rtdc"
    thr_coll = write.QueueWriterThread(
        event_queue=event_queue,
        write_queue_size=mp_spawn.Value("L", 0),
        feat_nevents=feat_nevents,
        path_out=path_out,
        write_threshold=500,
    )
    thr_coll.run()
    with h5py.File(path_out) as h5:
        assert h5["events/mask"].shape == (3000, 80, 320)
        assert h5["events/temp"].shape == (3000,)
