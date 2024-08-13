from collections import deque
import multiprocessing as mp

import numpy as np

from dcnum import write


mp_spawn = mp.get_context('spawn')


def setup():
    global event_queue
    global writer_dq
    global feat_nevents
    batch_size = 1000
    num_batches = 3
    num_events = batch_size * num_batches
    event_queue = mp.Queue()
    writer_dq = deque()
    feat_nevents = mp_spawn.Array("i", num_events)

    # Create 1000 events with at most two repetitions in a frame
    np.random.seed(42)
    rng = np.random.default_rng()
    number_order = rng.choice(batch_size, size=batch_size, replace=False)

    # create a sample event
    event = {
        "temp": np.atleast_1d(rng.normal(23)),
        "mask": rng.random((1, 80, 320)) > .5,
    }
    for ii in range(num_batches):
        for idx in number_order:
            event_queue.put((ii*batch_size + idx, event))


def main():
    thr_coll = write.QueueCollectorThread(
        event_queue=event_queue,
        writer_dq=writer_dq,
        feat_nevents=feat_nevents,
        write_threshold=500,
    )
    thr_coll.run()
