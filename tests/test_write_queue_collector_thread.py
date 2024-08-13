import collections
import multiprocessing as mp
import pathlib

import numpy as np

from dcnum import write


data_path = pathlib.Path(__file__).parent / "data"


def test_event_stash():
    feat_nevents = [1, 3, 1, 5]
    stash = write.EventStash(index_offset=0,
                             feat_nevents=feat_nevents)
    assert stash.size == 10
    assert np.all(stash.nev_idx == [1, 4, 5, 10])
    assert stash.num_frames == 4
    assert not stash.is_complete()
    stash.add_events(index=0,
                     events={"deform": np.array([.1]),
                             "area_um": np.array([100])})
    assert not stash.is_complete()
    stash.add_events(index=1,
                     events={"deform": np.array([.1, .2, .3]),
                             "area_um": np.array([100, 120, 150])})
    assert not stash.is_complete()
    stash.add_events(index=2,
                     events={"deform": np.array([.1]),
                             "area_um": np.array([100])})
    assert not stash.is_complete()
    stash.add_events(index=3,
                     events={"deform": np.array([.1, .2, .3, .4, .5]),
                             "area_um": np.array([100, 110, 120, 130, 140])})
    assert stash.is_complete()

    assert np.all(stash.events["deform"]
                  == [.1, .1, .2, .3, .1, .1, .2, .3, .4, .5])
    assert np.all(stash.events["area_um"]
                  == [100, 100, 120, 150, 100, 100, 110, 120, 130, 140])


def test_queue_collector_thread():
    # keyword arguments
    event_queue = mp.Queue()
    writer_dq = collections.deque()
    feat_nevents = np.array([1, 3, 1, 5])
    write_threshold = 2
    # queue collector thread
    qct = write.QueueCollectorThread(
        event_queue=event_queue,
        writer_dq=writer_dq,
        feat_nevents=feat_nevents,
        write_threshold=write_threshold
    )
    # put data into queue
    event_queue.put((0, {"deform": np.array([.1]),
                         "area_um": np.array([100])}))
    event_queue.put((1, {"deform": np.array([.1, .2, .3]),
                         "area_um": np.array([100, 120, 150])}))
    event_queue.put((2, {"deform": np.array([.1]),
                         "area_um": np.array([100])}))
    event_queue.put((3, {"deform": np.array([.1, .2, .3, .4, .5]),
                         "area_um": np.array([100, 110, 120, 130, 140])}))
    # collect information from queue (without the threading part)
    qct.run()

    # Test whether everything is in order.
    # We have a write threshold of 2, so there should data in batches of two
    # frames stored n the writer_dq.

    # BATCH 1
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in ["index_unmapped", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    # BATCH 2
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3, .4, .5])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in ["index_unmapped", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    assert len(writer_dq) == 0


def test_queue_collector_thread_with_full_stash():
    # keyword arguments
    event_queue = mp.Queue()
    writer_dq = collections.deque()
    feat_nevents = np.array([1, 3, 1, 5])
    write_threshold = 2
    # queue collector thread
    qct = write.QueueCollectorThread(
        event_queue=event_queue,
        writer_dq=writer_dq,
        feat_nevents=feat_nevents,
        write_threshold=write_threshold
    )

    # This is what this test here is about. We fill up the buffer_dq which
    # essentially means:
    # "In a previous iteration of the main loop in `run`,
    # all events for the current slice (context: write_threshold) have already
    # been placed in the buffer_dq" or
    # "There was one segmentation worker
    # that was really slow in the second-last iteration, slower than all
    # the other workers that filled up the event_queue for next slice before
    # that worker".
    # What we do is filling up the buffer_dq instead of the event_queue.
    qct.buffer_dq.append((0, {"deform": np.array([.1]),
                              "area_um": np.array([100])}))
    qct.buffer_dq.append((1, {"deform": np.array([.1, .2, .3]),
                              "area_um": np.array([100, 120, 150])}))
    qct.buffer_dq.append((2, {"deform": np.array([.1]),
                              "area_um": np.array([100])}))
    qct.buffer_dq.append((3, {"deform": np.array([.1, .2, .3, .4, .5]),
                              "area_um": np.array([100, 110, 120, 130, 140])}))

    # The following call will hang indefinitely, if the buffer_dq is not
    # emptied prior to the while loop that polls event_queue.
    qct.run()

    # Test whether everything is in order.
    # We have a write threshold of 2, so there should data in batches of two
    # frames stored n the writer_dq.

    # BATCH 1
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in ["index_unmapped", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    # BATCH 2
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3, .4, .5])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in ["index_unmapped", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    assert len(writer_dq) == 0
