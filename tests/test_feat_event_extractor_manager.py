import collections
import logging
import multiprocessing as mp
import queue

from dcnum.feat import (
    EventExtractorManagerThread, Gate, QueueEventExtractor
)
from dcnum.read import HDF5Data
from dcnum.segm import SegmenterManagerThread
from dcnum.segm.segm_thresh import SegmentThresh
import numpy as np

from helper_methods import retrieve_data


mp_spawn = mp.get_context("spawn")


def test_event_extractor_manager_thread():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    hd = HDF5Data(path)
    assert "image" in hd
    log_queue = mp_spawn.Queue()

    slot_chunks = mp_spawn.Array("i", 1)
    slot_states = mp_spawn.Array("u", 1)

    thr_segm = SegmenterManagerThread(
        segmenter=SegmentThresh(),
        image_data=hd.image_corr,
        slot_states=slot_states,
        slot_chunks=slot_chunks,
        debug=True,
    )
    thr_segm.start()

    fe_kwargs = QueueEventExtractor.get_init_kwargs(
        data=hd,
        gate=Gate(hd),
        num_extractors=1,
        log_queue=log_queue,
        log_level=logging.DEBUG,
    )

    thr_feat = EventExtractorManagerThread(
        slot_chunks=slot_chunks,
        slot_states=slot_states,
        fe_kwargs=fe_kwargs,
        num_workers=1,
        labels_list=thr_segm.labels_list,
        writer_dq=collections.deque(),
        debug=True)
    thr_feat.run()
    thr_segm.join()

    assert fe_kwargs["worker_monitor"][0] == 40

    index, event = fe_kwargs["event_queue"].get(timeout=1)
    # empty all queues
    for qu in [fe_kwargs["event_queue"], fe_kwargs["log_queue"]]:
        while True:
            try:
                qu.get(timeout=.1)
            except queue.Empty:
                break

    assert index == 0
    assert np.allclose(event["deform"][0], 0.07405636775888857)
