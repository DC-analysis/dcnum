import pathlib
import tempfile
import threading
import time

import multiprocessing as mp

import numpy as np

from dcnum import logic
from dcnum import segm

mp_spawn = mp.get_context('spawn')

here = pathlib.Path(__file__).parent


class Benchmark:
    def __init__(self):
        # generate a job
        path_in = here / "cache" / "2025-02-09_09.46_M003_Reference_30000.rtdc"
        if not path_in.is_file():
            raise ValueError(
                f"Please download '{path_in.name}' from "
                f"https://dcor.mpl.mpg.de/dataset/naiad-reference-data "
                f"and place it in the '{path_in.parent}' directory.")

        self.tmp_path = tempfile.mkdtemp(prefix=pathlib.Path(__file__).name)
        self.path_out = pathlib.Path(self.tmp_path) / "out.rtdc"

        self.job = logic.DCNumPipelineJob(path_in=path_in,
                                          path_out=self.path_out,
                                          basin_strategy="tap",
                                          segmenter_code="thresh",
                                          )

        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.task_background()

    def benchmark(self):
        seg_cls = segm.get_available_segmenters()[self.job["segmenter_code"]]
        num_slots = 2
        slot_chunks = mp_spawn.Array("i", num_slots, lock=False)
        slot_states = mp_spawn.Array("u", num_slots, lock=False)
        stop_event = threading.Event()

        fake_extractor = SlotStateInvalidator(slot_states=slot_states,
                                              slot_chunks=slot_chunks,
                                              stop_event=stop_event)
        fake_extractor.start()

        thr_segm = segm.SegmenterManagerThread(
            segmenter=seg_cls(**self.job["segmenter_kwargs"]),
            image_data=self.runner.dtin.image_corr,
            bg_off=(self.runner.dtin["bg_off"]
                    if "bg_off" in self.runner.dtin else None),
            slot_states=slot_states,
            slot_chunks=slot_chunks,
        )
        thr_segm.run()

        stop_event.set()
        fake_extractor.join()

    def teardown(self):
        self.runner.close()


class SlotStateInvalidator(threading.Thread):
    """Pretend to be the feature extractor"""
    def __init__(self, slot_states, slot_chunks, stop_event, *args, **kwargs):
        super(SlotStateInvalidator, self).__init__(*args, **kwargs)
        self.slot_states = slot_states
        self.slot_chunks = slot_chunks
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            for cur_slot in np.argsort(self.slot_chunks):
                if self.slot_states[cur_slot] == "e":
                    time.sleep(0.1)
                    self.slot_states[cur_slot] = "s"
                    break
            else:
                time.sleep(0.1)
