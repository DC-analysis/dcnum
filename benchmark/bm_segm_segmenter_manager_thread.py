import pathlib
import tempfile
import threading
import time

import multiprocessing as mp

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
                                          num_procs=6,
                                          )

        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.task_background()

        self.slot_register = logic.SlotRegister(job=self.job,
                                                data=self.runner.dtin,
                                                num_slots=2)
        self.u_worker = logic.UniversalWorkerThread(
            slot_register=self.slot_register)
        self.u_worker.start()

    def benchmark(self):
        seg_cls = segm.get_available_segmenters()[self.job["segmenter_code"]]
        fake_extractor = SlotStateInvalidator(slot_register=self.slot_register)
        fake_extractor.start()
        thr_segm = segm.SegmenterManagerThread(
            segmenter=seg_cls(num_workers=2, **self.job["segmenter_kwargs"]),
            slot_register=self.slot_register,
        )
        thr_segm.run()
        self.slot_register.close()
        fake_extractor.join()
        self.u_worker.join()

    def teardown(self):
        self.runner.close()


class SlotStateInvalidator(threading.Thread):
    """Pretend to be the feature extractor"""
    def __init__(self, slot_register, *args, **kwargs):
        super(SlotStateInvalidator, self).__init__(*args, **kwargs)
        self.slot_register = slot_register

    def run(self):
        while self.slot_register.state != "q":
            for cs in self.slot_register:
                if cs.state == "e":
                    time.sleep(0.1)
                    cs.state = "i"
                    break
            else:
                time.sleep(0.1)
