import pathlib
import tempfile

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
                                          debug=True,
                                          )

        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.task_background()

        self.slot_register = logic.SlotRegister(job=self.job,
                                                data=self.runner.dtin,
                                                num_slots=10)

        # fill up all num_slots+1 slots with data
        self.slot_register.task_load_all(logger=self.runner.logger)

        # perform segmentation
        seg_cls = segm.get_available_segmenters()[self.job["segmenter_code"]]
        segmenter = seg_cls(**self.job["segmenter_kwargs"])

        for cs in self.slot_register.slots:
            segmenter.segment_chunk(cs.chunk, self.slot_register.slots)
            cs.state = "m"

        for _ in range(11):
            self.slot_register.task_label_masks(logger=self.runner.logger)
            self.slot_register.task_process_labels(logger=self.runner.logger)

        for slot in self.slot_register.slots:
            assert slot.state == "e"

        segmenter.close()

    def benchmark(self):
        # call `task_extract_features` num_slots+1 times
        for _ in range(11):
            self.slot_register.task_extract_features(logger=self.runner.logger)

        for slot in self.slot_register.slots:
            assert slot.state == "i"

    def teardown(self):
        self.slot_register.close()

        assert self.slot_register.state == "q"
        # empty the queues
        for q in [self.runner.log_queue, self.slot_register.event_queue]:
            while True:
                try:
                    q.get(timeout=1)
                except BaseException:
                    break

        self.runner.log_queue.cancel_join_thread()
        self.slot_register.event_queue.cancel_join_thread()

        self.runner.close()
