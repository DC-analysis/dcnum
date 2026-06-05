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
                                          )

        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.task_background()

        log_queue = self.runner.log_queue
        log_queue.cancel_join_thread()

        self.slot_register = logic.SlotRegister(job=self.job,
                                                data=self.runner.dtin,
                                                num_slots=6)
        self.slot_register.event_queue.cancel_join_thread()

        self.u_workers = []
        for _ in range(5):
            uw = logic.UniversalWorkerProcess(
                slot_register=self.slot_register,
                log_queue=log_queue,
            )
            self.u_workers.append(uw)
            uw.start()

    def benchmark(self):
        seg_cls = segm.get_available_segmenters()[self.job["segmenter_code"]]
        thr_segm = segm.SegmenterManagerThread(
            segmenter=seg_cls(num_workers=6, **self.job["segmenter_kwargs"]),
            slot_register=self.slot_register,
        )
        thr_segm.run()
        thr_segm.segmenter.close()

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

        for uw in self.u_workers:
            uw.join()

        self.runner.close()
