import pathlib
import tempfile
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
                                          debug=True,
                                          )

        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.task_background()

        log_queue = mp_spawn.Queue()
        log_queue.cancel_join_thread()

        self.slot_register = logic.SlotRegister(job=self.job,
                                                data=self.runner.dtin,
                                                num_slots=2)
        self.u_worker = logic.UniversalWorkerThread(
            slot_register=self.slot_register,
            log_queue=log_queue,
        )
        self.u_worker.start()

    def benchmark(self):
        seg_cls = segm.get_available_segmenters()[self.job["segmenter_code"]]

        segmenter = seg_cls(**self.job["segmenter_kwargs"], debug=True)

        for chunk in range(self.slot_register.num_chunks):
            while True:
                # Find the slot that has the `chunk`
                # (preloaded from disk by UniversalWorker)
                cs = self.slot_register.find_slot(state="s", chunk=chunk)
                if cs is None:
                    time.sleep(.01)
                else:
                    break

            state_warden = self.slot_register.reserve_slot_for_task(
                current_state="s",
                next_state="i",
                chunk_slot=cs,
                batch_size=None)
            if state_warden is not None and state_warden.batch_size:

                with state_warden:
                    # We have a free slot to compute the segmentation
                    # `segment_chunk` populates the `cs.labels` array.
                    segmenter.segment_chunk(cs.chunk, self.slot_register.slots)

        segmenter.close()

    def teardown(self):
        self.slot_register.close()
        self.u_worker.join()
        self.runner.close()
