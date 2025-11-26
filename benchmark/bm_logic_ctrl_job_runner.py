import pathlib
import shutil
import tempfile

import multiprocessing as mp

import h5py

from dcnum import logic


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
                                          num_procs=6,
                                          )
        self.runner = None

    def benchmark(self):
        self.runner = logic.DCNumJobRunner(job=self.job)
        self.runner.run()

    def verify(self):
        with h5py.File(self.path_out) as h5:
            assert h5["events/mask"][0].shape == (80, 320)
            assert h5["events/deform"].shape[0] > 10000

    def teardown(self):
        if self.runner is not None:
            self.runner.close()
        shutil.rmtree(self.tmp_path, ignore_errors=True)
