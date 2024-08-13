import importlib
import os
import pathlib
import sys
import time
import timeit

import numpy as np


here = pathlib.Path(__file__).parent


def print_underline(msg):
    print(msg)
    print("-" * len(msg))


def run_benchmark(bm_path, repeats=5):
    print_underline(f"Running {bm_path}")
    bm_path = pathlib.Path(bm_path).resolve()
    os.chdir(f"{bm_path.parent}")
    bm_mod = importlib.import_module(f"{bm_path.stem}")

    reps = []
    print("Running...", end="\r")
    for ii in range(repeats):
        t = timeit.timeit(bm_mod.main, setup=bm_mod.setup, number=1)
        reps.append(t)
        print(f"Running {ii + 1}/{repeats}", end="\r")
    print(reps)
    print(f"best={min(reps):.3g}, mean={np.mean(reps):.3g}")
    return reps


if __name__ == "__main__":
    benchmark_paths = []
    for arg in sys.argv[1:]:
        if arg.startswith("bm_"):
            benchmark_paths.append(arg)

    if not benchmark_paths:
        print("No benchmarking script specified, running all benchmarks.")
        benchmark_paths = here.glob("bm_*.py")

    results = {}

    for bmp in sorted(benchmark_paths):
        bmp = pathlib.Path(bmp)
        print("")
        res = run_benchmark(bmp)
        with bmp.with_suffix(".txt").open("a") as fd:
            fd.write(time.strftime(f"%Y-%m-%d_%H.%M.%S {res}\n"))
        print("")
