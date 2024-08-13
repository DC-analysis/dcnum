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


def run_benchmark(bm_path):
    print_underline(f"Running {bm_path}")
    bm_path = pathlib.Path(bm_path).resolve()
    os.chdir(f"{bm_path.parent}")
    bm_mod = importlib.import_module(f"{bm_path.stem}")
    t = timeit.Timer(bm_mod.main)
    res = t.repeat(repeat=5, number=10)
    print(f"best={min(res):.3g}, mean={np.mean(res):.3g}")
    return res


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
