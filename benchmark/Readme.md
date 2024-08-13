This directory contains benchmarking scripts used for optimizing dcnum performance.
To run all benchmarks, execute `python benchmark.py`. You can also specify
individual benchmarks or a list of benchmarks (path to `bm_*.py` file)
as arguments to `benchmark.py`.

The benchmarks are also ideal use cases for identifying bottlenecks with
tools such as [line profiler](https://kernprof.readthedocs.io/en/latest/),
since benchmarks can be designed to run in single threads.

    pip install line_profiler
    kernprof -lv benchmark.py
