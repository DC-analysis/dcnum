This directory contains benchmarking scripts used for optimizing dcnum performance.
To run all benchmarks, execute `python benchmark.py`. You can also specify
individual benchmarks or a list of benchmarks (path to `bm_*.py` file)
as arguments to `benchmark.py`.

The benchmarks are also ideal use cases for identifying bottlenecks with
tools such as [line profiler](https://kernprof.readthedocs.io/en/latest/),
since benchmarks can be designed to run in single threads.

    pip install line_profiler
    kernprof -lv benchmark.py

Note that some files require testing data.

    mkdir cache
    pushd cache
    wget --content-disposition "https://dcor.mpl.mpg.de/dataset/400ae3d4-9f8a-44f8-887d-1e4f6150deee/resource/de319c9c-8d4a-4e17-9ae1-4d57a42f4508/download/250209_blood_2025-02-09_09.46_m003_reference_30000.rtdc"
