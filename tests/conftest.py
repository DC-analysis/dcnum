import atexit
import os
import shutil
import tempfile
import time

from dcnum.os_env_st import request_single_threaded

# Request single-threaded computation before numpy is imported (e.g.
# via hdf5plugin/h5py below), so that the tests compute in the same way
# as the worker processes (e.g. identical results in debug mode).
request_single_threaded()

# Load the HDF5 compression filters for all tests (dcnum loads them
# lazily, which does not cover tests that use h5py directly).
import hdf5plugin  # noqa: E402, F401

TMPDIR = tempfile.mkdtemp(prefix=time.strftime(
    "dcnum_test_%H.%M_"))


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    tempfile.tempdir = TMPDIR
    atexit.register(shutil.rmtree, TMPDIR, ignore_errors=True)
    # Disable JIT compiler during testing for coverage
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
