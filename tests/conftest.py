import atexit
import os
import shutil
import tempfile
import time

# Load the HDF5 compression filters for all tests (dcnum loads them
# lazily, which does not cover tests that use h5py directly).
import hdf5plugin  # noqa: F401

from dcnum.os_env_st import request_single_threaded


request_single_threaded()

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
