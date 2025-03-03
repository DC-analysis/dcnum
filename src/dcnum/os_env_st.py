import logging
import os


logger = logging.getLogger(__name__)

os_env_threading = [
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMPY_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]
"""environment variables that define number of threads libraries will use"""


class RequestSingleThreaded:
    """Context manager for starting a process with specific environment

    When entering the context, the environment variables defined in
    ``os_env_threading`` are all set to "1", telling the relevant libraries
    that they should work in single-threaded mode.
    When exiting the context, these environment variables are reset to
    their original values (or unset if applicable).

    Note that it makes only sense to use this context manager when
    starting new multiprocessing processes. When the process spawns,
    the environment from the current thread is copied. Setting the
    environment variable after e.g. importing numpy has no effect
    on how many threads numpy will use.
    """
    def __init__(self):
        self.previous_env = {}

    def __enter__(self):
        """Ask nicely for single-threaded computation using `os.environ`

        Note that this only affects new processes in which the
        relevant libraries have not yet been imported.
        """
        for key in os_env_threading:
            if key in os.environ:
                self.previous_env[key] = os.environ[key]
            os.environ[key] = "1"
        return self

    def __exit__(self, type, value, traceback):
        """Restore the previous environment"""
        for key in os_env_threading:
            if key not in self.previous_env:
                os.environ.pop(key)
            else:
                os.environ[key] = self.previous_env[key]


def confirm_single_threaded():
    """Warn via logs when environment variables are not set to single thread"""
    # Sanity checks
    for os_env in os_env_threading:
        # You should disable multithreading for all major tools that
        # use dcnum.logic. We don't want multithreading, because dcnum
        # uses linear code and relies on multiprocessing for
        # parallelization. This has to be done before importing numpy
        # or any other library affected. In your scripts, you can use:
        #
        val_act = os.environ.get(os_env)
        if val_act != "1":
            logger.warning(
                f"Make sure to set the environment variable {os_env} to "
                f"'1' (disables multithreading)! Other values will reduce "
                f"performance and your system may become unresponsive. "
                f"The current value is '{val_act}'.")


def request_single_threaded():
    """Set the environment variable to single thread

    This function must be called before importing the multithreaded
    libraries (such as numpy) in order for them to pick up the
    environment variables.
    """
    for key in os_env_threading:
        os.environ[key] = "1"
