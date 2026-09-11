from collections.abc import Callable
import importlib
import logging
import multiprocessing as mp
import os
import threading
import time

import psutil


class DCNUMHaltInterrupt(BaseException):
    """Used for generally halting all tasks"""


class LazyLoader:
    def __init__(self,
                 modname: str,
                 sibling: str | None = None,
                 action_before_import: Callable | None = None,
                 action_after_import: Callable | None = None,
                 ):
        """Lazily load a module

        Parameters
        ----------
        modname: str
            The name of the module (e.g. ``"scipy.ndimage"``)
        sibling: str
            The ``__name__`` of a sibling of the module. This is useful
            for performing relative imports. Consider this module
            structure:

            - ``module``
              - ``submod_1``
              - ``submod_2``

            If ``submod_1`` would like to lazily import ``submod_2``::

                submod_2 = LazyLoader("submod_2", sibling==__name__)
        action_before_import: Callable
            Called before the import.
        action_after_import: Callable
            Method that should be called after the actual import.
            Must accept the module as an argument. This is useful
            if any setup steps need to be made after import (e.g.
            for ensuring reproducibility).
        """
        if sibling:
            sibling = sibling.rsplit(".", 1)[0]
            modname = f"{sibling}.{modname}"
        self._modname = modname
        self._mod = None
        self._action_before = action_before_import
        self._action_after = action_after_import

    def __getattr__(self, attr):
        """If the module is accessed, load it and return what was asked for"""
        try:
            return getattr(self._mod, attr)
        except AttributeError:
            if self._mod is None:
                # before import
                if self._action_before is not None:
                    self._action_before()
                # module is unset, load it
                self._mod = importlib.import_module(self._modname)
                # after import
                if self._action_after is not None:
                    self._action_after(self._mod)
            else:
                # Module is loaded or does not exist,
                # exception unrelated to LazyLoader.
                raise

        # retry getattr if module was just loaded for first time
        # call this outside exception handler in case it raises new exception
        return getattr(self._mod, attr)

    def module_available(self):
        """Check whether the module is available by reading its version"""
        try:
            self.__getattr__("__version__")
        except (ImportError, AttributeError):
            return False
        else:
            return True


def cpu_count() -> int:
    """Get the number of processes available

    ``multiprocessing.cpu_count()`` returns the number of logical CPUs
    available. We are interested in the physical CPUs, because there is
    no performance advantage to using all logical CPUs and because using
    more CPUs would imply more workers and thus more RAM usage.

    On cluster systems, with jobs possibly having CPU affinity, we want
    to stick to that limitation as well.

    This method handles both cases by:

    1. Get the number of physical CPUs using `psutil.cpu_count(logical=False)`
    2. Get the number of CPUs according to process affinity
    3. Get the number of CPUs using `multiprocessing.cpu_count()`
    4. Return the minimum of all of the above
    """
    num_cpus = []

    # physical CPUs
    cpu_physical = psutil.cpu_count(logical=False)
    if cpu_physical is not None:
        num_cpus.append(cpu_physical)

    # CPUs according to process affinity
    if hasattr(os, "sched_getaffinity"):
        cpu_affin = len(os.sched_getaffinity(0))
    elif hasattr(os, "process_cpu_count"):
        cpu_affin = os.process_cpu_count()
    else:
        cpu_affin = os.cpu_count()
    if cpu_affin is not None:
        num_cpus.append(cpu_affin)

    # Fallback
    num_cpus.append(mp.cpu_count())

    return min(num_cpus)


def join_worker(worker,
                timeout=30,
                retries=10,
                logger=None,
                name=None):
    """Patiently join a worker (Thread or Process)"""
    logger = logger or logging.getLogger(__name__)
    for _ in range(retries):
        worker.join(timeout=timeout)
        if worker.is_alive():
            logger.info(f"Waiting for '{name}' ({worker}")
        else:
            if hasattr(worker, "close"):
                worker.close()
            logger.debug(f"Joined thread '{name}'")
            break
    else:
        logger.error(f"Failed to join thread '{name}'")
        raise ValueError(f"Thread '{name}' ({worker}) did not join "
                         f"within {timeout * retries}s!")


def start_workers_threaded(worker_list, logger, name):
    def target(worker_list, logger, name):
        tw0 = time.perf_counter()
        w = None
        ii = 0
        try:
            for ii, w in enumerate(worker_list):
                w.start()
        except BrokenPipeError:
            logger.error(f"Worker {ii+1}: {w} failed to spawn")
            raise DCNUMHaltInterrupt(f"Worker {ii+1}: {w} failed to spawn")
        else:
            logger.info(f"{len(worker_list)} {name} spawn time: "
                        f"{time.perf_counter() - tw0:.1f}s")

    thr = threading.Thread(target=target,
                           args=(worker_list, logger, name),
                           )
    thr.start()
    return thr


def setup_h5py(h5py):
    """Hook for LazyLoader that imports hdf5plugin"""
    # Make sure hdf5plugin is loaded so we can access zstd-compressed data
    import hdf5plugin  # noqa: F401


h5py = LazyLoader("h5py", action_after_import=setup_h5py)
"""Lazily loaded h5py module"""
