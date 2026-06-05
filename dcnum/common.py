import importlib
import logging
import multiprocessing as mp
import os
import threading
import time
from typing import Callable


class LazyLoader:
    def __init__(self,
                 modname: str,
                 sibling: str = None,
                 action: Callable = None,
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
        action: Callable
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
        self._action = action

    def __getattr__(self, attr):
        """If the module is accessed, load it and return what was asked for"""
        try:
            return getattr(self._mod, attr)
        except BaseException:
            if self._mod is None:
                # module is unset, load it
                self._mod = importlib.import_module(self._modname)
                # call the action method
                if self._action is not None:
                    self._action(self._mod)
            else:
                # Module is loaded or does not exist,
                # exception unrelated to LazyLoader.
                raise

        # retry getattr if module was just loaded for first time
        # call this outside exception handler in case it raises new exception
        return getattr(self._mod, attr)


def cpu_count() -> int:
    """Get the number of processes

    Try to get the number of CPUs the current process can use first.
    Fallback to `mp.cpu_count()`
    """
    try:
        if hasattr(os, "sched_getaffinity"):
            num_cpus = len(os.sched_getaffinity(0))
        elif hasattr(os, "process_cpu_count"):
            num_cpus = os.process_cpu_count()
        else:
            num_cpus = os.cpu_count()
    except BaseException:
        num_cpus = None

    if num_cpus is None:
        num_cpus = mp.cpu_count()
    return num_cpus


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
        for w in worker_list:
            w.start()
        logger.info(f"{len(worker_list)} {name} spawn time: "
                    f"{time.perf_counter() - tw0:.1f}s")

    thr = threading.Thread(target=target, args=(worker_list, logger, name))
    thr.start()
    return thr


def setup_h5py(h5py):
    """Hook for LazyLoader that imports hdf5plugin"""
    # Make sure hdf5plugin is loaded so we can access zstd-compressed data
    import hdf5plugin  # noqa: F401


h5py = LazyLoader("h5py", action=setup_h5py)
"""Lazily loaded h5py module"""
