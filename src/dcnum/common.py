import multiprocessing as mp
import os


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


def join_worker(worker, timeout, retries, logger, name):
    """Patiently join a worker (Thread or Process)"""
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
        raise ValueError(f"Thread '{name}' ({worker}) did not join"
                         f"within {timeout * retries}s!")
