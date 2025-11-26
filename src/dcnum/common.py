def join_worker(worker, timeout, retries, logger, name):
    """Patiently join a worker (Thread or Process)"""
    for _ in range(retries):
        worker.join(timeout=timeout)
        if worker.is_alive():
            logger.info(f"Waiting for '{name}' ({worker}")
        else:
            logger.debug(f"Joined thread '{name}'")
            break
    else:
        logger.error(f"Failed to join thread '{name}'")
        raise ValueError(f"Thread '{name}' ({worker}) did not join"
                         f"within {timeout * retries}s!")
