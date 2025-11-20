def join_thread_helper(thr, timeout, retries, logger, name):
    for _ in range(retries):
        thr.join(timeout=timeout)
        if thr.is_alive():
            logger.info(f"Waiting for '{name}' ({thr}")
        else:
            logger.debug(f"Joined thread '{name}'")
            break
    else:
        logger.error(f"Failed to join thread '{name}'")
        raise ValueError(f"Thread '{name}' ({thr}) did not join"
                         f"within {timeout * retries}s!")
