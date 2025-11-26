import logging
import threading
import time

import pytest

from dcnum.common import join_worker


def test_join_basic():
    thr = threading.Thread(target=lambda: None)
    thr.start()
    logger = logging.getLogger(__name__)
    join_worker(worker=thr,
                timeout=1,
                retries=10,
                logger=logger,
                name="hans"
                )


def test_join_basic_timeout():
    thr = threading.Thread(target=lambda: time.sleep(.3))
    thr.start()
    logger = logging.getLogger(__name__)
    join_worker(worker=thr,
                timeout=.1,
                retries=10,
                logger=logger,
                name="hans"
                )


def test_join_basic_timeout_error():
    thr = threading.Thread(target=lambda: time.sleep(2))
    thr.start()
    logger = logging.getLogger(__name__)
    with pytest.raises(ValueError, match="did not join"):
        join_worker(worker=thr,
                    timeout=.1,
                    retries=1,
                    logger=logger,
                    name="hans"
                    )
