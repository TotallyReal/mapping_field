import logging
from pathlib import Path
import functools

import pytest

FULL_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
SIMPLE_FORMAT = '%(message)s'

def log_to_file(log_format: str = FULL_FORMAT):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Fixture to capture logs of a single test into a file.
            """
            # Determine log folder
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)

            # Build log filename: <test_file>__<test_name>.log
            test_file = Path(func.__code__.co_filename).stem
            test_name = func.__name__
            log_file = log_dir / f"{test_file}__{test_name}.log"

            # Create file handler
            handler = logging.FileHandler(log_file, mode="w")
            handler.setFormatter(logging.Formatter(log_format))

            # Get root logger and attach handler
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)  # capture DEBUG+ logs

            try:
                # Run the test
                return func(*args, **kwargs)
            finally:
                # Cleanup handler
                logger.removeHandler(handler)
                handler.close()

        return wrapper
    return decorator


_DEBUG_STATE = 0

def debug_step(value: int):
    global _DEBUG_STATE
    if _DEBUG_STATE == value:
        _DEBUG_STATE += 1
