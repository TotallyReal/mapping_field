import functools
import logging

from pathlib import Path

import pytest

from mapping_field.log_utils.tree_loggers import simplify_tree
from mapping_field.new_code.mapping_field import NamedFunc, Var


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var._instances = {}
    NamedFunc._instances = {}

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


def pytest_runtest_makereport(item, call):
    """Hook called when pytest makes a test report."""
    # We only care about the actual test call phase
    if call.when == "call" and call.excinfo is not None:
        save_logs_to_file(item)

def save_logs_to_file(item):
    # Determine log folder
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Build log filename: <test_file>__<test_name>.log
    test_file = item.fspath.basename[:-3]   # remove the '.py' at the end
    test_name = item.name
    log_file = log_dir / f"{test_file}__{test_name}.log_context"

    simplify_tree.context.save_element(log_file)

_DEBUG_STATE = 0

@pytest.fixture
def simple_logs():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))

    logging.getLogger().addHandler(handler)

def debug_step(value: int):
    global _DEBUG_STATE
    if _DEBUG_STATE == value:
        _DEBUG_STATE += 1
