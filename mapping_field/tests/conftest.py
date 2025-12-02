import functools
import logging
from logging import Logger

from pathlib import Path
from typing import Any

import pytest

from _pytest.nodes import Item
from _pytest.runner import CallInfo

from mapping_field.global_config import PROJECT_ROOT, SRC_ROOT
from mapping_field.log_utils.tree_loggers import simplify_tree
from mapping_field.mapping_field import NamedFunc, Var, simplifier_context


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var._instances = {}
    NamedFunc._instances = {}
    simplifier_context.clear()


@pytest.fixture(autouse=True)
def reset_logs():
    simplify_tree.reset()

TESTS_DIR = Path(__file__).parent


FILES_ORDER = {
    "mapping_field",
    "promises",
    "property_engines",
    "arithmetics",
    "conditions"
}

def pytest_collection_modifyitems(session, config, items):
    """
    Order the tests according to FILES_ORDER.

    This function is called by pytest after tests are collected and before running them.
    """
    # --- Build file -> order mapping ---
    file_to_order = {}
    for i, name in enumerate(FILES_ORDER):
        full_name = f"{name}_test.py"

        file_path = TESTS_DIR / full_name
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} listed in FILES_ORDER does not exist")

        file_to_order[full_name] = i

    # --- Apply order markers ---
    for item in items:
        if any(item.iter_markers(name="order")):
            continue    # Already marked

        file_name = Path(item.fspath).name
        if file_name in file_to_order:
            item.add_marker(pytest.mark.order(file_to_order[file_name]))


FULL_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
SIMPLE_FORMAT = "%(message)s"

LOG_DIR = SRC_ROOT / "test/logs"


def log_to_file(file_name: str | None = None, log_format: str = FULL_FORMAT, loggers: list[str | Logger] | None = None):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Fixture to capture logs of a single test into a file.
            """

            # Log File
            LOG_DIR.mkdir(exist_ok=True)
            _file_name = file_name
            if _file_name is None:
                test_file = Path(func.__code__.co_filename).stem
                test_name = func.__name__
                _file_name = f"{test_file}__{test_name}"
            log_file = LOG_DIR / f"{_file_name}.log"

            # Create file handler
            handler = logging.FileHandler(log_file, mode="w")
            handler.setFormatter(logging.Formatter(log_format))

            # Get root logger and attach handler
            if loggers is None:
                target_loggers = [logging.getLogger()]
            else:
                target_loggers = [logging.getLogger(lg_) if isinstance(lg_, str) else lg_
                                  for lg_ in loggers]
            for lg in target_loggers:
                lg.addHandler(handler)
                lg.setLevel(logging.DEBUG)

            try:
                # Run the test
                return func(*args, **kwargs)

            finally:
                # Cleanup handler
                for lg in target_loggers:
                    lg.removeHandler(handler)
                handler.close()

        return wrapper

    return decorator


def pytest_configure(config):
    config.first_failure_logged = False


def pytest_runtest_makereport(item: Item, call: CallInfo[Any]):
    """Hook called when pytest makes a test report."""
    global _first_failure_logged

    # Run after failed tests
    if call.when == "call" and (call.excinfo is not None):

        # Only run after the first failed test
        config = item.config
        if config.first_failure_logged:
            return
        config.first_failure_logged = True

        save_logs_to_file(item)


def save_logs_to_file(item: Item):
    # Determine log folder
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Build log filename: <test_file>__<test_name>.log
    test_file = item.fspath.basename[:-3]  # remove the '.py' at the end
    test_name = item.name
    log_file = log_dir / f"{test_file}__{test_name}.log_context"

    simplify_tree.root_context.save_element(log_file)


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
