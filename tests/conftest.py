"""
conftest.py for clairvoyance2.
"""

import pytest


def pytest_addoption(parser):
    # See https://jwodder.github.io/kbits/posts/pytest-mark-off/
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-internet",
        action="store_true",
        default=False,
        help="Run internet-requiring tests",
    )
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all tests regardless of marks",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-all"):
        if not config.getoption("--run-slow"):
            skipper = pytest.mark.skip(reason="Only run when --run-slow is given")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(skipper)
        if not config.getoption("--run-internet"):
            skipper = pytest.mark.skip(reason="Only run when --run-internet is given")
            for item in items:
                if "internet" in item.keywords:
                    item.add_marker(skipper)
