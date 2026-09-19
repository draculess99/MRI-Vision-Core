"""Real MRNet access is opt-in; the default suite is entirely synthetic."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-mrnet", action="store_true", default=False,
                     help="Run optional read-only checks against the local MRNet dataset")


def pytest_configure(config):
    config.addinivalue_line("markers", "mrnet: optional read-only local MRNet integration check")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-mrnet"):
        skip = pytest.mark.skip(reason="Local MRNet check requires --run-mrnet")
        for item in items:
            if "mrnet" in item.keywords:
                item.add_marker(skip)
