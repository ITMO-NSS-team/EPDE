#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:32:56 2021

@author: mike_ubuntu
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--path", action="store")

@pytest.fixture(scope='session')
def name(request):
    path_value = request.config.option.path
    if path_value is None:
        pytest.skip()
    return path_value

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------
# Settings used to be mutable module scalars in ``epde.globals``, so tests drove
# the operators by poking them and restored them in a teardown fixture. They are
# now read from one resolved, immutable config, so a test states the config it
# wants and the helper puts it back afterwards.

from contextlib import contextmanager

from epde.interface.search_config import (load_search_config, active_config,
                                          set_active_config,
                                          reset_active_config)


@contextmanager
def using_config(**overrides):
    """Run the block with a config built from ``overrides`` (flat kwarg names).

    Restores whatever was active before, so a test cannot leak its settings
    into the next one -- the job the ``restore_metric`` teardown fixtures used
    to do for the individual globals.
    """
    previous = active_config()
    set_active_config(load_search_config(overrides=overrides))
    try:
        yield active_config()
    finally:
        set_active_config(previous)


@pytest.fixture(autouse=True)
def _isolate_search_config():
    """No test may leave a modified configuration behind."""
    previous = active_config()
    yield
    set_active_config(previous)
