#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:32:56 2021

@author: mike_ubuntu
"""

import pytest

def pytest_addoption(parser):
    """
    Adds a command-line option `--path` to pytest.
    
    This option allows users to specify a path relevant to equation discovery,
    potentially influencing data loading or saving locations.
    
    Args:
        parser: The pytest argument parser.
    
    Returns:
        None.
    """
    parser.addoption("--path", action="store")

@pytest.fixture(scope='session')
def name(request):
    """
    Return the path to the data directory specified in the pytest configuration.
    
    This fixture retrieves the path to the data directory. If no path is provided via the command line, the test is skipped, ensuring that tests requiring data are only executed when the data path is explicitly specified.
    
    Args:
        request: The pytest request object, used to access configuration values.
    
    Returns:
        str: The path to the data directory.
    """
    path_value = request.config.option.path
    if path_value is None:
        pytest.skip()
    return path_value