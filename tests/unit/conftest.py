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