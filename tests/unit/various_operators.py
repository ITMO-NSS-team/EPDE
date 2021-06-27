#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:47:11 2021

@author: mike_ubuntu
"""

import sys
import getopt

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1])

def test_mutation():
    raise NotImplementedError('TBD')
    
def test_selection():
    raise NotImplementedError('TBD')

def test_crossover():
    raise NotImplementedError('TBD')

def test_rps():
    raise NotImplementedError('TBD')

# TBD: write these tests and add tests for suboperators