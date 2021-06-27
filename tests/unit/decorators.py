#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:07:06 2021

@author: mike_ubuntu
"""

import sys
import getopt

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1] + '/epde') # $1 "--path=" pwd

from src.decorators import History_Extender #Extend_history,

def test_grids_cache():
#    print(opt[0][0], args, sys.argv)
#    print(sys.path)
#    raise Exception('Done and done')
    class dummy():
        def __init__(self):
            self._history = f'Object created'
            self.state = ' test state '
            
        def add_history(self, string = ''):
            self._history += string
            
        @History_Extender(' -> done something', 'ba')
        def something_ba(self):
            print(1)

        @History_Extender(' -> done something', 'a')            
        def something_a(self):
            print(1)    

        @History_Extender(' -> done something', 'b')            
        def something_b(self):
            print(1)    

    dum = dummy()
    dum.something_ba()
    print(dum._history)
    
    dum.something_b()
    print(dum._history)

    dum.something_a()
    print(dum._history)
#    raise NotImplementedError()
        