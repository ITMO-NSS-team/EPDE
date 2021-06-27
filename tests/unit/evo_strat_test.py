#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:33:48 2021

@author: mike_ubuntu
"""

import sys
import getopt

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1])

class dummy_operator():
    def __init__(self, test_output):
        self._val = test_output
        
    def apply(self, arg):
        return str(arg) + self._val

def test_basic_strat():
#    print('paths:', sys.path, 'added:', opt[0][1])
    from epde.src.eq_search_strategy import Input_block, Evolutionary_block, link, Evolutionary_strategy
    from epde.src.operators.ea_stop_criteria import Iteration_limit
#    from epde.src.operators import 
    input_op = Input_block('a')
    do1 = dummy_operator('1'); do2 = dummy_operator('2'); do3 = dummy_operator('3'); 
    operator1 = Evolutionary_block(do1)
    operator2 = Evolutionary_block(do2)
    operator3 = Evolutionary_block(do3, terminal = True)
    
    get_0th = lambda x: x[0]
    for op in [operator1, operator2, operator3]:
        op.set_input_combinator(get_0th)
    link(input_op, operator1); link(operator1, operator2); link(operator2, operator3)
    
    evo_strategy = Evolutionary_strategy(Iteration_limit, {'limit' : 100})
    evo_strategy.create_linked_blocks({'initial':input_op, '1st':operator1, '2nd':operator2, 
                                       '3rd':operator3}, suppress_structure_check = True)
    print(evo_strategy.iteration('a'))
    
    
def test_director():
    from epde.src.eq_search_strategy import Strategy_director
    from epde.src.operators.ea_stop_criteria import Iteration_limit
    test_strat = Strategy_director(Iteration_limit, {'limit' : 100})
    test_strat.strategy_assembly()
    test_strat._constructor._strategy.check_integrity()

