#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:15:31 2023

@author: maslyaev
"""

from abc import ABC, abstractproperty

from epde.optimizers.blocks import EvolutionaryBlock, InputBlock
from epde.optimizers.strategy import Strategy

def link(op1, op2):
    '''

    Set the connection of operators in format op1 -> op2

    '''
    assert isinstance(op1, (EvolutionaryBlock, InputBlock)) and isinstance(op2, (EvolutionaryBlock, InputBlock)), 'An operator is not defined as the Placed operator object'
    op1.add_outgoing(op2)
    op2.add_incoming(op1)


class StrategyBuilder(ABC):     # OperatorBuilder
    def __init__(self, strategy_class = Strategy):
        self.reset()
    
    def reset(self):
        self.blocks_labeled = dict()
        self.blocks_connected = dict() # dict of format {op_label : (True, True)}, where in 
                                       # value dict first element is "connected with input" and
                                       # the second - "connected with output"
        self.reachable = dict()
        self.initial_label = None        
    
    @abstractproperty
    def processer(self):
        raise NotImplementedError('Tring to return a property of an abstract class.')

    def add_init_operator(self, operator_label : str = 'initial'):
        self.initial_label = operator_label
        new_block = InputBlock(to_pass = None)
        self.reachable[operator_label] = {new_block.op_id,}
        self.blocks_labeled[operator_label] = new_block        
        self.blocks_connected[operator_label] = [True, False]

    def set_input_combinator(self, non_default : dict = dict()):
        get_0th = lambda x: x[0]
        
        for key, op in self.blocks_labeled.items():
            if key not in non_default.keys():
                op.set_input_combinator(get_0th)
            else:
                op.set_input_combinator(non_default[key])

    def add_operator(self, operator_label, operator, parse_operator_args = 'inspect',
                     terminal_operator : bool = False):
        new_block = EvolutionaryBlock(operator, parse_operator_args = parse_operator_args,
                                                   terminal=terminal_operator)
        self.blocks_labeled[operator_label] = new_block
        self.reachable[operator_label] = {new_block.op_id,}
        self.blocks_connected[operator_label] = [False, terminal_operator]

    def link(self, label_out, label_in):
        link(self.blocks_labeled[label_out], self.blocks_labeled[label_in])
        self.reachable[label_out].union(self.reachable[label_in])
        self.blocks_connected[label_out][1] = True
        self.blocks_connected[label_in][0] = True
    
    def check_connectedness(self):
        return len(self.blocks_labeled) == len(self.reachable[self.initial_label])
    
    def assemble(self, suppress_structure_check = False):
        self._processer.create_linked_blocks(blocks = self.blocks_labeled, suppress_structure_check = suppress_structure_check)

    
def add_sequential_operators(builder : StrategyBuilder, operators : list):
    '''
    

    Parameters
    ----------
    builder : SectorProcesserBuilder,
        MOEADD evolutionary strategy builder (sector processer), which will contain added operators.
    operators : list,
        Operators, which will be added into the processer. The elements of the list shall be tuple in 
        format of (label, operator), where the label is str (e.g. 'selection'), while the operator is 
        an object of subclass of CompoundOperator.

    Returns
    -------
    builder : OperatorBuilder
        Modified builder.

    '''
    builder.add_init_operator('initial')
    for idx, operator in enumerate(operators):
        builder.add_operator(operator[0], operator[1], terminal_operator = (idx == len(operators) - 1))

    builder.set_input_combinator()
    builder.link('initial', operators[0][0])
    for op_idx, _ in enumerate(operators[:-1]):
        builder.link(operators[op_idx][0], operators[op_idx + 1][0])

    builder.assemble()
    return builder        

    
class OptimizationPatternDirector(object):
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : StrategyBuilder):
        print(f'setting builder with {sector_processer_builder}')
        self._builder = sector_processer_builder