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


class StrategyBuilder():     # OperatorBuilder  ABC
    """
    Class instance for building a strategy

    Attributes:
        processer (`Strategy`): class strategy, that is used in this algorithm
        blocks_labeled (`dict`): dictionary with operator blocks in evolutionary algorithm 
        blocks_connected (`dict`): dictionary with names of operators and list with boolean flags about "neighbours" (some operator before and some operator after)
        reachable (`dict`): dictionary with operators, that definitely not a single node: key - operator's name, item - block's id
        initial_label (`str`): name of operator, that is executed first in the queue of operators
    """
    def __init__(self, strategy_class = Strategy):
        self.strategy_class = strategy_class
        self.reset(strategy_class)
    
    def reset(self, strategy_class):
        """
        Setting initial parameters for evolutionary strategy

        Args:
            strategy_class (`Strategy`): class with choosing optimization strategy

        Returns:
            None
        """
        self._processer = strategy_class()
        self.blocks_labeled = dict()
        self.blocks_connected = dict() # dict of format {op_label : (True, True)}, where in 
                                       # value dict first element is "connected with input" and
                                       # the second - "connected with output"
        self.reachable = dict()
        self.initial_label = None        
    
    @abstractproperty
    def processer(self):
        # raise NotImplementedError('Tring to return a property of an abstract class.')
        return self._processer

    def add_init_operator(self, operator_label : str = 'initial'):
        """
        Setting operator, that is started optimization

        Args:
            operator_label (`str`): name of operator, that is started operator's quene
        
        Returns:
            None
        """
        self.initial_label = operator_label
        new_block = InputBlock(to_pass = None)
        self.reachable[operator_label] = {new_block.op_id,}
        self.blocks_labeled[operator_label] = new_block        
        self.blocks_connected[operator_label] = [True, False]

    def set_input_combinator(self, non_default : dict = dict()):
        """
        Setting callable method for each operator, that performs a combination operation with input data

        Args:
            non_default (`dict`): dictionary with callable methods. 
                key - name of operator, value - callable method for combination input data by the operator
            
        Returns:
            None
        """
        get_0th = lambda x: x[0]
        
        for key, op in self.blocks_labeled.items():
            if key not in non_default.keys():
                op.set_input_combinator(get_0th)
            else:
                op.set_input_combinator(non_default[key])

    def add_operator(self, operator_label, operator, parse_operator_args = 'inspect',
                     terminal_operator : bool = False):
        """
        Adding new evolutionary operator

        Args:
            operator_label (`str`): name of new operator
            operator (`Block`): directly the operator himself
            parse_operator_args (`str`): 
            terminal_operator (`boolean`): The terminality marker of the evolutionary block: if ``True``, than the execution of 
                the LinkedBlocks will terminate after appying the block's operator.
        
        Return:
            None
        """
        new_block = EvolutionaryBlock(operator, parse_operator_args = parse_operator_args,
                                                   terminal=terminal_operator)
        self.blocks_labeled[operator_label] = new_block
        self.reachable[operator_label] = {new_block.op_id,}
        self.blocks_connected[operator_label] = [False, terminal_operator]

    def link(self, label_out, label_in):
        """
        Create connect between operators (the result of one goes to the input of the other)

        Args:
            label_out (`str`): name of operator, which goes to the entrance of the second
            label_in (`str`): name of operator, which takes the result of the first

        Returns:
            None
        """
        link(self.blocks_labeled[label_out], self.blocks_labeled[label_in])
        self.reachable[label_out].union(self.reachable[label_in])
        self.blocks_connected[label_out][1] = True
        self.blocks_connected[label_in][0] = True
    
    def check_connectedness(self):
        return len(self.blocks_labeled) == len(self.reachable[self.initial_label])
    
    def assemble(self, suppress_structure_check = False):
        self._processer.create_linked_blocks(blocks = self.blocks_labeled, suppress_structure_check = suppress_structure_check)

    
def add_sequential_operators(builder : StrategyBuilder, operators : list):
    """
    Adding to builder operators and create link between them

    Args:
        builder (`SectorProcesserBuilder`): MOEADD evolutionary strategy builder (sector processer), which will contain added operators.
        operators (`list`): Operators, which will be added into the processer. The elements of the list shall be tuple in 
            format of (label, operator), where the label is str (e.g. 'selection'), while the operator is 
            an object of subclass of CompoundOperator.

    Returns:
        builder (`OperatorBuilder`): Modified builder.
    """
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
    """
    Base class of director for optimization's strategy

    Attributes:
        builder (`StrategyBuilder`)
    """
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : StrategyBuilder):
        print(f'setting builder with {sector_processer_builder}')
        self._builder = sector_processer_builder
