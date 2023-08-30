#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:28:40 2023

@author: maslyaev
"""

from typing import Callable
from abc import ABC, abstractproperty

import numpy as np

class Block(ABC):
    """
    Base class, that is used to build other base classes

    Attributes:
        _incoming (`list`): 
        _outgoing (`list`):
        id_set (`boolean`):
        initial (`boolean`): 
        applied (`boolean`): flag, that the block has been applied
        terminal (`boolean`): The terminality marker of the evolutionary block: if ``True``, than the execution of 
            the LinkedBlocks will terminate after appying the block's operator.
        combinator (`Callable`): Method to define, how the block performs the combination of inputs, passed from "upper" blocks
    """
    def __init__(self, initial = False, terminal = False):
        self._incoming = []; self._outgoing = []
        self.id_set = False; self.initial = initial
        self.applied = False; self.terminal = terminal
    
    def add_incoming(self, incoming):
        self._incoming.append(incoming)
        
    def add_outgoing(self, outgoing):
        self._outgoing.append(outgoing)    

    @property
    def op_id(self):
        if not self.id_set:
            self._id = np.random.randint(0, 1e6)
            self.id_set = True
        return self._id
    
    @property
    def available(self):
        return all([block.applied for block in self._incoming])

    def apply(self, *args):
        self.applied = True

    def set_input_combinator(self, combinator : Callable):
        """
        Method to define, how the block performs the combination of inputs, passed from "upper" 
        blocks. In most scenarios, this input will be a list of one element (from a single 
        upper block), thus the combinator shall be a (lambda) function to select the first 
        element of the list.
        """
        self.combinator = combinator


class EvolutionaryBlock(Block):
    """
    Class, that represents an evolutionary operator, placed into the analogue of the 
    "computational graph" of the iteration of evolutionary algorithm.
    
    Attributes:
        _operator (`epde.operators.utils.template.Compound_Operator`): The evolutionary operator, contained by the block.
        terminal (`boolean`): The terminality marker of the evolutionary block: if ``True``, than the execution of 
            the LinkedBlocks will terminate after appying the block's operator.
        args_keys (`list`): names of arguments of `_opeartor`
        op_id (`integer`): individual operator's id
    """
    def __init__(self, operator, parse_operator_args = None, terminal = False): #, initial = False
        """
        Args:
            operator (`epde.operators.utils.template.Compound_Operator`): The evolutionary operator, contained by the block.
            parse_operator_args (`str|tuple|list`): Approach to getting additional arguments for operator.apply method. If ``None``, 
                no arguments will be obtained, if tuple, then the elements of the tuple (of type ``str``) will be considered as the keys. Other options are 'use inspect' or 'use operator attribute'.
                In the former case, the ``inspect`` library will be used, while in latter the 
                class will try to take the arguments from the operator object.
            terminal (`boolean`): The terminality marker of the evolutionary block: if ``True``, than the execution of 
                the LinkedBlocks will terminate after appying the block's operator.
        """
        self._operator = operator
        if parse_operator_args is None:
            self.arg_keys = []
        elif parse_operator_args == 'inspect':
            self.arg_keys = self._operator.get_suboperator_args()
            extra = ['self', 'population_subset', 'population']
            for arg_key in extra:
                try:
                    self.arg_keys.remove(arg_key)
                except:
                    pass
        elif parse_operator_args == 'use operator attribute':
            self.arg_keys = self._operator.arg_keys
        elif isinstance(parse_operator_args, (list, tuple)):
            self.arg_keys = parse_operator_args
        else:
            raise TypeError('Wrong argument passed as the parse_operator_args argument')
        super().__init__(terminal = terminal)
        
    def check_integrity(self):
        """
        Checking block:
            - that block-terminal does not have any output (attribute `_outgoing`)
            - that block (not terminal) have any output (attribute `_outgoing`)
        """
        if self.terminal and len(self._outgoing) > 0:
            raise ValueError('The block is set as the terminal, while it has some output')
        if not self.terminal and len(self._outgoing) == 0:
            raise ValueError('The block is not set as the terminal, while it has no output')
    
    def apply(self, EA_kwargs):
        """
        Applying `operator` with arguments in the evolution algorithm by now

        Args:
            EA_kwargs (`dict`): dictionary with names and values for parameters in the evolution algorithm

        Returns:
            None
        """
        self.check_integrity()
        kwargs = {kwarg_key : EA_kwargs[kwarg_key] for kwarg_key in self.arg_keys}
        self.output = self._operator.apply(self.combinator([block.output for block in self._incoming]),
                                           arguments = kwargs)
        super().apply()

    @property
    def op_id(self):
        if not self.id_set:
            self._id = np.random.randint(0, 1e6) #self._operator.op_id
            self.id_set = True
        return self._id


class InputBlock(Block):
    """
    Blocks with fucntionality running before each evolutionary step
    """
    def __init__(self, to_pass):
        self.set_output(to_pass)
        self.applied = True
        super().__init__(initial = True)

    def add_incoming(self, incoming):
        raise TypeError('Objects of this type shall not have incoming links from other operators')

    def set_output(self, to_pass):
        self.output = to_pass
        
    @property
    def available(self):
        return True
        
    
class LinkedBlocks(object):
    """
    The sequence (not necessarily chain: divergencies can be present) of blocks with evolutionary
    operators; it represents the modular and customizable structure of the evolutionary operator.

    Attributes:
        blocks_labeled (`dict`): dictionary with names and arguments of operators 
        supress_structure_check (`boolean`): checking of structure
        initial (`list`): keeping initialized evolution operators
        output (`ndarray`): result from applying operator with evoluyion algorithm's arguments
    """
    def __init__(self, blocks_labeled : dict, suppress_structure_check : bool = False):
        self.blocks_labeled = blocks_labeled
        self.suppress_structure_check = suppress_structure_check
        self.initial = [(idx, block) for idx, block in enumerate(self.blocks_labeled.values()) if block.initial]
        self.terminated = False
        if len(self.initial) > 1:
            raise NotImplementedError('The ability to process the multiple initial blocks is not implemented')
    
    def reset_traversal_cond(self):
        for block in self.blocks_labeled.values():
            if not isinstance(block, InputBlock): block.applied = False 
            
    def traversal(self, input_obj, EA_kwargs):
        '''
        Sequential execution of the evolutionary algorithm's blocks.

        Args:
            input_obj (`ndarray`): source data
            EA_kwargs (`dict`): dictionary with names and values for parameters in the evolution algorithm
        '''
        self.reset_traversal_cond()
        
        self.initial[0][1].set_output(input_obj)
        delayed = []
        processed = [self.initial[0][1],]
        self.terminated = False
        while not self.terminated:
            processed_new = []
            for vertex in processed:
                if vertex.available:
                    vertex.apply(EA_kwargs)
                    processed_new.extend(vertex._outgoing)
                    if vertex.terminal:
                        self.terminated = True
                        self.final_vertex = vertex
                else:
                    delayed.append(vertex)
            processed = delayed + processed_new
            delayed = []
        
    @property
    def output(self):
        return self.final_vertex.output