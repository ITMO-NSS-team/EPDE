#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 19:14:25 2022

@author: maslyaev
"""

from warnings import warn
from typing import Union, Callable
from abc import ABC

import numpy as np
import time
import warnings
from copy import deepcopy
from functools import reduce

import epde.globals as global_var
from epde.moeadd.moeadd_supplementary import fast_non_dominated_sorting, slow_non_dominated_sorting, NDL_update, Equality, Inequality, acute_angle        

def link(op1, op2):
    '''
    
    Set the connection of operators in format op1 -> op2
    
    '''
    assert isinstance(op1, (EvolutionaryBlock, InputBlock)) and isinstance(op2, (EvolutionaryBlock, InputBlock)), 'An operator is not defined as the Placed operator object'
    op1.add_outgoing(op2)
    op2.add_incoming(op1)
        
class Block(ABC):
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
            self._id =  np.random.randint(0, 1e6)
            self.id_set = True
        return self._id
    
    @property
    def available(self):
        return all([block.applied for block in self._incoming])

    def apply(self, *args):
        self.applied = True

    def set_input_combinator(self, combinator : Callable):
        '''
        
        Method to define, how the block performs the combination of inputs, passed from "upper" 
        blocks. In most scenarios, this input will be a list of one element (from a single 
        upper block), thus the combinator shall be a (lambda) function to select the first 
        element of the list.
        
        '''
        self.combinator = combinator
    
class EvolutionaryBlock(Block):
    '''
    
    Class, that represents an evolutionary operator, placed into the analogue of the 
    "computational graph" of the iteration of evolutionary algorithm.
    
    Arguments:
    -----------
        operator : epde.operators.template.Compound_Operator
            The evolutionary operator, contained by the block.
    
        parse_operator_args : None, str, or tuple/list
            Approach to getting additional arguments for operator.apply method. If ``None``, 
            no arguments will be obtained, if tuple, then the elements of the tuple (of type 
            ``str``) will be considered as the keys. Other options are 'use inspect' or 'use operator attribute'.
            In the former case, the ``inspect`` library will be used, while in latter the 
            class will try to take the arguments from the operator object.
            
        terminal : bool
            True, if the block is terminal (i.e. final for the EA iteration: no other blocks 
            will be executed after it), otherwise False.
    
    Parameters:
    -----------
        _operator : epde.operators.template.Compound_Operator
            The evolutionary operator, contained by the block
            
        terminal : bool
            The terminality marker of the evolutionary block: if ``True``, than the execution of 
            the LinkedBlocks will terminate after appying the block's operator.
            
    '''
    def __init__(self, operator, parse_operator_args = None, terminal = False): #, initial = False
        self._operator = operator
        if parse_operator_args is None:
            self.arg_keys = []
        elif parse_operator_args == 'inspect':
            import inspect
            self.arg_keys = inspect.getfullargspec(self._operator.apply).args
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
        if self.terminal and len(self._outgoing) > 0:
            raise ValueError('The block is set as the terminal, while it has some output')
        if not self.terminal and len(self._outgoing) == 0:
            raise ValueError('The block is not set as the terminal, while it has no output')
    
    def apply(self, EA_kwargs):
        self.check_integrity()
        kwargs = {kwarg_key : EA_kwargs[kwarg_key] for kwarg_key in self.arg_keys}
        self.output = self._operator.apply(self.combinator([block.output for block in self._incoming]), **kwargs)
        super().apply()

     
class InputBlock(Block):
    def __init__(self, to_pass):
        self.set_output(to_pass); self.applied = True
        super().__init__(initial = True)

    def add_incoming(self, incoming):
        raise TypeError('Objects of this type shall not have incoming links from other operators')

    def set_output(self, to_pass):
        self.output = to_pass
        
    @property
    def available(self):
        return True
        
    
class LinkedBlocks(object):
    '''
    
    The sequence (not necessarily chain: divergencies can be present) of blocks with evolutionary
    operators; it represents the modular and customizable structure of the evolutionary operator.
    
    '''
    def __init__(self, blocks_labeled : dict, suppress_structure_check : bool = False):
        self.blocks_labeled = blocks_labeled
        self.suppress_structure_check = suppress_structure_check
        self.check_correctness()
        self.initial = [(idx, block) for idx, block in enumerate(self.blocks_labeled.values()) if block.initial]
        self.terminated = False
#        print(self.initial)
        if len(self.initial) > 1:
            raise NotImplementedError('The ability to process the multiple initial blocks is not implemented')
    
    def check_correctness(self):
        if not self.suppress_structure_check:        
            if not 'initial' in self.blocks_labeled.keys():
                raise KeyError('Mandatory initial block is missing, or incorrectly labeled')            
            
            if not 'mutation' in self.blocks_labeled.keys():
                raise KeyError('Required evolutionary operator of mutation calculation is missing, or incorrectly labeled')
    
            if not 'crossover' in self.blocks_labeled.keys():
                raise KeyError('Required evolutionary operator of crossover calculation is missing, or incorrectly labeled')


    def reset_traversal_cond(self):
        for block in self.blocks_labeled.values():
            if not isinstance(block, InputBlock): block.applied = False 
            
    def traversal(self, input_obj, EA_kwargs):
        '''
        
        Sequential execution of the evolutionary algorithm's blocks.
        
        '''
        self.reset_traversal_cond()
        
        self.initial[0][1].set_output(input_obj)
        delayed = []; processed = [self.initial[0][1],]
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


class MOEADDSectorProcesser(object):
    def __init__(self):
        self.blocks = dict()
        self._linked_blocks = None
        
    def create_linked_blocks(self, blocks = None, suppress_structure_check = False):
        self.suppress_structure_check = suppress_structure_check
        if self.suppress_structure_check and global_var.verbose.show_warnings:
            warn('The tests of the strategy integrity are suppressed: valuable blocks of EA may go missing and that will not be noticed')
        if blocks is None:
            if len(self.blocks) == 0:
                raise ValueError('Attempted to construct linked blocks object without defined evolutionary blocks')
            self.linked_blocks = LinkedBlocks(self.blocks, suppress_structure_check)
        else:
            if not isinstance(blocks, dict):
                raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
            self.linked_blocks = LinkedBlocks(blocks, suppress_structure_check)
            
    def check_integrity(self):
        if not isinstance(self.blocks, dict):
            raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
        if not isinstance(self.linked_blocks, LinkedBlocks):
            raise TypeError('self.linked_blocks object is not of type LinkedBlocks')
        self.check_correctness()
        
    def iteration(self, population_subset, EA_kwargs : dict):
        if not 'weight' in EA_kwargs:
            raise ValueError('Internal logic error: MOEADD iteration requires weight to process.')
        self.check_integrity()
        self.linked_blocks.blocks_labeled['initial'].set_output(population_subset)
        self.linked_blocks.traversal(EA_kwargs)
        return self.linked_blocks.output

    def check_correctness(self):
        self.linked_blocks.check_correctness()
        
    def apply_block(self, label, operator_kwargs):
        self.linked_blocks.blocks_labeled[label]._operator.apply(**operator_kwargs)
        
    def modify_block_params(self, block_label, param_label, value, suboperator_sequence = None):
        '''
        example call: ``evo_strat.modify_block_params(block_label = 'rps', param_label = 'sparsity', value = some_value, suboperator_sequence = ['fitness_calculation', 'sparsity'])``
        '''
        if suboperator_sequence is None:
            self.linked_blocks.blocks_labeled[block_label]._operator.params[param_label] = value
        else:
            if isinstance(suboperator_sequence, str):
                if suboperator_sequence not in self.linked_blocks.blocks_labeled[block_label]._operator.suboperators.keys():
                    err_message = f'Suboperator for the control sequence {suboperator_sequence} is missing'
                    raise KeyError(err_message)
                temp = self.linked_blocks.blocks_labeled[block_label]._operator.suboperators[suboperator_sequence]
            elif isinstance(suboperator_sequence, (tuple, list)):
                temp = self.linked_blocks.blocks_labeled[block_label]._operator
                for suboperator in suboperator_sequence:
                    if suboperator not in temp.suboperators.keys():
                        err_message = f'Suboperator for the control sequence {suboperator} is missing'
                        raise KeyError(err_message)
                    temp = temp.suboperators[suboperator]

            else:
                raise TypeError('Incorrect type of suboperator sequence. Need str or list/tuple of str')
            temp.params[param_label] = value