#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 19:14:25 2022

@author: maslyaev
"""

from warnings import warn
from typing import Callable
from abc import ABC, abstractproperty

import numpy as np

import epde.globals as global_var


def link(op1, op2):
    '''

    Set the connection of operators in format op1 -> op2

    '''
    assert isinstance(op1, (EvolutionaryBlock, InputBlock)) and isinstance(op2, (EvolutionaryBlock, InputBlock)), 'An operator is not defined as the Placed operator object'
    op1.add_outgoing(op2)
    op2.add_incoming(op1)


class OperatorBuilder(ABC):    
    @abstractproperty
    def processer(self):
        pass


class SectorProcesserBuilder(OperatorBuilder):
    """
    Class of sector process builder for moeadd. 
    
    Attributes:
    ------------
    
    operator : Evolutionary_operator object
        the evolutionary operator, which is being constructed for the evolutionary algorithm, which is applied to a population;
        
    Methods:
    ------------
    
    reset()
        Reset the evolutionary operator, deleting all of the declared suboperators.
        
    set_evolution(crossover_op, mutation_op)
        Set crossover and mutation operators with corresponding evolutionary operators, each of the Specific_Operator type object, to improve the 
        quality of the population.
    
    set_param_optimization(param_optimizer)
        Set parameter optimizer with pre-defined Specific_Operator type object to optimize the parameters of the factors, present in the equation.
        
    set_coeff_calculator(coef_calculator)
        Set coefficient calculator with Specific_Operator type object, which determines the weights of the terms in the equations.
        
    set_fitness(fitness_estim)
        Set fitness function value estimator with the Specific_Operator type object. 
    
    """
    
    def __init__(self):  # , stop_criterion, stop_criterion_kwargs
        self.reset() # stop_criterion, stop_criterion_kwargs
    
    def reset(self): # stop_criterion, stop_criterion_kwargs
        self._processer = MOEADDSectorProcesser() # stop_criterion, stop_criterion_kwargs
        self.blocks_labeled = dict()
        self.blocks_connected = dict() # dict of format {op_label : (True, True)}, where in 
                                       # value dict first element is "connected with input" and
                                       # the second - "connected with output"
        self.reachable = dict()
        self.initial_label = None
    
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
    
    @property
    def processer(self):
        self._processer.check_correctness()
        return self._processer


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
            self._id = np.random.randint(0, 1e6)
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
        if self.terminal and len(self._outgoing) > 0:
            raise ValueError('The block is set as the terminal, while it has some output')
        if not self.terminal and len(self._outgoing) == 0:
            raise ValueError('The block is not set as the terminal, while it has no output')
    
    def apply(self, EA_kwargs):
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
        if len(self.initial) > 1:
            raise NotImplementedError('The ability to process the multiple initial blocks is not implemented')
    
    def check_correctness(self):
        return True
        # if not self.suppress_structure_check:        
        #     if not 'initial' in self.blocks_labeled.keys():
        #         raise KeyError('Mandatory initial block is missing, or incorrectly labeled')            
            
        #     if not 'mutation' in self.blocks_labeled.keys():
        #         raise KeyError('Required evolutionary operator of mutation calculation is missing, or incorrectly labeled')
    
        #     if not 'crossover' in self.blocks_labeled.keys():
        #         raise KeyError('Required evolutionary operator of crossover calculation is missing, or incorrectly labeled')


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
                    # print(f'applying {vertex}')                    
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
        
    def run(self, population_subset, EA_kwargs : dict):
        self.check_integrity()
        self.linked_blocks.traversal(population_subset, EA_kwargs)
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