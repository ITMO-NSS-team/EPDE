#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:04:58 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Iterable
from warnings import warn

import epde.globals as global_var
from epde.operators.template import Compound_Operator

from epde.operators.equation_selections import Tournament_selection
from epde.operators.equation_elitism import Fraction_elitism
from epde.operators.equation_mutations import PopLevel_mutation, PopLevel_mutation_elite, Refining_Equation_mutation, Equation_mutation, Term_mutation, Parameter_mutation
from epde.operators.equation_crossovers import PopLevel_crossover, Equation_crossover, Param_crossover, Term_crossover
from epde.operators.equation_sparsity import LASSO_sparsity
from epde.operators.equation_coeffcalc import LinReg_based_coeffs
from epde.operators.equation_fitness import L2_fitness, Solver_based_fitness
from epde.operators.equation_right_part_selection import Poplevel_Right_Part_Selector, Eq_Right_Part_Selector, Status_respecting_ERPS
from epde.operators.equation_truncate import Truncate_worst

class Operator_builder(ABC):    
    
    @abstractproperty
    def strategy(self):
        pass

class Strategy_builder(Operator_builder):
    """
    Class of evolutionary operator builder. 
    
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
    
    def __init__(self, stop_criterion, stop_criterion_kwargs):
        self.reset(stop_criterion, stop_criterion_kwargs)
    
    def reset(self, stop_criterion, stop_criterion_kwargs):
        self._strategy = Evolutionary_strategy(stop_criterion, stop_criterion_kwargs)
#        self.operators = []
        self.blocks_labeled = dict()
        self.blocks_connected = dict() # dict of format {op_label : (True, True)}, where in 
                                    # value dict first element is "connected with input" and
                                    # the second - "connected with output"
        self.reachable = dict()
        self.initial_label = None
    
    def add_init_operator(self, operator_label : str = 'initial'):
        self.initial_label = operator_label
        new_block = Input_block(to_pass = None)
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

    def add_operator(self, operator_label, operator, parse_operator_args = None, 
                     terminal_operator : bool = False): #, input_operator : bool = False
#        if input_operator: 
#        else:
        new_block = Evolutionary_block(operator, parse_operator_args = parse_operator_args,
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
        self._strategy.create_linked_blocks(blocks = self.blocks_labeled, suppress_structure_check = suppress_structure_check)
    
    @property
    def strategy(self):
        self._strategy.check_correctness()
        return self._strategy
        #self.reset()
#        return strategy


class Strategy_director(object):    
    def __init__(self, stop_criterion, stop_criterion_kwargs): # baseline = True
#        self._constructor = None
#        if baseline:
        self._constructor = Strategy_builder(stop_criterion, stop_criterion_kwargs)
    
    @property
    def constructor(self):
        return self._constructor
    
    @constructor.setter
    def constructor(self, constructor):
        self._constructor = constructor

    def strategy_assembly(self, **kwargs):
        selection = Tournament_selection(['part_with_offsprings', 'tournament_groups'])
        selection.params = {'part_with_offsprings' : 0.2, 'tournament_groups' : 2} if not 'selection_params' in kwargs.keys() else  kwargs['selection_params']

        elitism = Fraction_elitism(['elite_fraction'])
        elitism.params = {'elite_fraction' : 0.2}

        truncation = Truncate_worst(['population_size'])
        truncation.params = {'population_size' : None}

        param_mutation = Parameter_mutation(['r_param_mutation', 'strict_restrictions', 'multiplier'])
        term_mutation = Term_mutation(['forbidden_tokens'])
        eq_mutation = Equation_mutation(['r_mutation', 'type_probabilities'])
        ref_eq_mutation = Refining_Equation_mutation(['r_mutation', 'type_probabilities'])
        mutation = PopLevel_mutation_elite(['indiv_mutation_prob'])

        param_mutation.params = {'r_param_mutation' : 0.2, 'strict_restrictions' : True, 'multiplier' : 0.1} if not 'param_mutation_params' in kwargs.keys() else kwargs['param_mutation_params']
        term_mutation.params = {'forbidden_tokens': []} if not 'term_mutation_params' in kwargs.keys() else kwargs['term_mutation_params']
        eq_mutation.params = {'r_mutation' : 0.3, 'type_probabilities' : []}
        ref_eq_mutation.params = {'r_mutation' : 0.3, 'type_probabilities' : []}
        mutation.params = {'indiv_mutation_prob' : 0.5} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']

        ref_eq_mutation.suboperators = {'Mutation' : [param_mutation, term_mutation]}
        eq_mutation.suboperators = {'Mutation' : [param_mutation, term_mutation]}
        mutation.suboperators = {'Equatiion_mutation' : {'elite' : ref_eq_mutation, 'non-elite' : eq_mutation}}

        param_crossover = Param_crossover(['proportion'])
        term_crossover = Term_crossover(['crossover_probability'])
        eq_crossover = Equation_crossover([])
        crossover = PopLevel_crossover([])        

        param_crossover.params = {'proportion' : 0.4} if not 'param_crossover_params' in kwargs.keys() else kwargs['param_crossover_params']
        term_crossover.params = {'crossover_probability' : 0.3} if not 'term_crossover_params' in kwargs.keys() else kwargs['term_crossover_params']
        eq_crossover.params = {} if not 'eq_crossover_params' in kwargs.keys() else kwargs['eq_crossover_params']
        crossover.params = {} if not 'crossover_params' in kwargs.keys() else kwargs['crossover_params']

        eq_crossover.suboperators = {'Param_crossover' : param_crossover, 'Term_crossover' : term_crossover} 
        crossover.suboperators = {'Equation_crossover' : eq_crossover}
        
        lasso_coeffs = LASSO_sparsity(['sparsity'])
        linreg_coeffs = LinReg_based_coeffs([])
        
        lasso_coeffs.params = {'sparsity' : 1} if not 'lasso_coeffs_params' in kwargs.keys() else kwargs['lasso_coeffs_params']
        linreg_coeffs.params = {} if not 'linreg_coeffs_params' in kwargs.keys() else kwargs['linreg_coeffs_params']

        fitness_eval = L2_fitness(['penalty_coeff'])
        fitness_eval.suboperators = {'sparsity' : lasso_coeffs, 'coeff_calc' : linreg_coeffs}
        fitness_eval.params = {'penalty_coeff' : 0.5} if not 'fitness_eval_params' in kwargs.keys() else kwargs['fitness_eval_params']
        
        rps1 = Poplevel_Right_Part_Selector([])
        rps1.params = {} if not 'rps_params' in kwargs.keys() else kwargs['rps_params']
        
        eq_rps = Status_respecting_ERPS([])
        eq_rps.suboperators = {'fitness_calculation' : fitness_eval}
        eq_rps.params = {} if not 'eq_rps_params' in kwargs.keys() else kwargs['eq_rps_params']
        
        rps1.suboperators = {'eq_level_rps' : eq_rps}
        
        rps2 = deepcopy(rps1)
        
        self._constructor.add_init_operator('initial')

        self._constructor.add_operator('rps1', rps1, parse_operator_args = 'inspect', 
                                       terminal_operator = False)

        self._constructor.add_operator('selection', selection, parse_operator_args = 'inspect', 
                                       terminal_operator = False)
        
        self._constructor.add_operator('crossover', crossover, parse_operator_args = 'inspect', 
                                       terminal_operator = False)

        self._constructor.add_operator('elitism', elitism, parse_operator_args = 'inspect', 
                                       terminal_operator = False)
        
        self._constructor.add_operator('mutation', mutation, parse_operator_args = 'inspect', 
                                       terminal_operator = False)        

        self._constructor.add_operator('rps2', rps2, parse_operator_args = 'inspect', 
                                       terminal_operator = False)        
        
        self._constructor.add_operator('truncation', truncation, parse_operator_args = 'inspect', 
                                       terminal_operator = True)        
        
        self._constructor.set_input_combinator()
        
        self._constructor.link('initial', 'rps1')
        self._constructor.link('rps1', 'selection')
        self._constructor.link('selection', 'crossover')
        self._constructor.link('crossover', 'elitism')
        self._constructor.link('elitism', 'mutation')
        self._constructor.link('mutation', 'rps2')
        self._constructor.link('rps2', 'truncation')
        
        self._constructor.assemble()
        
    
class Strategy_director_solver(object):    
    def __init__(self, stop_criterion, stop_criterion_kwargs): # baseline = True
#        self._constructor = None
#        if baseline:
        self._constructor = Strategy_builder(stop_criterion, stop_criterion_kwargs)
    
    @property
    def constructor(self):
        return self._constructor
    
    @constructor.setter
    def constructor(self, constructor):
        self._constructor = constructor

    def strategy_assembly(self, **kwargs):
        selection = Tournament_selection(['part_with_offsprings', 'tournament_groups'])
        selection.params = {'part_with_offsprings' : 0.2, 'tournament_groups' : 2} if not 'selection_params' in kwargs.keys() else  kwargs['selection_params']

        elitism = Fraction_elitism(['elite_fraction'])
        elitism.params = {'elite_fraction' : 0.2}

        truncation = Truncate_worst(['population_size'])
        truncation.params = {'population_size' : None}

        param_mutation = Parameter_mutation(['r_param_mutation', 'strict_restrictions', 'multiplier'])
        term_mutation = Term_mutation(['forbidden_tokens'])
        eq_mutation = Equation_mutation(['r_mutation', 'type_probabilities'])
        ref_eq_mutation = Refining_Equation_mutation(['r_mutation', 'type_probabilities'])
#        mutation = PopLevel_mutation(['indiv_mutation_prob', 'elitism'])
        mutation = PopLevel_mutation_elite(['indiv_mutation_prob'])

        param_mutation.params = {'r_param_mutation' : 0.2, 'strict_restrictions' : True, 'multiplier' : 0.1} if not 'param_mutation_params' in kwargs.keys() else kwargs['param_mutation_params']
        term_mutation.params = {'forbidden_tokens': []} if not 'term_mutation_params' in kwargs.keys() else kwargs['term_mutation_params']
        eq_mutation.params = {'r_mutation' : 0.3, 'type_probabilities' : []}
        ref_eq_mutation.params = {'r_mutation' : 0.3, 'type_probabilities' : []}
        mutation.params = {'indiv_mutation_prob' : 0.5} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']
        
        ref_eq_mutation.suboperators = {'Mutation' : [param_mutation, term_mutation]}
        eq_mutation.suboperators = {'Mutation' : [param_mutation, term_mutation]}
        mutation.suboperators = {'Equatiion_mutation' : {'elite' : ref_eq_mutation, 'non-elite' : eq_mutation}}


        param_crossover = Param_crossover(['proportion'])
        term_crossover = Term_crossover(['crossover_probability'])
        eq_crossover = Equation_crossover([])
        crossover = PopLevel_crossover([])        

        param_crossover.params = {'proportion' : 0.4} if not 'param_crossover_params' in kwargs.keys() else kwargs['param_crossover_params']
        term_crossover.params = {'crossover_probability' : 0.3} if not 'term_crossover_params' in kwargs.keys() else kwargs['term_crossover_params']
        eq_crossover.params = {} if not 'eq_crossover_params' in kwargs.keys() else kwargs['eq_crossover_params']
        crossover.params = {} if not 'crossover_params' in kwargs.keys() else kwargs['crossover_params']

        eq_crossover.suboperators = {'Param_crossover' : param_crossover, 'Term_crossover' : term_crossover} 
        crossover.suboperators = {'Equation_crossover' : eq_crossover}
        
        lasso_coeffs = LASSO_sparsity(['sparsity'])
        linreg_coeffs = LinReg_based_coeffs([])
        
        lasso_coeffs.params = {'sparsity' : 1} if not 'lasso_coeffs_params' in kwargs.keys() else kwargs['lasso_coeffs_params']
        linreg_coeffs.params = {} if not 'linreg_coeffs_params' in kwargs.keys() else kwargs['linreg_coeffs_params']

        fitness_eval = Solver_based_fitness(['lambda_bound', 'learning_rate', 'eps', 'tmin', 'tmax', 'verbose'])
        fitness_eval.suboperators = {'sparsity' : lasso_coeffs, 'coeff_calc' : linreg_coeffs}
        fitness_eval.params = {'lambda_bound' : 1000, 'learning_rate' : 1e-5, 
                               'eps' : 1e-6, 'tmin' : 1000, 'tmax' : 1e5, 'verbose' : False} if not 'fitness_eval_params' in kwargs.keys() else kwargs['fitness_eval_params']
        
        rps1 = Poplevel_Right_Part_Selector([])
        rps1.params = {} if not 'rps_params' in kwargs.keys() else kwargs['rps_params']

        eq_rps = Status_respecting_ERPS([])
        eq_rps.suboperators = {'fitness_calculation' : fitness_eval}
        eq_rps.params = {} if not 'eq_rps_params' in kwargs.keys() else kwargs['eq_rps_params']
        
        rps1.suboperators = {'eq_level_rps' : eq_rps}
        
        rps2 = deepcopy(rps1)
        
        self._constructor.add_init_operator('initial')

        self._constructor.add_operator('rps1', rps1, parse_operator_args = 'inspect', 
                                       terminal_operator = False)

        self._constructor.add_operator('selection', selection, parse_operator_args = 'inspect', 
                                       terminal_operator = False)
        
        self._constructor.add_operator('crossover', crossover, parse_operator_args = 'inspect', 
                                       terminal_operator = False)

        self._constructor.add_operator('elitism', elitism, parse_operator_args = 'inspect', 
                                       terminal_operator = False)
        
        self._constructor.add_operator('mutation', mutation, parse_operator_args = 'inspect', 
                                       terminal_operator = False)        

        self._constructor.add_operator('rps2', rps2, parse_operator_args = 'inspect', 
                                       terminal_operator = False)        
        
        self._constructor.add_operator('truncation', truncation, parse_operator_args = 'inspect', 
                                       terminal_operator = True)                
        
        self._constructor.set_input_combinator()
        
        self._constructor.link('initial', 'rps1')
        self._constructor.link('rps1', 'selection')
        self._constructor.link('selection', 'crossover')
        self._constructor.link('crossover', 'elitism')
        self._constructor.link('elitism', 'mutation')
        self._constructor.link('mutation', 'rps2')
        self._constructor.link('rps2', 'truncation')
        
        self._constructor.assemble()
        
        
def link(op1, op2):
    '''
    
    Set the connection of operators in format op1 -> op2
    
    '''
    assert isinstance(op1, (Evolutionary_block, Input_block)) and isinstance(op2, (Evolutionary_block, Input_block)), 'An operator is not defined as the Placed operator object'
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
    
class Evolutionary_block(Block):
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
            the Linked_Blocks will terminate after appying the block's operator.
            
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
#        print('Input: from combinator:', self.combinator([block.output for block in self._incoming]), 'parents output:', [block.output for block in self._incoming])
        self.output = self._operator.apply(self.combinator([block.output for block in self._incoming]), **kwargs)
#        print('output:', self.output)        
        super().apply()

     
class Input_block(Block):
    def __init__(self, to_pass):
        self.set_output(to_pass); self.applied = True
        super().__init__(initial = True)
#        print(self.available)

    def add_incoming(self, incoming):
        raise TypeError('Objects of this type shall not have incoming links from other operators')

    def set_output(self, to_pass):
        self.output = to_pass
        
    @property
    def available(self):
        return True
        
    
class Linked_Blocks(object):
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
            if not isinstance(block, Input_block): block.applied = False 
            
    def traversal(self, input_obj, EA_kwargs):
        '''
        
        Sequential execution of the evolutionary algorithm's blocks.
        
        '''
        self.reset_traversal_cond()
        
#        import time
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
    
    
class Evolutionary_strategy(object):
    '''

    Class, containing the evolutionary strategyfor evolutionary algorithm of equation 
    discovery: the 
    
    '''
    def __init__(self, stop_criterion, sc_init_kwargs):
        self._stop_criterion = stop_criterion(**sc_init_kwargs)
        self.blocks = dict()
        self._linked_blocks = None
        self.run_performed = False
        
    def create_linked_blocks(self, blocks = None, suppress_structure_check = False):
        self.suppress_structure_check = suppress_structure_check
        if self.suppress_structure_check and global_var.verbose.show_warnings:
            warn('The tests of the strategy integrity are suppressed: valuable blocks of EA may go missing and that will not be noticed')
        if blocks is None:
            if len(self.blocks) == 0:
                raise ValueError('Attempted to construct linked blocks object without defined evolutionary blocks')
            self.linked_blocks = Linked_Blocks(self.blocks, suppress_structure_check)
        else:
            if not isinstance(blocks, dict):
                raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
            self.linked_blocks = Linked_Blocks(blocks, suppress_structure_check)
            
    def check_integrity(self):
        if not isinstance(self.blocks, dict):
            raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
        if not isinstance(self.linked_blocks, Linked_Blocks):
            raise TypeError('self.linked_blocks object is not of type Linked_Blocks')
        self.check_correctness()
        
    def iteration(self, population_subset, EA_kwargs = None):
        self.check_integrity()
        self.linked_blocks.blocks_labeled['initial'].set_output(population_subset)
        self.linked_blocks.traversal(EA_kwargs)
        return self.linked_blocks.output
    
    def run(self, initial_population : Iterable, EA_kwargs : dict):
        self._stop_criterion.reset()
        population = initial_population
        iter_idx = 0
        while not self._stop_criterion.check():
            iter_idx += 1; log_message = ''
            if global_var.verbose.iter_idx:
                log_message += f'Equation search epoch {iter_idx}.'
            if global_var.verbose.iter_fitness:
                log_message += f'Achieved fitness of {np.max([equation.fitness_value for equation in population])}'
            if global_var.verbose.iter_stats:
                raise NotImplementedError('Evolutionary optimizer statistics output not yet implemented')
            self.linked_blocks.traversal(population, EA_kwargs)
            if log_message:
                print(log_message)
            population = self.linked_blocks.output
        self.run_performed = True
        
    def check_correctness(self):
        self.linked_blocks.check_correctness()        
        
    @property
    def result(self):
        if not self.run_performed:
            raise ValueError('Trying to get the output of the strategy before running it.')
        return self.linked_blocks.output
        
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

        
    def get_fitness(self, solution, return_res = False):
        raise NotImplementedError
        if not 'fitness' in self.linked_blocks.blocks_labeled.keys():
            raise KeyError('Required evolutionary operator of fitness calculation is missing, or incorrectly labeled')
        self.linked_blocks.blocks_labeled['fitness']._operator.apply(solution)
        if return_res:
            return solution.fitness_value
