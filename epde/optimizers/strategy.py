#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:46:03 2023

@author: maslyaev
"""

from warnings import warn

import epde.globals as global_var
from epde.optimizers.blocks import link, LinkedBlocks, EvolutionaryBlock, InputBlock, OperatorBuilder


class Strategy():
    def __init__(self):
        self._blocks = dict()
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
            
    @property
    def result(self):
        if not self.run_performed:
            raise ValueError('Trying to get the output of the strategy before running it.')
        return self.linked_blocks.output