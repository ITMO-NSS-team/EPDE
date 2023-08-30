#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:46:03 2023

@author: maslyaev
"""

from warnings import warn

import epde.globals as global_var
from epde.optimizers.blocks import LinkedBlocks


class Strategy():
    """
    Class instance of base for create evolution strategy

    Attributes:
        suppress_structure_check (`boolean`): flag about checking of structure
        blocks (`dict`): ductuonary with operators for the strategy
        linked_blocks (`LinkedBlocks`): the sequence (not necessarily chain: divergencies can be present) of blocks with evolutionary operators
        run_performed (`boolean`): flag, that strategy was running (exclude the possibility of taking a result from a strategy that has not been applied)        
    """
    def __init__(self):
        self._blocks = dict()
        self._linked_blocks = None
        
    def create_linked_blocks(self, blocks = None, suppress_structure_check = False):
        """
        Creating sequence of `blocks` with evolution operators

        Args:
            blocks (`dict`): dictionary with names and values of the evolution operator
            suppress_structure_check (`boolean`): flag about checking of structure
        
        Returns:
            None
        """
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
        """
        Method for checking type on blocks of evolutional operators
        """
        if not isinstance(self.blocks, dict):
            raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
        if not isinstance(self.linked_blocks, LinkedBlocks):
            raise TypeError('self.linked_blocks object is not of type LinkedBlocks')
            
    def apply_block(self, label, operator_kwargs):
        """
        Running process of choosing block evolutionary operator

        Args:
            label (`str`): name of evolutional operator
            operator_kwargs (`dict`): dictionary with argumenta of EA, where keys are names and items are values of the arguments

        Returns:
            None
        """
        self.linked_blocks.blocks_labeled[label]._operator.apply(**operator_kwargs)
            
    @property
    def result(self):
        if not self.run_performed:
            raise ValueError('Trying to get the output of the strategy before running it.')
        return self.linked_blocks.output