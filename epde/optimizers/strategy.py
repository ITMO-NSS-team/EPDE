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
    Base class for defining evolutionary strategies. It provides a foundation for implementing custom search algorithms for optimization problems. Subclasses should implement specific evolutionary operators and selection mechanisms to guide the search process.
    
    
        Attributes:
            suppress_structure_check (`boolean`): flag about checking of structure
            blocks (`dict`): dictionary with operators for the strategy
            linked_blocks (`LinkedBlocks`): the sequence (not necessarily chain: divergencies can be present)
                                            of blocks with evolutionary operators
            run_performed (`boolean`): flag, that strategy was running (exclude the possibility of taking a 
                                       result from a strategy that has not been applied)
    """

    def __init__(self):
        """
        Initializes a new instance of the Strategy class.
        
                This method sets up the internal data structures required for managing
                the search process. It initializes the block storage for equation components,
                flags for controlling structure checks, and run status.
        
                Args:
                    self: The object instance.
        
                Fields:
                    _blocks (dict): A dictionary to store blocks, where keys are block identifiers and values are block objects.
                    _linked_blocks (None): Placeholder for linked block information, initially set to None.
                    suppress_structure_check (bool): A flag to suppress structure checks, initially set to True.
                    run_performed (bool): A flag indicating whether a run has been performed, initially set to False.
        
                Returns:
                    None.
        
                Why:
                    This initialization ensures that the search strategy starts with a clean slate,
                    ready to explore the space of possible equation structures. The flags allow
                    for controlling the level of validation during the search, and the block
                    storage will hold the building blocks of the discovered equations.
        """
        self._blocks = dict()
        self._linked_blocks = None
        self.suppress_structure_check = True
        self.run_performed = False
        # self.best_objectives = [0.,]
        # best_objectives (`float`): best objectives for each criteria, employed in optimization.        
        
    def create_linked_blocks(self, blocks = None, suppress_structure_check = False):
        """
        Creates a sequence of linked evolutionary algorithm building blocks.
        
                This method arranges the provided `blocks` into a sequential structure,
                preparing them for execution within the evolutionary algorithm. It also
                handles checks for structural integrity (unless suppressed) to ensure
                the proper functioning of the evolutionary strategy.
        
                Args:
                    blocks (`dict`, optional): A dictionary where keys are symbolic names
                        (e.g., "crossover", "mutation") and values are the corresponding
                        evolutionary operator blocks. If `None`, the blocks defined within
                        the strategy are used.
                    suppress_structure_check (`bool`, optional): A flag to disable structural
                        integrity checks.  Defaults to `False`. Disabling this is not
                        recommended, as it can lead to unexpected behavior if the blocks
                        are not correctly configured.
        
                Returns:
                    None
        
                Raises:
                    ValueError: If `blocks` is `None` and the strategy has no defined blocks.
                    TypeError: If `blocks` is not a dictionary.
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
        Verifies the structural integrity of the evolutionary operator blocks.
        
                This method ensures that the blocks of evolutionary operators are correctly 
                structured and of the expected types before the evolutionary process begins. 
                It validates that the `blocks` attribute is a dictionary and the `linked_blocks` 
                attribute is a `LinkedBlocks` object, raising a TypeError if either condition is not met.
                This is crucial for the correct functioning of the evolutionary algorithm in discovering
                differential equations.
        
                Args:
                    self: The Strategy instance.
        
                Returns:
                    None
        
                Raises:
                    TypeError: If `self.blocks` is not a dictionary or `self.linked_blocks` is not a `LinkedBlocks` instance.
        """
        if not isinstance(self.blocks, dict):
            raise TypeError('Blocks object must be of type dict with key - symbolic name of the block (e.g. "crossover", or "mutation", and value of type block')
        if not isinstance(self.linked_blocks, LinkedBlocks):
            raise TypeError('self.linked_blocks object is not of type LinkedBlocks')
            
    def apply_block(self, label, operator_kwargs):
        """
        Applies a specific evolutionary operator to a block within the equation discovery strategy.
        
        This method retrieves a labeled block and executes its associated evolutionary operator with the provided arguments.
        This step is crucial for evolving the structure and parameters of the equation being discovered.
        
        Args:
            label (`str`): The label of the block to which the operator will be applied.
            operator_kwargs (`dict`): Keyword arguments to be passed to the evolutionary operator's `apply` method.
        
        Returns:
            None
        """
        self.linked_blocks.blocks_labeled[label]._operator.apply(**operator_kwargs)
            
    @property
    def result(self):
        """
        Return the final result produced by the strategy after the computation.
        
                This property provides access to the processed output, ensuring that the strategy has been successfully executed. It acts as the endpoint for retrieving the derived insights or predictions.
        
                Args:
                    None
        
                Returns:
                    The output of the linked blocks, representing the final result of the strategy.
        
                Raises:
                    ValueError: If the strategy has not been run yet, indicating that the output is not yet available.
        """
        if not self.run_performed:
            raise ValueError('Trying to get the output of the strategy before running it.')
        return self.linked_blocks.output
