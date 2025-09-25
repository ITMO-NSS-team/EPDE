#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:15:31 2023

@author: maslyaev
"""

from abc import ABC, abstractmethod

from epde.optimizers.blocks import EvolutionaryBlock, InputBlock
from epde.optimizers.strategy import Strategy

def link(op1, op2):
    '''
    Set the connection between two computational blocks, defining the data flow direction.
    
    This method establishes a directed connection from `op1` to `op2`, indicating that the output of `op1` serves as an input to `op2`.
    This ensures proper data flow and dependency management within the computational graph.
    
    Args:
        op1 (EvolutionaryBlock or InputBlock): The source computational block.
        op2 (EvolutionaryBlock or InputBlock): The destination computational block.
    
    Returns:
        None
    '''
    assert isinstance(op1, (EvolutionaryBlock, InputBlock)) and isinstance(op2, (EvolutionaryBlock, InputBlock)), 'An operator is not defined as the Placed operator object'
    op1.add_outgoing(op2)
    op2.add_incoming(op1)


class StrategyBuilder():
    """
    Class for constructing an equation discovery strategy.
    
    
        Attributes:
            processer (`Strategy`): class strategy, that is used in this algorithm
            blocks_labeled (`dict`): dictionary with operator blocks in evolutionary algorithm 
            blocks_connected (`dict`): dictionary with names of operators and list with boolean flags about "neighbours" (some operator before and some operator after)
            reachable (`dict`): dictionary with operators, that definitely not a single node: key - operator's name, item - block's id
            initial_label (`str`): name of operator, that is executed first in the queue of operators
    """

    def __init__(self, strategy_class = Strategy):
        """
        Initializes the StrategyBuilder with a specific strategy.
        
        This constructor prepares the builder to create and manage solving strategies.
        It sets the strategy class to be used and immediately resets the builder
        to ensure a clean initial state with the provided strategy.
        
        Args:
            strategy_class (class): The strategy class to use for solving. Defaults to Strategy.
        
        Returns:
            None
        
        Class Fields:
            strategy_class (class): The class used to instantiate the solving strategy.
            strategy (object): An instance of the `strategy_class`, used for solving.
        
        The StrategyBuilder needs to know which strategy to use, so it can later produce specific solver instances.
        The reset is called to ensure that the builder is in a consistent state before any solving begins.
        """
        self.strategy_class = strategy_class
        self.reset(strategy_class)
    
    def reset(self, strategy_class):
        """
        Resets the internal state of the strategy builder with a new optimization strategy.
        
        This prepares the builder for constructing a new equation by clearing any previously
        stored information about the equation's structure, such as labeled blocks,
        connectivity, and reachability. This ensures a clean slate for the next equation
        discovery process.
        
        Args:
            strategy_class (`Strategy`): The class defining the optimization strategy to be used.
        
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
    
    @property
    def processer(self):
        """
        Provides access to the data processing component.
        
        This property allows retrieval of the configured data processor,
        which is responsible for preparing the input data for the equation discovery process.
        It is essential for ensuring that the data is in the correct format and scale
        before being used to train and validate candidate equation models.
        
        Args:
            None
        
        Returns:
            The configured data processing component.
        """
        return self._processer

    def add_init_operator(self, operator_label : str = 'initial'):
        """
        Adds an initial operator to the strategy, marking the starting point for the evolutionary search.
        
                This method designates a specific operator as the starting point for the optimization process. It initializes the necessary data structures to track reachable blocks and their connections, ensuring the evolutionary algorithm begins exploring the search space from the defined origin.
        
                Args:
                    operator_label (str):  A unique identifier for the initial operator. Defaults to 'initial'.
        
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
        Sets the method for combining inputs for each processing block.
        
        This method configures how each block within the strategy combines its inputs. 
        If a custom combination method is not provided for a specific block, a default method 
        that selects the first input is used. This ensures that each block has a defined 
        input combination strategy, which is crucial for the overall data processing pipeline 
        within the equation discovery process.
        
        Args:
            non_default (dict): A dictionary where keys are block names and values are callable 
                methods that define how to combine inputs for the corresponding block. If a block's 
                name is not present in this dictionary, a default input combination method is used.
        
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
        Adds a new evolutionary operator to the strategy.
        
                This method registers a new operator within the strategy builder, associating it with a label and defining its behavior within the evolutionary process. The operator is encapsulated within an `EvolutionaryBlock`, which manages its execution and argument parsing. The `terminal_operator` flag determines whether applying this operator concludes a sequence of operations. This registration allows the strategy to utilize the operator during the evolutionary search for optimal equation structures.
        
                Args:
                    operator_label (`str`): A unique name for the operator.
                    operator (`Block`): The operator itself, an instance of `epde.operators.utils.template.Compound_Operator`.
                    parse_operator_args (`str`): Specifies how arguments for the operator's `apply` method should be parsed. Options include 'inspect' (using the `inspect` library), 'use operator attribute' (retrieving arguments from the operator's `arg_keys` attribute), or a list/tuple of argument names. Defaults to 'inspect'.
                    terminal_operator (`bool`): Indicates whether applying this operator should terminate the current sequence of operations. Defaults to `False`.
        
                Returns:
                    None
        """
        new_block = EvolutionaryBlock(operator, parse_operator_args = parse_operator_args,
                                                   terminal=terminal_operator)
        self.blocks_labeled[operator_label] = new_block
        self.reachable[operator_label] = {new_block.op_id,}
        self.blocks_connected[operator_label] = [False, terminal_operator]

    def link(self, label_out, label_in):
        """
        Connects two operators, directing the output of one to the input of the other.
        
        This establishes a data flow dependency between the specified operators, 
        allowing the system to chain operations and build more complex processing pipelines.
        This is a crucial step in constructing a computational graph where the output of one 
        operator serves as the input for another, enabling the system to perform sequential 
        data transformations and analysis.
        
        Args:
            label_out (str): The label of the operator whose output will be used as input.
            label_in (str): The label of the operator that will receive the output as input.
        
        Returns:
            None
        """
        link(self.blocks_labeled[label_out], self.blocks_labeled[label_in])
        self.reachable[label_out].union(self.reachable[label_in])
        self.blocks_connected[label_out][1] = True
        self.blocks_connected[label_in][0] = True
    
    def check_connectedness(self):
        """
        Checks if all labeled blocks are reachable from the initial block.
        
        This ensures that the discovered equation structure is based on a cohesive and interconnected set of relationships within the data.
        
        Args:
            self: The StrategyBuilder instance.
        
        Returns:
            bool: True if the number of labeled blocks equals the number of blocks reachable from the initial block, False otherwise.
        """
        return len(self.blocks_labeled) == len(self.reachable[self.initial_label])
    
    def assemble(self, suppress_structure_check = False):
        """
        Assembles the linked blocks into a cohesive structure using the configured processor.
        
                This process prepares the discovered equation components for further analysis and refinement.
        
                Args:
                    suppress_structure_check: Whether to bypass structural validation during linked block creation.
        
                Returns:
                    None.
        """
        self._processer.create_linked_blocks(blocks = self.blocks_labeled, suppress_structure_check = suppress_structure_check)

    
def add_sequential_operators(builder : StrategyBuilder, operators : list):
    """
    Adds a sequence of operators to the strategy builder, linking them together to form a processing pipeline.
    
        This method streamlines the construction of an evolutionary strategy by sequentially adding and connecting operators. It initializes the pipeline with a starting operator, then iterates through the provided operators, adding each one and linking it to the previous one. Finally, it assembles the complete pipeline.
    
        Args:
            builder (`StrategyBuilder`): The strategy builder to which the operators will be added.
            operators (`list`): A list of operators to be added sequentially. Each element should be a tuple containing a label (str) and an operator (a subclass of `CompoundOperator`).
    
        Returns:
            `StrategyBuilder`: The modified strategy builder with the added and linked operators.
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
    Base class for directing the optimization process, defining the overall strategy and workflow.
    
    
        Attributes:
            builder (`StrategyBuilder`)
    """

    def __init__(self):
        """
        Initializes the OptimizationPatternDirector.
        
        This director orchestrates the construction of optimization patterns
        by utilizing a builder object. The initialization process sets up
        the director with a clean slate, ready to receive a builder and
        begin the pattern construction.
        
        Args:
            self: The OptimizationPatternDirector instance.
        
        Returns:
            None.
        
        Why:
            This initialization ensures that the director is ready to
            construct a new optimization pattern. The director needs to be
            initialized before it can work with a builder to create a
            specific optimization strategy.
        """
        self._builder = None

    @property
    def builder(self):
        """
        Returns the builder associated with this object.
                The builder is responsible for constructing the optimization pattern.
        
                Returns:
                    The builder object.
        
                Why:
                    The builder encapsulates the logic for creating complex optimization patterns,
                    allowing for a modular and configurable approach to pattern construction.
        """
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : StrategyBuilder):
        """
        Sets the strategy builder for constructing sector processors.
        
        This method configures the internal builder responsible for creating sector processors,
        allowing the director to utilize specific construction strategies.
        
        Args:
            sector_processer_builder (StrategyBuilder): The builder to use for creating sector processors.
        
        Returns:
            None.
        
        Why:
            This allows customization of how sector processors are created, enabling the director
            to adapt to different equation discovery approaches and optimization techniques.
        """
        print(f'setting builder with {sector_processer_builder}')
        self._builder = sector_processer_builder