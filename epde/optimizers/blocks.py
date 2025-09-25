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
    Base class for constructing equation components.
    
    
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
        """
        Initializes a new block (state) within the equation discovery process.
        
        A block represents a potential component or term within a differential equation.
        This constructor sets up the block's initial state, including its connections
        to other blocks (incoming and outgoing transitions) and flags indicating its role
        in the equation (initial, terminal). These flags are important for the evolutionary
        algorithm to correctly construct and evaluate candidate equations.
        
        Args:
            initial (bool): True if this block can be the starting point of an equation.
            terminal (bool): True if this block can be the ending point of an equation.
        
        Returns:
            None
        
        Fields:
            _incoming (list): List of incoming transitions to this state.
            _outgoing (list): List of outgoing transitions from this state.
            id_set (bool): Flag indicating if the state ID has been assigned.
            initial (bool): Flag indicating if the state is an initial state.
            applied (bool): Flag indicating if the state has been applied.
            terminal (bool): Flag indicating if the state is a terminal state.
        """
        self._incoming = []; self._outgoing = []
        self.id_set = False; self.initial = initial
        self.applied = False; self.terminal = terminal
    
    def add_incoming(self, incoming):
        """
        Adds an incoming connection to the block's list of incoming connections.
        
        This is crucial for tracking data dependencies and ensuring that the block receives the necessary inputs for its operations within the computational graph.
        
        Args:
            incoming: The incoming connection to add.
        
        Returns:
            None
        """
        self._incoming.append(incoming)
        
    def add_outgoing(self, outgoing):
        """
        Adds a new outgoing connection to this block.
        
                This is crucial for establishing the flow of data and dependencies between blocks,
                ensuring that the computational graph accurately reflects the relationships
                defined by the discovered equations.
        
                Args:
                    outgoing: The outgoing connection to add.
        
                Returns:
                    None.
        """
        self._outgoing.append(outgoing)    

    @property
    def op_id(self):
        """
        Return the unique identifier for this processing block.
        
                The ID is lazily generated upon first access to ensure each block has a distinct identity for tracking and management within the equation discovery process. This ID is used to differentiate blocks during evolutionary operations and analysis.
        
                Args:
                    self: The Block instance.
        
                Returns:
                    int: The operation ID, a randomly generated integer if not previously set.
        """
        if not self.id_set:
            self._id = np.random.randint(0, 1e6)
            self.id_set = True
        return self._id
    
    @property
    def available(self):
        """
        Check if all incoming blocks have been processed.
        
                This check ensures that the current block is ready for further computations 
                by verifying that all its dependencies (incoming blocks) have been successfully applied.
        
                Returns:
                     bool: True if all incoming blocks have been applied, False otherwise.
        """
        return all([block.applied for block in self._incoming])

    def apply(self, *args):
        """
        Applies the block's operation.
        
        This method marks the block as processed within the equation discovery workflow. By setting the `applied` flag, it ensures that this block is not re-evaluated, contributing to the efficiency of the search process.
        
        Args:
            *args: Variable length argument list (not directly used but kept for potential future extensions).
        
        Returns:
            None. Sets the `applied` attribute to True, signaling that this block has been processed.
        """
        self.applied = True

    def set_input_combinator(self, combinator : Callable):
        """
        Sets the function to combine inputs from upstream blocks.
        
        This combinator is used to merge the outputs of the blocks connected to the current block.
        It is essential for handling scenarios where a block receives inputs from multiple sources,
        allowing the block to process the combined input effectively.
        
        Args:
            combinator (Callable): A callable (e.g., a function or lambda expression) that
                takes a list of inputs and returns a single combined output.
        
        Returns:
            None
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
        """
        Initializes an evolutionary block with an operator and configures how the operator's arguments are parsed.
        
                This setup determines how the block interacts with the evolutionary process, defining the operation it performs and how it obtains necessary parameters. The configuration of argument parsing is crucial for adapting the operator to different stages of the evolutionary search.
        
                Args:
                    operator (`epde.operators.utils.template.Compound_Operator`): The evolutionary operator that this block will apply.
                    parse_operator_args (`str|tuple|list`): Specifies the method for obtaining arguments for the operator's `apply` method.
                        - If `None`, no arguments are passed.
                        - If a `tuple` or `list`, its elements (strings) are used as keys to retrieve arguments.
                        - If `'use inspect'`, the `inspect` library is used to determine the required arguments.
                        - If `'use operator attribute'`, arguments are taken from the operator's `arg_keys` attribute.
                    terminal (`bool`): Indicates whether this block is the final step in a sequence of operations. If `True`, the evolutionary process terminates after this block.
        
                Returns:
                    None
        """
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
                if arg_key in self.arg_keys:
                # try:
                    self.arg_keys.remove(arg_key)
                # except:
                #     pass
        elif parse_operator_args == 'use operator attribute':
            self.arg_keys = self._operator.arg_keys
        elif isinstance(parse_operator_args, (list, tuple)):
            self.arg_keys = parse_operator_args
        else:
            raise TypeError('Wrong argument passed as the parse_operator_args argument')
        super().__init__(terminal = terminal)
        
    def check_integrity(self):
        """
        Verifies the block's output connections based on whether it's designated as a terminal block. This ensures that terminal blocks do not produce further outputs, while non-terminal blocks must have at least one output to propagate the computation forward.
        
                Args:
                    self: The instance of the EvolutionaryBlock.
        
                Returns:
                    None. Raises a ValueError if the block's output connections are inconsistent with its terminal status.
        
                Why: This check ensures that the computational graph defined by the blocks is valid, preventing terminal blocks from producing further outputs and ensuring that non-terminal blocks contribute to the overall computation. This is crucial for the correct evolution and evaluation of equation structures.
        """
        if self.terminal and len(self._outgoing) > 0:
            raise ValueError('The block is set as the terminal, while it has some output')
        if not self.terminal and len(self._outgoing) == 0:
            raise ValueError('The block is not set as the terminal, while it has no output')
    
    def apply(self, EA_kwargs):
        """
        Applies the operator within the evolutionary algorithm using provided arguments.
        
        This step propagates information through the computational graph, combining the outputs
        of incoming blocks according to the specified operator and arguments. This allows the algorithm
        to explore different equation structures and parameter values.
        
        Args:
            EA_kwargs (dict): A dictionary containing the keyword arguments required by the operator.
                These arguments define the specific parameters and settings used in the operator's application.
        
        Returns:
            None: The method updates the `output` attribute of the block with the result of the operator application.
        """
        self.check_integrity()
        kwargs = {kwarg_key : EA_kwargs[kwarg_key] for kwarg_key in self.arg_keys}
        self.output = self._operator.apply(self.combinator([block.output for block in self._incoming]),
                                           arguments = kwargs)
        super().apply()

    @property
    def op_id(self):
        """
        Return the unique identifier of the evolutionary operator.
        
                This ID is crucial for tracking the operator's performance and lineage
                throughout the evolutionary process. If an ID hasn't been assigned yet,
                a new random integer ID is generated to ensure each operator can be
                uniquely identified.
        
                Args:
                    self: The EvolutionaryBlock instance.
        
                Returns:
                    int: The operator ID.
        """
        if not self.id_set:
            self._id = np.random.randint(0, 1e6) #self._operator.op_id
            self.id_set = True
        return self._id


class InputBlock(Block):
    """
    Blocks with fucntionality running before each evolutionary step
    """

    def __init__(self, to_pass):
        """
        Initializes the InputBlock with a specified initial value.
        
        This block serves as a starting point in the computational graph, 
        providing an initial value that can be passed to subsequent operations.
        The `applied` flag is set to True, indicating that this block is always 
        considered to have been "applied" since it represents the initial input.
        
        Args:
            to_pass (any): The initial value to be held by this block. This value 
                           will be the output of this block.
        
        Returns:
            None.
        
        Initializes:
            output (any): The output value, initialized using the provided `to_pass` argument.
            applied (bool): A boolean flag indicating that the block has been applied, initialized to True.
        """
        self.set_output(to_pass)
        self.applied = True
        super().__init__(initial = True)

    def add_incoming(self, incoming):
        """
        Adds an incoming link to the operator.
        
        This method prevents establishing connections from preceding operations.
        This is because this type of block is designed to be a starting point,
        receiving external data rather than results from other operators.
        
        Args:
            incoming: The incoming link to be added.
        
        Raises:
            TypeError: Always raised, indicating that incoming links are not allowed.
        
        Returns:
            None
        """
        raise TypeError('Objects of this type shall not have incoming links from other operators')

    def set_output(self, to_pass):
        """
        Sets the output of this processing block. This value will be available to subsequent blocks in the processing chain.
        
                Args:
                    to_pass: The value to set as the output. This will typically be data that needs to be processed by subsequent blocks.
        
                Returns:
                    None.
        
                Why: This method allows the block to pass its processed data to the next block in a sequential manner, enabling a chain of operations to be performed.
        """
        self.output = to_pass
        
    @property
    def available(self):
        """
        Indicates if the input data stream is ready for processing.
        
                This property always returns True, signifying that the input block is continuously ready to supply data.
                This ensures that the equation discovery process can proceed without interruption, as the input data is always considered available.
        
                Returns:
                    bool: Always True, indicating continuous data availability.
        """
        return True
        
    
class LinkedBlocks(object):
    """
    A sequence of blocks with evolutionary operators, allowing for a modular and customizable evolutionary process. The structure supports complex relationships between blocks, enabling diverse evolutionary strategies.
    
    
        Attributes:
            blocks_labeled (`dict`): dictionary with names and arguments of operators 
            supress_structure_check (`boolean`): checking of structure
            initial (`list`): keeping initialized evolution operators
            output (`ndarray`): result from applying operator with evoluyion algorithm's arguments
    """

    def __init__(self, blocks_labeled : dict, suppress_structure_check : bool = False):
        """
        Initializes the BlockProgram object, preparing it for execution by associating blocks and managing program state.
        
                The initialization process involves storing the blocks, setting the structure check flag, identifying initial blocks, and setting the terminated flag to False.
                This setup is crucial for the subsequent execution and analysis of the program's structure.
        
                Args:
                    blocks_labeled (dict): A dictionary mapping block IDs to Block objects, representing the program's blocks.
                    suppress_structure_check (bool): A boolean indicating whether to suppress structure checks during execution.
        
                Returns:
                    None
        
                Class Fields Initialized:
                    blocks_labeled (dict): A dictionary mapping block IDs to Block objects, representing the program's blocks.
                    suppress_structure_check (bool): A boolean indicating whether to suppress structure checks during execution.
                    initial (list): A list of tuples, where each tuple contains the index and the Block object that is marked as the initial block.
                    terminated (bool): A boolean indicating whether the program has terminated, initialized to False.
        
                Raises:
                    NotImplementedError: If there are multiple initial blocks, as the current implementation only supports a single entry point.
        """
        self.blocks_labeled = blocks_labeled
        self.suppress_structure_check = suppress_structure_check
        self.initial = [(idx, block) for idx, block in enumerate(self.blocks_labeled.values()) if block.initial]
        self.terminated = False
        if len(self.initial) > 1:
            raise NotImplementedError('The ability to process the multiple initial blocks is not implemented')
    
    def reset_traversal_cond(self):
        """
        Resets the 'applied' flag for all blocks, except InputBlocks, to prepare for a new equation discovery iteration.
        
                This method ensures that only InputBlocks retain their 'applied' status, allowing the algorithm to re-evaluate the contribution of other blocks in constructing potential equation candidates. By resetting the 'applied' flag, the search process can explore alternative equation structures and avoid being biased by previously evaluated combinations.
        
                Args:
                    self: The instance of the LinkedBlocks class.
        
                Returns:
                    None.
        """
        for block in self.blocks_labeled.values():
            if not isinstance(block, InputBlock): block.applied = False 
            
    def traversal(self, input_obj, EA_kwargs):
        """
        Sequentially executes the linked blocks of the equation discovery pipeline.
        
                This method orchestrates the application of evolutionary algorithm blocks,
                propagating data and control flow through the defined computational graph.
                It continues until a terminal condition is met, effectively driving the
                equation discovery process towards a potential solution.
        
                Args:
                    input_obj (`ndarray`): The initial data for equation discovery.
                    EA_kwargs (`dict`): Parameters for the evolutionary algorithm within the blocks.
        
                Returns:
                    None: The method updates the internal state of the LinkedBlocks instance,
                          specifically the `final_vertex` attribute, which stores the terminal
                          block containing the discovered equation.
        """
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
        """
        Returns the output of the final processing step in the linked sequence.
        
        This provides the final result after all transformations and operations have been applied.
        
        Args:
            self: The object instance.
        
        Returns:
            The output of the final vertex, representing the overall result of the processing pipeline.
        """
        return self.final_vertex.output