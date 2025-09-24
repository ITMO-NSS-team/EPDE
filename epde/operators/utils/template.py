#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect

from functools import wraps
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from epde.structure.main_structures import Term, Equation, SoEq
from epde.structure.encoding import Gene, Chromosome
from epde.optimizers.moeadd.moeadd import ParetoLevels

OPERATOR_LEVELS = ('custom level', 'term level', 'gene level', 'chromosome level',
                   'population level')

OPERATOR_LEVELS_SUPPORTED_TYPES = {'custom level': None, 'term level': Term, 'gene level': Gene,
                                   'chromosome level': Chromosome, 'population level': ParetoLevels}

def add_base_param_to_operator(operator, target_dict):
    """
    Adds necessary parameters to the operator to ensure its proper functioning within the evolutionary process.
    
        It iterates through the default parameters defined for the given operator type and
        populates the operator's parameter dictionary. If a parameter key
        is present in the provided `target_dict`, its value is used, allowing customization.
        Otherwise, the predefined default value is applied, ensuring a complete parameter set.
        This ensures that each operator has all the necessary parameters, using defaults where
        custom values are not provided, which is crucial for the evolutionary algorithm to
        explore the search space effectively.
    
        Args:
            operator: The operator to which parameters are being added.
            target_dict: A dictionary containing parameter values to override the defaults.
    
        Returns:
            None. The operator's parameter dictionary is modified in place.
    """
    params_container = EvolutionaryParams()
    for param_key, param_value in params_container.get_default_params_for_operator(operator.key).items():
        operator.params[param_key] = target_dict[param_key] if param_key in target_dict.keys(
        ) else param_value

class CompoundOperator():
    """
    Represents a general operator with customizable functionality.
    
    
        Attributes:
            suboperators (`dict`): dictionary with name of suboperators and its argumetns
            params (`dict`): dictionary with names and values of parameters for the operator
            param_keys (`list`): names of parameters of the operator
            level_index (`tuple`): abstraction level, to indicate which classes of objects the operator is applied to
            operator_tags (`set`): log about operator
        '''
    """


    def __init__(self, param_keys: list = []):
        """
        Initializes a compound operator, preparing it to manage parameters and sub-operators for equation discovery.
        
                This setup is crucial for organizing the components of a complex equation and managing their individual parameters during the evolutionary search process. By initializing the parameter keys and internal dictionaries, the compound operator ensures a structured environment for defining and manipulating equation structures.
        
                Args:
                    param_keys (list, optional): A list of parameter keys that this operator will manage. Defaults to an empty list.
        
                Returns:
                    None
        """
        self.param_keys = param_keys
        self._params = {}
        self._suboperators = {}

        self.use_default_tags()

    @property
    def params(self):
        """
        Returns the parameters of the compound operator.
        
                These parameters define the specific configuration of the operator, 
                allowing it to be adapted and optimized during the equation discovery process.
        
                Returns:
                    dict: The parameters of the compound operator.
        """
        return self._params

    @params.setter
    def params(self, param_dict: dict):
        """
        Sets the operator's parameters, ensuring they align with the expected keys. This is crucial for configuring the operator's behavior within the equation discovery process.
        
                Args:
                    param_dict (dict): A dictionary containing the parameters to be set. The keys of this dictionary must match the expected parameter keys defined for the operator.
        
                Raises:
                    KeyError: If the keys in the input dictionary do not match the expected parameter keys. This ensures that the operator is configured correctly with the necessary parameters.
        
                Returns:
                    None: This method modifies the operator's internal state by updating its parameters.
        
                Class Fields:
                    _params (dict): A dictionary storing the parameters of the object. Initialized with the values from the input `param_dict`.
        """
        if set(self.param_keys) != set(param_dict.keys()):
            print('self.param_keys:', set(self.param_keys),
                  ' param_dict.keys():', set(param_dict.keys()))
            raise KeyError('Wrong keys of param dict')
        self._params = param_dict

    @property
    def suboperators(self):
        """
        Returns the suboperators that constitute this compound operator.
        
                This property provides access to the individual operators that are combined to form the larger, more complex operator.
                Accessing suboperators allows for inspection, manipulation, or analysis of the operator's structure.
        
                Returns:
                    list: The list of suboperators.
        """
        return self._suboperators

    def set_suboperators(self, operators: dict, probas: dict = {}):
        """
        Sets the sub-operators that define the composite evolutionary operation.
        
        This method configures the building blocks of a more complex evolutionary step.
        By combining simpler operators, EPDE can explore a wider range of equation structures.
        
        Args:
            operators (dict): A dictionary where keys are names (strings) and values are the corresponding sub-operators.
                Sub-operators can be other CompoundOperators, lists, tuples, or dictionaries of operators,
                allowing for hierarchical construction of evolutionary steps.
            probas (dict, optional): A dictionary specifying the probabilities of execution for each sub-operator.
                Keys are the names of the operators, and values are their corresponding probabilities.
                Defaults to an empty dictionary, implying uniform probabilities.
        
        Returns:
            None
        """
        if not all([isinstance(key, str) and (isinstance(value, (list, tuple, dict)) or
                                              issubclass(type(value), CompoundOperator))
                    for key, value in operators.items()]):
            print([(key, isinstance(key, str),
                    value, (isinstance(value, (list, tuple, dict)) or 
                            issubclass(type(value), CompoundOperator)))
                    for key, value in operators.items()])
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - CompoundOperator, list, tuple or dict')
        self._suboperators = SuboperatorContainer(suboperators = operators, probas = probas) 

    def get_suboperator_args(self, personal=False):
        """
        Collect the arguments required by this operator and its constituent sub-operators.
        
        This method aggregates the arguments needed for the current operator and recursively gathers arguments from all sub-operators within the compound structure. This is essential for understanding the complete set of inputs required to execute the entire computational graph represented by this operator.
        
        Args:
            personal (`boolean`): If True, return only the arguments directly associated with this operator, without including those of its sub-operators. Defaults to False, which means arguments from sub-operators are included.
        
        Returns:
            `set`: A set containing the names of all arguments required by this operator and, if `personal` is False, its sub-operators.
        """
        args = self.arguments
        if not personal:
            for operator in self.suboperators:
                args = args.union(operator.get_suboperator_args())

        technical_args = ['arguments', 'objective', 'obj', 'self']
        for arg in technical_args:
            if arg in args:
                args.remove(arg)

        return args

    def _check_objective_type(method):
        """
        Wraps a method to validate the type of the objective being processed by the operator.
        
                This decorator ensures that the objective passed to the decorated method
                conforms to the expected type based on the operator's defined level.
                This type checking is bypassed for 'custom level' operators, allowing
                greater flexibility in objective types when needed.
        
                Args:
                    method: The method to be wrapped.
        
                Returns:
                    The wrapped method.
        
                Why:
                    This type checking ensures that operators at different levels
                    receive objectives of the correct type, maintaining consistency
                    and preventing unexpected errors during the equation discovery
                    process. It is important because different equation levels may
                    require different data types for processing.
        """
        @wraps
        def wrapper(self, *args, **kwargs):
            objective = args[0]
            try:
                level_descr = [
                    tag for tag in self.operator_tags if 'level' in tag][0]
            except IndexError:
                level_descr = 'custom level'
            if level_descr == 'custom level':
                result = method(self, *args, **kwargs)
            else:
                processing_type = OPERATOR_LEVELS_SUPPORTED_TYPES[level_descr]
                # TODO: переписать выбор типа и тэги под кастомные объекты
                if isinstance(objective, self.operator_tags[processing_type]):
                    result = method(self, *args, **kwargs)
                else:
                    raise TypeError(
                        f'Incorrect input type of the EA operator objective: {type(objective)} does not match {processing_type}')
                return result
        return wrapper

    def parse_suboperator_args(self, arguments: dict):
        """
        Parses and distributes arguments to sub-operators.
        
        This method takes a dictionary of arguments and distributes them to the appropriate sub-operators based on their declared arguments.
        It ensures that each sub-operator receives only the arguments it needs, facilitating modular execution and reducing potential conflicts.
        
        Args:
            arguments (dict): A dictionary containing argument names and their corresponding values.
        
        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries:
                - The first dictionary contains all arguments relevant to the compound operator and its sub-operators.
                - The second dictionary is structured with sub-operator names as keys, and each key maps to a dictionary of arguments specific to that sub-operator.
        """
        def parse_args(keys, args):
            return {key: args[key] for key in keys}

        operators_args = {}
        for key in self.suboperators.keys():
            operators_args[key] = parse_args(
                self.suboperators[key].get_suboperator_args(), arguments)

        return parse_args(self.get_suboperator_args(True), arguments), operators_args

    @_check_objective_type
    def apply(self, objective, arguments: dict):
        """
        Applies the compound operator to the given objective by delegating to its sub-operators.
        
                This abstract method serves as a template for applying a sequence of operators.
                It parses the arguments, separating those relevant to the compound operator itself
                from those intended for its constituent sub-operators. This ensures that each
                operator in the sequence receives the correct inputs for its specific transformation.
                The base class implementation raises a NotImplementedError to enforce that concrete
                subclasses define the specific application logic.
        
                Args:
                    objective: The objective to which the compound operator is applied. This represents
                        the current state of the equation being built or modified.
                    arguments (dict): A dictionary containing arguments for the compound operator and
                        its sub-operators. These arguments guide the behavior of the operators during
                        the equation discovery process.
        
                Returns:
                    NotImplementedError: Always raises a NotImplementedError, as the base class
                        implementation is abstract. Subclasses must implement the actual application logic.
        """
        self_args, subop_args = self.parse_suboperator_args(
            arguments=arguments)
        raise NotImplementedError('Trying to apply abstract superclass of the operator.')

    @property
    def level_index(self):
        """
        Determines the precedence level of the operator within the equation discovery process.
        
        This method helps to establish the order in which operators are applied when constructing and evaluating equation candidates.
        Operators with higher precedence levels are applied before those with lower levels, influencing the structure of the discovered equations.
        
        Args:
            None
        
        Returns:
            tuple: A tuple containing the index (representing precedence) and the level (a string describing the level) of the operator.
                   Returns (0, 'custom level') if the operator's level is not found in the OPERATOR_LEVELS list,
                   indicating a user-defined or less common operator precedence.
        """
        try:
            return [(idx, level) for idx, level in enumerate(OPERATOR_LEVELS)
                    if level in self.operator_tags][0]
        except IndexError:
            return (0, 'custom level')

    def use_default_tags(self):
        """
        Resets the operator's tags to an empty set.
        
                This ensures that the operator relies on the default tagging behavior
                defined within the system. This is useful when specific tags are
                no longer relevant or when reverting to a standard configuration
                for equation discovery.
        
                Args:
                    self: The CompoundOperator instance.
        
                Returns:
                    None. The method modifies the operator's tags in place.
        """
        self._tags = set()

    @property
    def operator_tags(self):
        """
        Returns the tags associated with the operator. These tags provide metadata
                about the operator, which can be used to filter and organize operators
                during the equation discovery process. For example, tags might indicate
                the type of operation (e.g., 'derivative', 'nonlinear') or the variables
                involved.
        
                This information is valuable for guiding the search and selection of
                equation structures within the EPDE framework.
        
                Returns:
                    list: A list of tags associated with the operator.
        """
        return self._tags

    @property
    def arguments(self):
        """
        Returns the set of arguments used by the operator.
        
        This method is designed to provide a way to inspect the dependencies
        or inputs required by this operator, which is essential for
        understanding its role within a larger equation or model.
        Since this is a base class, it returns an empty set.
        
        Args:
            None
        
        Returns:
            set: An empty set, as a `CompoundOperator` has no arguments by default.
        """
        return set()


class SuboperatorContainer():
    """
    Object, implemented to contain the suboperators of a CompoundOperator object. 
        Main purpose: support of uneven probabilities of similar suboperator application.
    
        Class Methods:
        - __init__
        - __getitem__
        - keys
        - __iter__
    """

    def __init__(self, suboperators: dict = {}, probas: dict = {}):
        """
        Object that manages a collection of sub-operators, assigning probabilities to each for selection during the compound operator's execution. This allows for a flexible and weighted application of different sub-operators within a larger operation.
        
                Args:
                    suboperators (dict, optional): A dictionary mapping labels to sub-operators (or lists/tuples/arrays of sub-operators). Defaults to {}.
                    probas (dict, optional): A dictionary mapping labels to probabilities for sub-operators. Defaults to {}.
        
                Raises:
                    ValueError: If the number of sub-operators for a given label does not match the number of defined probabilities for that label.
        
                Returns:
                    None
        """
        self.suboperators = suboperators
        self.probas = {}

        for label, oper in suboperators.items():
            if isinstance(oper, CompoundOperator):
                operator_probability = 1
            elif isinstance(oper, (list, tuple, np.ndarray)) and label not in probas.keys():
                operator_probability = np.full(
                    fill_value=1./len(oper), shape=len(oper))
            elif isinstance(oper, (list, tuple, np.ndarray)) and label in probas.keys():
                if len(oper) != len(probas[label]):
                    raise ValueError(
                        f'Number of passed suboperators for {label} does not match defined probabilities.')
                operator_probability = probas[label]
            self.probas[label] = operator_probability

    def __getitem__(self, item):
        """
        Retrieves a sub-operator based on the given item, enabling the evolutionary search for optimal equation structures.
        
                If the sub-operator at the given index is a CompoundOperator, it is returned directly, preserving the hierarchical structure of the equation.
                Otherwise, if it's a list, tuple, or NumPy array, a random element is chosen from it
                based on the corresponding probability distribution. This introduces stochasticity, allowing the evolutionary algorithm to explore a wider range of equation candidates.
        
                Args:
                    item: The index or key of the sub-operator to retrieve.
        
                Returns:
                    The sub-operator at the given index, or a randomly chosen element from it. This element will be used in constructing and evaluating candidate equations.
        """
        if isinstance(self.suboperators[item], CompoundOperator):
            return self.suboperators[item]
        elif isinstance(self.suboperators[item], (list, tuple, np.ndarray)):
            return np.random.choice(a=self.suboperators[item], p=self.probas[item])

    def keys(self):
        """
        Returns a view of the names (keys) of the suboperators stored within this container.
        
        This allows iteration and inspection of available suboperators, which represent
        individual components or terms within a larger equation or model being constructed.
        Accessing keys is essential for managing and manipulating these suboperators during
        the equation discovery process.
        
        Args:
            self: The SuboperatorContainer instance.
        
        Returns:
            dict_keys: A view object containing the names (keys) of the suboperators.
        """
        return self.suboperators.keys()

    def __iter__(self):
        """
        Returns an iterator for the suboperators within this container. This allows to seamlessly traverse and process individual suboperators, which is essential for applying evolutionary algorithms and multi-objective optimization techniques to discover the best equation structures.
        
                Args:
                    None
        
                Returns:
                    SuboperatorContainerIterator: An iterator object for traversing the suboperators within this container.
        """
        return SuboperatorContainerIterator(self)


class SuboperatorContainerIterator(object):
    """
    An iterator for traversing suboperators within a container.
    
        This iterator flattens and iterates through a container's suboperators,
        handling nested CompoundOperator instances and other iterable values.
    
        Attributes:
            _suboperators: A list containing all suboperators extracted from the
                container. CompoundOperator instances are appended directly, while
                other values are extended (assumed to be iterable).
            _idx: An integer representing the current index, initialized to 0.
    """

    def __init__(self, container):
        """
        Initializes the iterator with a container of suboperators.
        
                The constructor flattens the nested structure of suboperators within the
                provided container, preparing them for sequential access. This flattening
                ensures that all suboperators, regardless of their nesting level, can be
                iterated over in a consistent manner, which is essential for the equation
                discovery process where each operator needs to be evaluated and combined
                efficiently.
        
                Args:
                    container: An object holding suboperators, expected to have a
                        'suboperators' attribute which is a dictionary.
        
                Returns:
                    None
        """
        self._suboperators = []
        for val in container.suboperators.values():
            self._suboperators.append(val) if isinstance(
                val, CompoundOperator) else self._suboperators.extend(val)
        self._idx = 0

    def __next__(self):
        """
        Return the next suboperator.
        
                This iterator provides sequential access to the suboperators within the container.
                It's used to traverse the equation structure, enabling the evolutionary algorithm to explore and modify different parts of the equation during the search process.
        
                Args:
                    self: The iterator instance.
        
                Returns:
                    The next suboperator in the sequence.
        
                Raises:
                    StopIteration: If all suboperators have been visited. This signals the end of the current equation's structure traversal.
        """
        if self._idx < len(self._suboperators):
            res = self._suboperators[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration