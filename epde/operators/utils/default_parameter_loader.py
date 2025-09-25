#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:03:28 2023

@author: maslyaev
"""

import os
import json
import warnings

class ParamContainerMeta(type):
    """
    Metaclass for managing instances of ParamContainer classes.
    
        This metaclass implements a singleton pattern for ParamContainer classes,
        ensuring that only one instance of each ParamContainer subclass exists.
    
        Class Attributes:
        - _container_instances
    
        Class Methods:
        - __call__:
    """

    _container_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        """
        Ensures that only one instance of a parameter container class exists, centralizing parameter management for equation discovery.
        
                This method implements a singleton pattern, so that only one instance of each parameter container class is created.
                If the container was already created - the method returns a link to the existing container.
                Otherwise, it creates a new instance, stores it, and returns it.
                
                Args:
                    cls: The class to instantiate.
                    *args: Variable length argument list passed to the class constructor.
                    **kwargs: Arbitrary keyword arguments passed to the class constructor.
                
                Returns:
                    The singleton instance of the class.
        """
        if cls not in cls._container_instances:
            instance = super().__call__(*args, **kwargs)
            cls._container_instances[cls] = instance
            
        return cls._container_instances[cls]
    
    def reset(self):
        """
        Resets the container instances.
        
        This method clears the internal dictionary that stores container instances.
        This ensures that subsequent calls to retrieve container instances will create new instances if necessary,
        allowing for a fresh configuration or re-initialization of parameters.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._container_instances = {}

class EvolutionaryParams(metaclass = ParamContainerMeta):
    '''
    Loading of default parameters. 
        Inspired by https://github.com/aimclub/FEDOT/blob/master/fedot/core/repository/default_params_repository.py
    '''

    
    def __init__(self, parameter_file : str = None, mode : str = 'multi objective') -> None:
        """
        Initializes the EvolutionaryParams class, loading parameter settings essential for configuring the evolutionary search process.
        
                This ensures that the evolutionary algorithm has the necessary information to explore the search space effectively. If no parameter file is provided, default configurations are loaded based on the specified optimization mode (single or multi-objective).
        
                Args:
                    parameter_file (str, optional): The path to a JSON file containing custom parameter settings. Defaults to None, in which case a default parameter file is loaded.
                    mode (str, optional): Specifies the optimization mode, either 'single objective' or 'multi objective'. Defaults to 'multi objective'. This influences which default parameter file is loaded.
        
                Returns:
                    None
        
                Class Fields Initialized:
                    mode (str): The optimization mode, either 'single objective' or 'multi objective'.
                    _repo_path (str): The full path to the parameter file within the repository.
                    _repo (dict): The initialized repository (loaded from the parameter file), containing the parameter settings for the evolutionary algorithm.
        """
        if parameter_file is None:
            if mode == 'single objective':
                parameter_file = 'default_parameters_single_objective.json'
            elif mode == 'multi objective':
                parameter_file = 'default_parameters_multi_objective.json'
        
        self.mode = mode
        repo_folder = str(os.path.dirname(__file__))
        file = os.path.join('parameters', parameter_file)
        self._repo_path = os.path.join(repo_folder, file)
        self._repo = self._initialise_repo()

    def __enter__(self):
        """
        Enters the context, making the `EvolutionaryParams` instance available for use within a `with` statement. This ensures that the parameter settings for the evolutionary equation discovery process are properly managed and scoped.
        
                Args:
                    None
        
                Returns:
                    self: The instance of the `EvolutionaryParams` class, allowing for chained operations within the context.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exits the context manager.
        
        Releases the repository path to ensure resources are properly managed after equation discovery. This is crucial for maintaining a clean state and preventing potential conflicts in subsequent equation searches.
        
        Args:
            type: The exception type, if any occurred within the context.
            value: The exception value, if any.
            traceback: The traceback, if any.
        
        Returns:
            None.
        """
        self._repo_path = None
        # self._repo = None        

    def _initialise_repo(self) -> dict:
        """
        Initializes the parameter repository by loading its JSON file.
        
                This method reads the JSON file located at the repository path
                and loads its content into a Python dictionary, making the
                parameter settings accessible for configuring the evolutionary
                search process. This ensures that the evolutionary algorithm
                starts with a defined set of parameters.
        
                Args:
                    self: The object instance.
        
                Returns:
                    dict: A dictionary containing the data loaded from the
                        repository's JSON file, representing the initial
                        parameter configuration.
        """
        with open(self._repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        return repository_json

    def get_default_params_for_operator(self, operator_name : str) -> dict:
        """
        Retrieves the default parameter set associated with a specified operator.
        
        This function is essential for configuring the evolutionary search process, ensuring that each operator 
        is initialized with a valid and potentially effective set of parameters. These parameters guide the 
        operator's behavior during the equation discovery process.
        
        Args:
            operator_name (str): The unique identifier of the operator.
        
        Returns:
            dict: A dictionary containing the default parameter values for the specified operator.
        
        Raises:
            Exception: If no operator with the given name is found within the internal operator repository.
        """
        if operator_name in self._repo:
            return self._repo[operator_name]
        else:
            raise Exception(f'Operator with key {operator_name} is missing from the repo with params')
    
    def change_operator_param(self, operator_name : str, parameter_name : str, new_value):
        """
        Changes the value of a parameter for a given operator in the repository. This allows fine-tuning of the evolutionary search process by modifying operator-specific settings. A warning is issued if the new value's type differs from the original parameter's type.
        
                Args:
                    operator_name (str): The name of the operator whose parameter needs to be changed.
                    parameter_name (str): The name of the parameter to be changed.
                    new_value: The new value for the parameter.
        
                Returns:
                    None.
        """
        if type(new_value) != type(self._repo[operator_name][parameter_name]):
            old_type = type(self._repo[operator_name][parameter_name])
            new_type = type(new_value)
            warnings.warn(f'Possibly incorrect parameter change: from {old_type} to {new_type}.')
        self._repo[operator_name][parameter_name] = new_value