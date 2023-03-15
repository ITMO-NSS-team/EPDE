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
    _container_instances = {}
    
    def __call__(cls, *args, **kwargs): 
        if cls not in cls._container_instances:
            instance = super().__call__(*args, **kwargs)
            cls._container_instances[cls] = instance
            
        return cls._container_instances[cls]
    
    def reset(self):
        self._container_instances = {}

class EvolutionaryParams(metaclass = ParamContainerMeta):
    '''
    Loading of default parameters. 
    Inspired by https://github.com/aimclub/FEDOT/blob/master/fedot/core/repository/default_params_repository.py
    '''
    
    def __init__(self, parameter_file : str = None, mode : str = 'multi objective') -> None:
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
        return self

    def __exit__(self, type, value, traceback):
        self._repo_path = None

    def _initialise_repo(self) -> dict:
        with open(self._repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        return repository_json

    def get_default_params_for_operator(self, operator_name : str) -> dict:
        if operator_name in self._repo:
            return self._repo[operator_name]
        else:
            raise Exception(f'Operator with key {operator_name} is missing from the repo with params')
        # return {}
    
    def change_operator_param(self, operator_name : str, parameter_name : str, new_value):
        if type(new_value) != type(self._repo[operator_name][parameter_name]):
            old_type = type(self._repo[operator_name][parameter_name])
            new_type = type(new_value)
            warnings.warn(f'Possibly incorrect parameter change: from {old_type} to {new_type}.')
        self._repo[operator_name][parameter_name] = new_value