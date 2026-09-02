#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:03:28 2023

@author: maslyaev
"""

import warnings

from epde.interface.search_config import active_config


class ParamContainerMeta(type):
    _container_instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._container_instances:
            instance = super().__call__(*args, **kwargs)
            cls._container_instances[cls] = instance

        return cls._container_instances[cls]

    def reset(self):
        self._container_instances = {}


class EvolutionaryParams(metaclass=ParamContainerMeta):
    '''
    Per-operator parameters for the operators the strategy assembles.

    These used to be read from ``parameters/default_parameters_*.json``. They
    now come from the active search configuration
    (``evolution.operators``), which is resolved from
    :data:`~epde.interface.search_config.MULTI_OBJECTIVE_OPERATORS` /
    ``SINGLE_OBJECTIVE_OPERATORS`` under the caller's ``operators={...}``
    override. That makes the configuration the single place a default is
    written: the JSON files also declared ``pinn_loss_mult``,
    ``error_metric``, ``deepxde_config``, ``PBI_penalty``,
    ``number_of_neighbors`` and ``delta``, which are search settings in their
    own right, and the two declarations could silently disagree.

    Still a singleton, and still reset by ``EpdeSearch.__init__``: operators
    are constructed once per strategy and read it during assembly.
    '''

    def __init__(self) -> None:
        self._repo = self._initialise_repo()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @staticmethod
    def _initialise_repo() -> dict:
        # A copy: ``change_operator_param`` mutates this, and the config is
        # frozen and shared.
        return {name: dict(params) for name, params
                in active_config().evolution.operators.items()}

    @property
    def mode(self) -> str:
        return ('multi objective'
                if active_config().objectives.multiobjective_mode
                else 'single objective')

    def get_default_params_for_operator(self, operator_name: str) -> dict:
        if operator_name in self._repo:
            return self._repo[operator_name]
        else:
            raise Exception(f'Operator with key {operator_name} is missing from the repo with params')

    def change_operator_param(self, operator_name: str, parameter_name: str, new_value):
        if type(new_value) != type(self._repo[operator_name][parameter_name]):
            old_type = type(self._repo[operator_name][parameter_name])
            new_type = type(new_value)
            warnings.warn(f'Possibly incorrect parameter change: from {old_type} to {new_type}.')
        self._repo[operator_name][parameter_name] = new_value
