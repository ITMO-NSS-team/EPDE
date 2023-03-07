#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:40:00 2022

@author: maslyaev
"""

import numpy as np
import warnings
from copy import deepcopy

from typing import Union


class Gene(object):
    def __init__(self, key, value=None, value_type=None):
        if value is None and value_type is None:
            raise ValueError(
                'Both value and value type can not be None during gene initialization')
        elif type(value) != value_type and value_type is not None:
            raise ValueError(
                'Both value and value type can not be None during gene initialization')

        self.key = key
        self.val_type = value_type if value is None else type(value)
        self.value = value

    @property
    def value(self):
        if self._value is None:
            raise ValueError('The value of gene was not set before its call')
        return self._value

    @value.setter
    def value(self, val):
        if isinstance(val, self.val_type):
            self._value = val
        else:
            TypeError(
                f'Value of the incorrect type {type(val)} was passed into a gene of {self.val_type}')

    def __eq__(self, other):
        try:
            return self.key == other.key and self.value == other.value
        except AttributeError:
            warnings.warn(f'The equality method is not implemented for object of type {type(self)} or {type(other)}')

    def set_metaparam(self, key: str, value: Union[float, int]):
        assert key in self._value.metaparameters, f'Incorrect parameter key {key} passed into the gene, containing {self.key}'
        self._value.metaparameters[key]['value'] = value

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Chromosome(object):

    def __init__(self, equations, params):
        '''

        Parameters
        ----------
        equations : dict of epde.strucutre.main_structures.Equation objects
            List of equation, that form the chromosome.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # print('equations', equations)

        self.equation_type = type(next(iter(equations)))

        # eq.main_var_to_explain
        self.chromosome = {key: Gene(key=key, value=eq)
                           for key, eq in equations.items()}
        self.equation_keys = list(self.chromosome.keys())
        self.params_keys = list(params.keys())
        for key, arg in params.items():
            self.chromosome[key] = Gene(key=key, value=arg['value'])

    def replace_gene(self, gene_key, value):
        self.chromosome[gene_key].value = value

    def pass_parametric_gene(self, key, value):
        '''

        Parameters
        ----------
        key : tuple of format (parameter_name, variable_name) of 'str', or 'str' for parameter_name,
            The key, encoding with a tuple, dedicated to the altered metaparametric gene of the chromosome.
            First element of the key is the label of the parameter, while the second is the name of the parameter
        value : float or integer,
            The value, which is replacing the previous gene value.

        Returns
        -------
        None.

        '''
        if isinstance(key, str) and (key in self.chromosome[np.random.choice(self.equation_keys)]):
            for eq_name in self.equation_keys:
                self.chromosome[eq_name].set_metaparam(key=key, value=value)
        elif isinstance(key, (list, tuple)):
            self.chromosome[key[1]].set_metaparam(key=key, value=value)
        else:
            raise ValueError(
                'Incorrect value passed into genes parameters setting.')

    def __eq__(self, other):
        if set(self.chromosome.keys()) != set(self.chromosome.keys()):
            return False
        return all([self.chromosome[key] == other.chromosome[key] for key in self.chromosome.keys()])

    def __getitem__(self, key):
        return self.chromosome[key].value

    def __len__(self):
        return len(self.equation_keys)

    @property
    def text_form(self):
        return [(key, gene.value) for key, gene in self.chromosome.items()]

    @property
    def hash_descr(self):
        def get_gene_hash(gene):
            if isinstance(gene.val_type, (int, float, str, tuple)):
                return gene.value
            else:
                try:
                    return gene.value.hash_descr
                except AttributeError:
                    return None

        return tuple(map(get_gene_hash, self.chromosome.values()))

    def __iter__(self):
        return ChromosomeEqIterator(self)

    def same_encoding(self, other):
        cond_1 = all([key in other.chromosome.keys()
                     for key in self.chromosome.keys()])
        cond_2 = all([key in self.chromosome.keys()
                     for key in other.chromosome.keys()])
        return cond_1 and cond_2

    def __deepcopy__(self, memo):  # TODO: overload
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class ChromosomeEqIterator(object):
    def __init__(self, chromosome):
        self._chromosome = chromosome
        self._idx = 0
        self._chromosome_equation_labels = list(self._chromosome.equation_keys)

    def __next__(self):
        if self._idx < len(self._chromosome_equation_labels):
            res = self._chromosome[self._chromosome_equation_labels[self._idx]]
            self._idx += 1
            return res
        else:
            raise StopIteration
