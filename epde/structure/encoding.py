#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:40:00 2022

@author: maslyaev
"""

import warnings

class Gene(object):
    def __init__(self, key, value = None, value_type = None):
        if value is None and value_type is None:
            raise ValueError('Both value and value type can not be None during gene initialization')
        elif type(value) != value_type and value_type is not None:
            raise ValueError('Both value and value type can not be None during gene initialization')
            
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
            TypeError(f'Value of the incorrect type {type(val)} was passed into a gene of {self.val_type}')

    def __eq__(self, other):
        try:
            return self.key == other.key and self.value == other.value
        except AttributeError:
            warnings.warn(f'The equality method is not implemented for object of type {type(self)} or {type(other)}')


class Chromosome(object):
    def __init__(self, equations, **kwargs):
        self.equation_type = type(equations[0])
        
        self.chromosome = {eq.main_var_to_explain : Gene(key = eq.main_var_to_explain, value = eq)
                           for eq in equations}
        self.equation_keys = list(self.chromosome.keys())
        self.params_keys = list(kwargs.keys())
        for key, arg in kwargs.items():
            self.chromosome[key] = Gene(key = key, value = arg)
            
            
    def replace_gene(self, gene_key, value):
        self.chromosome[gene_key].value = value
        
    def __eq__(self, other):
        if set(self.chromosome.keys()) != set(self.chromosome.keys()):
            return False
        return all([self.chromosome[key] == other.chromosome[key] for key in self.chromosome.keys()])
    
    def __getitem__(self, key):
        return self.chromosome[key].value
    
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
        cond_1 = all([key in other.chromosome.keys() for key in self.chromosome.keys()])
        cond_2 = all([key in self.chromosome.keys() for key in other.chromosome.keys()])
        return cond_1 and cond_2
    

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