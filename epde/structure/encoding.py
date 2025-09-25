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
    """
    Represents a gene with a key, value, and value type.
    
    Class Methods:
    - __init__
    - value
    - value
    - __eq__
    - set_metaparam
    - __deepcopy__
    
    Attributes:
        key: The key or name of the gene.
        val_type: The expected type of the gene's value.
        value: The current value of the gene.
    
    Class Methods:
    - __init__:
    Initializes a new Gene object.
    
    Args:
        key: The key or name of the gene.
        value: The initial value of the gene (optional).
        value_type: The expected type of the gene's value (optional).
    
    Raises:
        ValueError: If both value and value_type are None.
        ValueError: If the type of value does not match value_type.
    
    Returns:
        None.
    
    Class Fields:
        key (str): The key or name of the gene.
        val_type (type): The expected type of the gene's value.
        value (any): The current value of the gene.
    """

    def __init__(self, key, value=None, value_type=None):
        """
        Initializes a new Gene object, representing a parameter within an equation.
        
                This method sets up a gene with a specific key (name), an initial value, and a data type.
                The value and value_type are crucial for defining the search space and constraints during the evolutionary process.
                The type checking ensures that the gene's value adheres to the expected data type, maintaining consistency and validity within the equation.
        
                Args:
                    key (str): The key or name of the gene.
                    value (any, optional): The initial value of the gene. Defaults to None.
                    value_type (type, optional): The expected type of the gene's value. Defaults to None.
        
                Raises:
                    ValueError: If both value and value_type are None.
                    ValueError: If the type of value does not match value_type.
        
                Returns:
                    None
        """
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
        """
        Gets the value of the gene.
        
                This value represents a specific parameter or component within the equation being evolved. Accessing it allows the evolutionary algorithm to evaluate the fitness and contribution of this gene to the overall equation.
        
                Raises:
                    ValueError: If the value has not been set, indicating an uninitialized gene.
        
                Returns:
                    The value of the gene.
        """
        if self._value is None:
            raise ValueError('The value of gene was not set before its call')
        return self._value

    @value.setter
    def value(self, val):
        """
        Sets the value of the gene, ensuring it conforms to the expected data type.
        
        This is crucial for maintaining the integrity of the equation being evolved,
        as each gene represents a specific parameter or component within the equation.
        Setting the correct value type ensures that the evolutionary process explores
        valid equation structures.
        
        Args:
            val: The new value to set for the gene.
        
        Raises:
            TypeError: If the provided value is not of the expected type.
        
        Returns:
            None.
        """
        if isinstance(val, self.val_type):
            self._value = val
        else:
            TypeError(
                f'Value of the incorrect type {type(val)} was passed into a gene of {self.val_type}')

    def __eq__(self, other):
        """
        Compares this gene with another to determine if they represent the same element.
        
        This comparison is crucial for evolutionary operations like crossover and mutation,
        ensuring that genetic material is combined and modified in a meaningful way
        by checking if the key-value pairs are identical.
        
        Args:
            other: The other gene to compare against.
        
        Returns:
            bool: True if both the key and value of the genes are equal, False otherwise.
        """
        try:
            return self.key == other.key and self.value == other.value
        except AttributeError:
            warnings.warn(f'The equality method is not implemented for object of type {type(self)} or {type(other)}')

    def set_metaparam(self, key: str, value: Union[float, int]):
        """
        Sets the value of a specified metaparameter for the gene.
        
        This allows fine-grained control over the gene's behavior during the evolutionary process, influencing how it adapts to the data.
        
        Args:
            key: The name of the metaparameter to set.
            value: The new value for the metaparameter.
        
        Returns:
            None.
        """
        assert key in self._value.metaparameters, f'Incorrect parameter key {key} passed into the gene, containing {self.key}'
        self._value.metaparameters[key]['value'] = value

    def __deepcopy__(self, memo):
        """
        Creates a deep copy of the Gene object.
        
                This method is essential for maintaining the integrity of the population
                during the evolutionary process. By creating a new instance with copied
                attributes, we ensure that modifications to a Gene in one generation do
                not inadvertently affect other Genes or previous generations. This is
                crucial for the evolutionary algorithm to explore the search space
                effectively and avoid unintended consequences due to shared object
                references.
        
                Args:
                    self: The Gene object to be copied.
                    memo: A dictionary used by `deepcopy` to prevent infinite recursion
                        when copying objects with circular references.
        
                Returns:
                    A deep copy of the Gene object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Chromosome(object):
    """
    Represents a chromosome in an evolutionary algorithm.
    
        The chromosome stores a collection of equations and provides methods for
        manipulating and comparing them.
    
        Class Methods:
        - __init__
        - replace_gene
        - pass_parametric_gene
        - __eq__
        - __getitem__
        - __len__
        - text_form
        - hash_descr
        - __iter__
        - same_encoding
        - __deepcopy__
    
        Attributes:
            equations (dict of epde.strucutre.main_structures.Equation objects): List of equation, that form the chromosome.
    """


    def __init__(self, equations, params):
        """
        Initializes a Chromosome object with a set of equations and parameters.
        
                The chromosome is a collection of genes, where each gene represents either an equation or a parameter.
                This structure facilitates the evolutionary process by allowing for the manipulation and optimization of both
                the equation structure and its associated parameters.
        
                Args:
                    equations (dict of epde.strucutre.main_structures.Equation objects):
                        A dictionary of equations, where keys are equation identifiers and values are Equation objects.
                        These equations form the basis of the chromosome and represent potential solutions.
                    params (dict):
                        A dictionary of parameters, where keys are parameter identifiers and values are dictionaries
                        containing parameter information (e.g., 'value'). These parameters are also incorporated into the
                        chromosome as genes, allowing for their optimization during the evolutionary process.
        
                Returns:
                    None
        """
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
        """
        Replaces the value of a specific gene in the chromosome. This is a crucial step in the evolutionary process, allowing the algorithm to explore different equation structures by modifying the parameters and operators within the chromosome.
        
                Args:
                    gene_key: The key of the gene to replace.
                    value: The new value for the gene.
        
                Returns:
                    None
        """
        self.chromosome[gene_key].value = value

    def pass_parametric_gene(self, key, value):
        """
        Updates a specific parameter within the chromosome's genes. This method facilitates the modification of equation parameters during the evolutionary search process, allowing the algorithm to explore different equation configurations.
        
                Args:
                    key (tuple of str or str): Specifies the parameter to be updated. If a tuple, it should be in the format (parameter_name, variable_name). If a string, it represents the parameter name.
                    value (float or int): The new value for the specified parameter.
        
                Returns:
                    None
        """
        if isinstance(key, str) and (key in self.chromosome[np.random.choice(self.equation_keys)]):
            for eq_name in self.equation_keys:
                self.chromosome[eq_name].set_metaparam(key=key, value=value)
        elif isinstance(key, (list, tuple)):
            self.chromosome[key[1]].set_metaparam(key=key, value=value)
        else:
            raise ValueError(
                'Incorrect value passed into genes parameters setting.')

    def __eq__(self, other):
        """
        Compares two `Chromosome` objects for equality.
        
                This comparison is crucial for the evolutionary process, ensuring that identical chromosomes are recognized and redundant solutions are avoided, thus maintaining diversity within the population.
        
                Args:
                    other (Chromosome): The `Chromosome` object to compare with.
        
                Returns:
                    bool: True if the two `Chromosome` objects are equal, False otherwise.
        """
        if set(self.chromosome.keys()) != set(self.chromosome.keys()):
            return False
        return all([self.chromosome[key] == other.chromosome[key] for key in self.chromosome.keys()])

    def __getitem__(self, key):
        """
        Retrieves the value of a gene at a specific index within the chromosome's genetic sequence.
        
        This enables access to individual components of a candidate equation,
        allowing for evaluation and manipulation during the evolutionary process.
        
        Args:
            key: The index of the gene to retrieve.
        
        Returns:
            The value of the gene (token) at the specified index.
        """
        return self.chromosome[key].value

    def __len__(self):
        """
        Returns the number of equation keys in the chromosome.
        
                This represents the complexity of the equation encoded by this chromosome.
                A chromosome with more equation keys potentially represents a more complex equation.
        
                Args:
                    self: The Chromosome instance.
        
                Returns:
                    int: The number of equation keys.
        """
        return len(self.equation_keys)

    @property
    def text_form(self):
        """
        Presents the chromosome's genetic information as a list of key-value pairs.
        
        This representation is useful for inspecting the chromosome's composition and 
        facilitates its interpretation within the evolutionary process. Each key represents
        a specific gene location, and the corresponding value reflects the gene's current state.
        
        Args:
            None
        
        Returns:
            list: A list of tuples, where each tuple contains a key (gene identifier) 
                  from the chromosome and its corresponding gene value.
        """
        return [(key, gene.value) for key, gene in self.chromosome.items()]

    @property
    def hash_descr(self):
        """
        Returns a hashable description of the chromosome.
        
                This method generates a tuple of hashable values representing the genes within the chromosome,
                allowing for efficient comparison and caching of chromosome states during the evolutionary process.
                It attempts to use the `hash_descr` method of gene values if available, otherwise it returns the gene's
                value directly if it's a simple type (int, float, str, tuple). This ensures that even complex gene
                representations can be efficiently compared.
        
                Args:
                    self: The object instance.
        
                Returns:
                    tuple: A tuple containing the hashable descriptions of the genes in the chromosome. Returns None for a gene if its value does not have a `hash_descr` attribute and is not a simple type.
        """
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
        """
        Returns an iterator for comparing chromosomes by equality.
        
                This iterator facilitates the comparison of chromosomes, enabling the evolutionary algorithm to assess the diversity and convergence of the population during the equation discovery process.
        
                Returns:
                    ChromosomeEqIterator: An iterator that allows comparing chromosomes for equality.
        """
        return ChromosomeEqIterator(self)

    def same_encoding(self, other):
        """
        Checks if two chromosomes are structurally compatible for genetic operations.
        
        This method determines if two chromosomes can undergo crossover or mutation
        by ensuring they possess the same set of genes (keys). This is crucial for
        maintaining the validity and consistency of the population during the
        evolutionary process, as it ensures that genetic material is exchanged
        between compatible individuals.
        
        Args:
            other: The other chromosome to compare with.
        
        Returns:
            bool: True if both chromosomes have the same keys, False otherwise.
        """
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
        """
        Creates a deep copy of the chromosome.
        
                This method is essential for evolutionary algorithms, where chromosomes
                (representing potential solutions) need to be duplicated without
                affecting the original. It creates a new chromosome instance and
                copies all attributes from the original chromosome to the new one
                using deepcopy. This ensures that all nested objects within the
                chromosome are also copied, preventing unintended modifications to
                the original chromosome during evolutionary operations like crossover
                or mutation. The `memo` dictionary is crucial to prevent infinite
                recursion when copying complex, potentially self-referential chromosome
                structures.
        
                Args:
                  self: The chromosome object to be copied.
                  memo: A dictionary used to keep track of objects that have
                    already been copied during the deep copy process, to prevent
                    infinite recursion.
        
                Returns:
                  The deep copy of the chromosome.
        """
            setattr(result, k, deepcopy(v, memo))
        return result


class ChromosomeEqIterator(object):
    """
    An iterator for traversing equation keys within a chromosome.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, chromosome):
        """
        Initializes the EquationKeysIterator for traversing equation keys.
        
        This iterator facilitates sequential access to the equation keys within a chromosome,
        enabling the evolutionary algorithm to explore different equation structures.
        
        Args:
            chromosome (Chromosome): The chromosome whose equation keys are to be iterated.
        
        Returns:
            None
        
        Class Fields:
            _chromosome (Chromosome): The chromosome being iterated over.
            _idx (int): The current index of the iterator.
            _chromosome_equation_labels (list): A list of equation keys from the chromosome.
        """
        self._chromosome = chromosome
        self._idx = 0
        self._chromosome_equation_labels = list(self._chromosome.equation_keys)

    def __next__(self):
        """
        Returns the next gene in the chromosome.
        
        Iterates through the chromosome based on the order of equation labels to ensure genes are accessed in a consistent and meaningful way during the evolutionary process. This ordered access is crucial for evaluating the fitness of the chromosome and guiding the search for optimal equation structures.
        
        Args:
            self: The object instance.
        
        Returns:
            The next gene in the chromosome.
        
        Raises:
            StopIteration: If the end of the chromosome is reached.
        """
        if self._idx < len(self._chromosome_equation_labels):
            res = self._chromosome[self._chromosome_equation_labels[self._idx]]
            self._idx += 1
            return res
        else:
            raise StopIteration
