#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:38:20 2022

@author: maslyaev
"""

import numpy as np
from functools import reduce
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

def check_uniqueness(obj, background):
    """
    Checks if a given equation structure is unique within a population of candidate solutions.
    
    This prevents redundant computations and ensures diversity within the evolutionary search process.
    
    Args:
        obj: The equation structure to check for uniqueness.
        background: A list of existing equation structures in the population.
    
    Returns:
        bool: True if the equation structure is not already present in the population, False otherwise.
    """
    return not any([elem == obj for elem in background])


class ComplexStructure(object):
    """
    Represents a complex structure with a name and a dictionary of components.
    
             Methods:
                 - __init__
                 - add_component
                 - get_component
                 - remove_component
                 - get_all_components
                 - set_name
                 - get_name
    
             Attributes:
                 - name (str): The name of the complex structure.
                 - components (dict): A dictionary storing the components of the structure.
    """

    def __init__(self, interelement_operator=np.add, *params):
        """
        Initializes the ComplexStructure object with an interelement operator.
        
                This method sets up the ComplexStructure by storing the provided operator,
                which will be used to combine individual elements within the structure during
                the equation discovery process. An empty history and a placeholder for the
                structure are also initialized. The interelement operator is crucial for
                defining how the discovered equation terms are combined to form a complete
                equation.
        
                Args:
                    interelement_operator: The operator (e.g., np.add, np.multiply) to use when combining elements. Defaults to np.add.
                    *params: Variable length argument list (currently unused).
        
                Returns:
                    None.
        """
        self._history = ''
        self.structure = None
        self.interelement_operator = interelement_operator
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Reconstructs a specific attribute of the object, bypassing the standard reconstruction process.
        
                This method allows for direct manipulation of the object's internal state, which is necessary when the standard reconstruction via evolutionary search or data loading fails to properly restore the object's attributes. It is used to ensure that critical components of the object are correctly initialized or updated, especially when dealing with complex dependencies or custom data structures.
        
                Args:
                    attribute (str): The name of the attribute to reconstruct.
                    value: The new value to assign to the attribute.
                    except_attrs (dict): A dictionary of attributes to exclude during the reconstruction process.
        
                Returns:
                    None.
        
                Raises:
                    ValueError: If the specified attribute is not supported for manual reconstruction.
        """
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
    
    def __eq__(self, other):
        """
        Compares two `ComplexStructure` objects for structural equality, regardless of element order.
        
                This method determines if two `ComplexStructure` instances represent the same underlying structure, even if the elements within their respective structures are arranged differently. This is crucial for identifying equivalent equation structures during the equation discovery process, where the order of terms might vary without affecting the equation's meaning.
        
                Args:
                    other: The `ComplexStructure` object to compare with.
        
                Returns:
                    bool: True if the structures are equivalent, False otherwise.
        
                Raises:
                    ValueError: If 'other' is not a `ComplexStructure` instance.
        """
        if type(other) != type(self):
            raise ValueError('Type of self and other are different')
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure]) and
                len(other.structure) == len(self.structure))

    def __iter__(self):
        """
        Returns an iterator for the columns within this structure.
        
        This method enables iteration over the structure's columns, which is essential
        for traversing and processing the data organized within. It returns a
        `CSIterator` object, facilitating access to individual columns for
        analysis or manipulation.
        
        Args:
            self: The ComplexStructure instance.
        
        Returns:
            CSIterator: An iterator object for the ComplexStructure.
        
        Why:
            Iterating through columns allows the evolutionary algorithm to
            explore different combinations and transformations of features,
            which is crucial for discovering the underlying differential
            equations.
        """
        return CSIterator(self)

    def hash_descr(self):
        """
        Returns a hashable description of the complex structure.
        
        This method creates a tuple of hashable descriptions, one for each term
        within the structure. This allows for efficient comparison and
        identification of similar complex structures based on their constituent
        terms.
        
        Args:
            self: The ComplexStructure instance.
        
        Returns:
            tuple: A tuple containing the hashable descriptions of the terms in the structure.
        """
        return tuple([term.hash_descr for term in self.structure])

    def set_evaluator(self, evaluator):
        """
        Sets the evaluator for the evolutionary algorithm.
        
        This method is no longer used directly. The evaluator is now configured
        within the evolutionary operator to ensure that the equation discovery
        process is tightly integrated with the search strategy. This change
        promotes a more modular and flexible design, allowing for easier
        experimentation with different evolutionary approaches.
        
        Args:
            evaluator: The evaluator to be set (not used).
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: Always raised, as the functionality has been moved.
        """
        raise NotImplementedError(
            'Functionality of this method has been moved to the evolutionary operator declaration')

    def evaluate(self, structural=False):
        """
        Evaluates the complex structure by recursively applying the interelement operator.
        
                This method orchestrates the evaluation of the entire complex structure.
                If the structure is composed of a single element, that element is evaluated directly.
                For multi-element structures, it iteratively applies the `interelement_operator`
                to the evaluated results of each element, effectively reducing the entire structure
                to a single representative value. This process mirrors how differential equations
                are constructed from simpler terms and operations.
        
                Args:
                    structural (bool): A flag indicating whether structural evaluation should be performed
                        on the individual elements within the complex structure.
        
                Returns:
                    The result of evaluating the complex structure, representing the overall value
                    obtained after applying the interelement operator across all elements.
        
                Raises:
                    ValueError: If the operands within the structure cannot be combined due to
                        incompatible shapes or types, indicating a potential issue with the
                        structure's composition.
                    AssertionError: If the complex structure is empty, as there are no elements
                        to evaluate or combine.
        """
        assert len(self.structure) > 0, 'Attempt to evaluate an empty complex structure'
        if len(self.structure) == 1:
            return self.structure[0].evaluate(structural)
        else:
            try:
                return reduce(lambda x, y: self.interelement_operator(x, y.evaluate(structural)),
                              self.structure[1:], self.structure[0].evaluate(structural))
            except ValueError:
                print([element.name for element in self.structure])
                raise ValueError('operands could not be broadcast together with shapes')

    def reset_saved_state(self):
        """
        Resets the saved state of the object and its constituent elements.
        
                This method ensures that the object and its internal structure are marked as not yet saved.
                It achieves this by resetting the `saved` and `saved_as` attributes to their initial, unsaved states.
                The method recursively calls itself on each element within the object's structure to propagate
                this reset throughout the entire data representation. This is crucial to ensure consistency when
                re-evaluating or re-fitting the equation discovery process, preventing the use of potentially outdated
                saved states.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    None.
        
                Class Fields (initialized or modified):
                    saved (dict): A dictionary tracking whether the object has been saved
                        in different states (True/False). Initialized to {True: False, False: False}.
                    saved_as (dict): A dictionary storing the file paths where the object
                        has been saved in different states (True/False). Initialized to {True: None, False: None}.
        """
        self.saved = {True: False, False: False}
        self.saved_as = {True: None, False: None}
        for elem in self.structure:
            elem.reset_saved_state()

    @property
    def name(self):
        """
        Gets the name of the complex structure.
        
                This name serves as an identifier for the structure within the equation discovery process, 
                allowing for easy referencing and organization of different structural components.
        
                Returns:
                    str: The name of the complex structure.
        """
        pass


class CSIterator(object):
    """
    An iterator for a case-insensitive sequence.
    
        This class wraps an existing sequence and provides an iterator that
        returns each element of the sequence in lowercase.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, complex_structure: ComplexStructure):
        """
        Initializes the iterator with a complex structure.
        
        This constructor prepares the iterator to traverse the given complex structure,
        starting from the initial element. The index is set to zero to begin iteration
        from the beginning of the structure.
        
        Args:
            complex_structure (ComplexStructure): The complex structure to be iterated over.
        
        Fields:
            _idx (int): The current index of the iterator, initialized to 0.
            _complex_structure (ComplexStructure): The complex structure being iterated.
        
        Returns:
            None.
        
        Why:
            The iterator needs to be initialized with the data structure it will traverse
            and a starting index to keep track of the current position during iteration,
            which is essential for systematically accessing elements within the structure.
        """
        self._idx = 0
        self._complex_structure = complex_structure

    def __next__(self):
        """
        Returns the next part of the equation's structure.
        
        This method enables sequential access to the components
        of the complex equation structure, which is essential for
        iterating through and processing the equation's elements
        during the evolutionary search.
        
        Args:
            self: The object instance.
        
        Returns:
            The next element in the equation's structure.
        
        Raises:
            StopIteration: If the end of the structure is reached,
                           indicating that all parts of the equation
                           have been processed.
        """
        if self._idx < len(self._complex_structure.structure):
            res = self._complex_structure.structure[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration
