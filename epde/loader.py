#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:27:53 2023

@author: maslyaev
"""

import sys
import os

import tempfile
import dill as pickle
import time

from typing import Union
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from epde.structure.factor import Factor
from epde.structure.main_structures import SoEq, Equation, Term 
from epde.interface.token_family import TFPool
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.single_criterion.optimizer import Population
from epde.cache.cache import Cache

TYPES = {'SoEq' : SoEq, 'TFPool' : TFPool, 'cache' : Cache,
         'ParetoLevels' : ParetoLevels, 'Population' : Population}    

# In TYPESPEC_ATTRS the first element of the value tuple is list of attributes not to be pickled
# while the second represents attributes for manual reconstruction.
TYPESPEC_ATTRS = {'SoEq' : (['tokens_for_eq', 'tokens_supp', 'latex_form'], ['vals']), 
                  'Factor' : (['latex_form', '_ann_repr', '_latex_constructor', '_evaluator'], []), 
                  'Equation' : (['pool', 'latex_form', '_history'], ['structure']), # , '_features', '_target'
                  'Term' : (['pool', 'latex_form'], ['structure']), 'TFPool' : ([], []), 'cache' : ([], []), 
                  'ParetoLevels' : (['levels'], ['population']), 'Population' : ([], [])}
                  

# LOADING_PRESETS = {'SoEq' : {'SoEq' : []}}

def get_size(obj, seen=None):
    """
    Get a recursive size of an object in bytes.
    
        This function recursively calculates the size of an object to estimate the memory footprint of data structures used within the equation discovery process. By traversing attributes, items, and slots, it provides a comprehensive size calculation, which is crucial for memory management and performance optimization when dealing with complex equation structures and large datasets. It handles dictionaries, objects with `__dict__` attributes, iterable objects, and objects with `__slots__`. It avoids infinite recursion by keeping track of already visited objects.
    
        Args:
            obj: The object to get the size of.
            seen: A set to keep track of already visited objects (default: None).
    
        Returns:
            The size of the object in bytes.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    try:
        slots = obj.__slots__
    except AttributeError:
        slots = []
    for slot in slots:
        size += get_size(getattr(obj, slot), seen)
    return size

def parse_obj_type(obj):
    """
    Parses an object to extract and return its type as a string.
    
    This function is used to determine the type of a given object, which is then utilized to represent it as a symbolic token within the equation discovery process.
    
    Args:
        obj: The object whose type needs to be determined.
    
    Returns:
        str: The type of the object as a string, extracted from the object's type information.
    """
    return str(type(obj)).split('.')[-1][:-2]
    
def get_typespec_attrs(obj):
    """
    Retrieves the type specification attributes for a given object.
    
    This function is used to access pre-defined attributes associated with the type of the input object.
    These attributes are essential for defining search space of equation structures.
    
    Args:
        obj: The object to retrieve the type specification attributes from.
    
    Returns:
        The type specification attributes associated with the object's type.
        These attributes are stored in a dictionary `TYPESPEC_ATTRS` under the key,
        which is a string representation of the object's type.
    """
    key = parse_obj_type(obj)   # Get key of the attribute, strapping extra symbols from type()
    return TYPESPEC_ATTRS[key]

def obj_to_pickle(obj, not_to_pickle: list = [], manual_pickle: list = []):
    """
    Template method for creating a dictionary representation of an object for persistence.  This method should be incorporated into classes that require custom serialization logic. It allows selective and recursive conversion of object attributes.
    
        Args:
            obj: The object to be serialized.
            not_to_pickle (list of str, optional): List of attribute names to exclude from serialization. Defaults to [].
            manual_pickle (list of str, optional): List of attribute names to serialize using a custom `obj_to_pickle` call on their values. Defaults to [].
    
        Returns:
            dict: A dictionary containing the serialized representation of the object.
    
        Why:
        This method facilitates the conversion of complex objects into a dictionary format suitable for saving or transmitting. It provides control over which attributes are included and how they are serialized, especially useful for handling nested objects or those requiring special serialization procedures.
    """
    ntp_add, mp_add = get_typespec_attrs(obj)
    not_to_pickle = ntp_add
    manual_pickle = mp_add
    
    dict_to_pickle = {'obj_type' : parse_obj_type(obj)}
    
    for key, elem in obj.__dict__.items():
        if key in not_to_pickle:
            continue
        elif key in manual_pickle:
            if isinstance(elem, dict):
                
                dict_to_pickle[key] = {'obj_type' : dict, 'keys' : [ekey for ekey in elem.keys()],
                                       'elements' : [obj_to_pickle(val, get_typespec_attrs(val)[0], get_typespec_attrs(val)[1])
                                                     for val in elem.values()]}
            elif isinstance(elem, Iterable):
                
                dict_to_pickle[key] = {'obj_type' : type(elem), 'elements' : [obj_to_pickle(list_elem,
                                                                                            get_typespec_attrs(list_elem)[0],
                                                                                            get_typespec_attrs(list_elem)[1])
                                                                              for list_elem in elem]}
            else:
                dict_to_pickle[key] = {'obj_type' : type(elem), 'elements' : obj_to_pickle(elem, get_typespec_attrs(elem)[0], 
                                                                                           get_typespec_attrs(elem)[1])}
        else:
            dict_to_pickle[key] = elem

    try:
        slots = obj.__slots__
    except AttributeError:
        slots = []

    for slot in slots:
        elem = getattr(obj, slot)
        if slot in not_to_pickle:
            continue
        elif slot in manual_pickle:
            if isinstance(elem, dict):
                dict_to_pickle[slot] = {'obj_type' : dict, 'keys' : [ekey for ekey in elem.keys()],
                                        'elements' : [obj_to_pickle(val, get_typespec_attrs(val)[0], get_typespec_attrs(val)[1])
                                                      for val in elem.values()]}
            elif isinstance(elem, Iterable):
                dict_to_pickle[slot] = {'obj_type' : type(elem), 'elements' : [obj_to_pickle(list_elem,
                                                                                             get_typespec_attrs(list_elem)[0],
                                                                                             get_typespec_attrs(list_elem)[1])
                                                                               for list_elem in elem]}
            else:
                dict_to_pickle[slot] = {'obj_type' : type(elem), 'elements' : obj_to_pickle(elem, get_typespec_attrs(elem)[0], 
                                                                                            get_typespec_attrs(elem)[1])}
        else:
            dict_to_pickle[slot] = elem
    
    return dict_to_pickle

def attrs_from_dict(obj, attributes, except_attrs: dict = {}):
    """
    Populates an object's attributes from a dictionary, with special handling for reconstruction and exceptions.
    
        This method sets an object's attributes based on a provided dictionary,
        accommodating specific reconstruction procedures and attribute exclusions.
        It ensures that the object is correctly initialized with the data necessary
        for equation discovery.
    
        Args:
            obj: The object to populate with attributes.
            attributes: A dictionary containing the attributes to set on the object.
            except_attrs: A dictionary specifying attributes to exclude or handle differently.
    
        Returns:
            None. This method modifies the object in place.
    """
    except_dict = except_attrs[parse_obj_type(obj)]
    if 'obj_type' not in except_dict.keys():
        except_dict['obj_type'] = None
    manual_reconstr = get_typespec_attrs(obj)[1]
    
    try:
        slots = obj.__slots__
        for slot in slots:
            if slot not in manual_reconstr:
                setattr(obj, slot, attributes[slot]) if slot in attributes.keys() else setattr(obj, slot, except_dict[slot])
    except AttributeError:
        slots = []
    
    obj.__dict__ = {key : item for key, item in attributes.items()
                    if key not in except_attrs.keys() and key not in slots and key not in manual_reconstr}
    for key, elem in except_attrs.items():
        if elem is not None and key not in slots and key not in manual_reconstr:
            obj.__dict__[key] = elem
    
    for man_attr in manual_reconstr:
        if man_attr not in attributes.keys():
            raise AttributeError(f'Object {man_attr} for reconstruction is missing from attributes dictionary.')

        obj.manual_reconst(man_attr, attributes[man_attr]['elements'], except_attrs)


def temp_pickle_save(obj : Union[SoEq, Cache, TFPool, ParetoLevels, Population], 
                     not_to_pickle = [], manual_pickle = []):
    """
    Saves a complex object to a temporary pickle file for efficient storage and retrieval during the equation discovery process.
    
        This method serializes the given object using the EPDELoader and saves it
        to a temporary file. This is useful for caching intermediate results or
        transferring complex data structures between different stages of the
        equation discovery workflow.
    
        Args:
            obj (Union[SoEq, Cache, TFPool, ParetoLevels, Population]): The object to be pickled and saved.
            not_to_pickle (list, optional): A list of attributes to exclude from pickling. Defaults to [].
            manual_pickle (list, optional): A list of attributes to pickle manually. Defaults to [].
    
        Returns:
            tempfile._TemporaryFileWrapper: The temporary file object containing the pickled data.
    """
    loader = EPDELoader()
    pickled_obj = loader.saves(obj, not_to_pickle = [], manual_pickle = [])
    
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.write(pickled_obj)
    return temp_file


class LoaderAssistant(object):
    """
    Provides preset configurations for various data structures and components.
    
        This class offers a collection of methods that return preset dictionaries
        tailored for initializing different aspects of a system, such as TFPools,
        caches, population data structures, and Pareto levels. These presets
        provide default configurations and initial states for these components.
    
        Class Methods:
        - system_preset
        - pool_preset
        - cache_preset
        - population_preset
        - pareto_levels_preset
    """

    def __init__(self):
        """
        Initializes the LoaderAssistant.
        
        This class manages the loading and preprocessing of data required for the equation discovery process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Why:
            The LoaderAssistant is initialized to prepare the data pipeline, ensuring that the data is in the correct format and structure for subsequent equation discovery steps.
        """
        pass
    
    @staticmethod
    def system_preset(pool: TFPool): # Validate correctness of attribute definitions
        return {'SoEq' :     {'tokens_for_eq' : TFPool(pool.families_demand_equation),
                              'tokens_supp' : TFPool(pool.families_equationless), 
                              'latex_form' : None},
                'Equation' : {'pool' : pool, 
                              'latex_form' : None,
                              '_history' : None},
                'Term'     : {'pool' : pool, 
                              'latex_form' : None},
                'Factor'   : {'_latex_constructor' : None, 
                              '_evaluator' : None}} 
    
    @staticmethod
    def pool_preset():
        """
        Returns a preset dictionary for a TFPool.
        
                This provides a standardized, empty TFPool configuration,
                ensuring consistency when initializing or resetting equation search spaces.
        
                Returns:
                    dict: A dictionary containing an empty TFPool preset.
        """
        return {'TFPool' : {}}
    
    @staticmethod
    def cache_preset():
        """
        Caches a preset, ensuring that default configurations are readily available for equation discovery. This improves efficiency by avoiding redundant computations when exploring the solution space.
        
                Args:
                    None
        
                Returns:
                    dict: An empty dictionary representing the cached preset. This can be extended in the future to hold actual preset configurations.
        """
        return {}
    
    @staticmethod
    def population_preset(pool: TFPool):
        """
        Generates a preset configuration for initializing the population of symbolic expressions.
        
                This method creates a dictionary containing configurations for
                various components of the population, such as 'Population', 'SoEq',
                'Equation', 'Term', and 'Factor'. These configurations define the
                initial state and available operations for the evolutionary search
                of differential equations. It sets up the initial pools of tokens
                and other necessary parameters for the search process.
        
                Args:
                  pool: A TFPool object containing the initial set of tokens
                    (functions and variables) to be used in constructing equations.
        
                Returns:
                  dict: A dictionary containing preset configurations for the
                    population. The dictionary has the following structure:
                    {
                      'Population': {},
                      'SoEq': {
                        'tokens_for_eq': TFPool, initialized with pool.families_demand_equation,
                        'tokens_supp': TFPool, initialized with pool.families_equationless,
                        'latex_form': None
                      },
                      'Equation': {
                        'pool': TFPool, the input pool,
                        'latex_form': None,
                        '_history': None
                      },
                      'Term': {
                        'pool': TFPool, the input pool,
                        'latex_form': None
                      },
                      'Factor': {
                        '_latex_constructor': None,
                        '_evaluator': None
                      }
                    }
        
                Why:
                    This preset configuration is crucial for setting up the initial
                    conditions and available building blocks for the evolutionary
                    search of differential equations. It defines the search space
                    and ensures that the algorithm starts with a valid and
                    well-defined population.
        """
        return {'Population' : {}, 
                'SoEq' :     {'tokens_for_eq' : TFPool(pool.families_demand_equation),
                              'tokens_supp' : TFPool(pool.families_equationless), 
                              'latex_form' : None},
                'Equation' : {'pool' : pool, 
                              'latex_form' : None,
                              '_history' : None},
                'Term'     : {'pool' : pool, 
                              'latex_form' : None},
                'Factor'   : {'_latex_constructor' : None, 
                              '_evaluator' : None}}
    
    @staticmethod
    def pareto_levels_preset(pool: TFPool):
        """
        Returns a preset configuration for equation discovery components.
        
                This method configures the search space for the evolutionary algorithm
                by defining settings for Pareto levels, symbolic equations (SoEq),
                equations, terms, and factors. It initializes these components with
                specific settings and associations, using a TFPool object to manage
                the available equation and term building blocks. This ensures that the
                search process starts with a well-defined and consistent set of
                components, facilitating the discovery of meaningful equations.
        
                Args:
                    pool: A TFPool object containing the token families used for
                        constructing equations and terms.
        
                Returns:
                    dict: A dictionary containing preset configurations for
                        'ParetoLevels', 'SoEq', 'Equation', 'Term', and 'Factor'. Each
                        key maps to a dictionary of settings specific to that component,
                        defining the initial state of the equation discovery process.
        """
        return {'ParetoLevels' : {}, 
                'SoEq' :     {'tokens_for_eq' : TFPool(pool.families_demand_equation),
                              'tokens_supp' : TFPool(pool.families_equationless), 
                              'latex_form' : None},
                'Equation' : {'pool' : pool, 
                              'latex_form' : None,
                              '_history' : None},
                'Term'     : {'pool' : pool, 
                              'latex_form' : None},
                'Factor'   : {'_latex_constructor' : None, 
                              '_evaluator' : None}}
        

class EPDELoader(object):
    '''
    Universal loader for EPDE objects, applicable to system of equations as 
        ``SoEq``, token families pool as ``TFPool``, populations as
    '''

    
    def __init__(self, directory = None):
        """
        Initializes the EPDELoader, setting up the designated directory for storing intermediate results and models.
        
                The directory is crucial for preserving progress and enabling efficient re-evaluation of solutions during the equation discovery process. If a custom directory is not provided, a default directory named 'epde_cache' is created two levels above the current working directory to keep the project workspace organized.
        
                Args:
                    directory (str, optional): The path to the directory for storing cached data. Defaults to None, which triggers the use of the default directory.
        
                Returns:
                    None
        
                Raises:
                    TypeError: If the provided directory is not a string or if the path is invalid and the directory cannot be created.
        
                Fields:
                    _directory (str): The directory used for caching objects.
        """
        if directory is not None:
            if not isinstance(directory, str):
                raise TypeError(f'Incorrect format of repo to save objects, expected str, got {type(directory)}.')

            if not os.path.isdir(directory):
                try:
                    os.mkdir(path=directory)
                except FileNotFoundError:
                    raise TypeError(f'Wrong path passed, can not create a directory with path {directory}')

            self._directory = directory
        else:
            self._directory = os.path.normpath((os.path.join(os.path.dirname(os.getcwd()), 
                                                             '..','epde_cache')))
    
    def save(self, obj, filename:str = None, not_to_pickle:list = [], manual_pickle:list = []):
        """
        Saves a discovered equation or model to a file, preserving its structure and parameters for later use. This allows for persistence of found solutions and their subsequent loading and application.
        
                Args:
                    obj: The equation or model object to be saved.
                    filename: The name of the file to save the object to.
                    not_to_pickle: A list of attribute names to exclude from pickling.
                    manual_pickle: A list of attribute names to manually handle during pickling.
        
                Returns:
                    None.
        """
        with open(filename, mode = 'wb') as file:
            pickle.dump(obj_to_pickle(obj, not_to_pickle, manual_pickle), file)
    
    def saves(self, obj, not_to_pickle:list = [], manual_pickle:list = []):
        """
        Serializes a given object into a pickle string, preparing it for storage or transfer within the EPDE framework.
        
                This method converts an object into a byte stream suitable for persisting its state.
                It provides mechanisms to exclude certain attributes from serialization and handle others with custom logic,
                ensuring compatibility and efficiency when reconstructing the object later.
                This is important because some objects might contain unserializable attributes or require special handling to be correctly restored.
        
                Args:
                    obj: The object to be serialized.
                    not_to_pickle: A list of attribute names that should not be included in the serialized data.
                    manual_pickle: A list of attribute names that require custom serialization logic.
        
                Returns:
                    bytes: A pickle string representing the serialized object.
        """
        pickling_form = obj_to_pickle(obj, not_to_pickle, manual_pickle)
        return pickle.dumps(pickling_form)        

    def use_pickles(self, obj_pickled, **kwargs):
        """
        Uses a previously serialized representation to reconstruct an object.
        
                This method leverages stored object data to instantiate and initialize a new object instance,
                bypassing the standard initialization process. This is useful for efficiently restoring
                complex objects with pre-computed states, ensuring consistency and speed in object creation.
        
                Args:
                    obj_pickled (dict): A dictionary containing the serialized object's data, including its type and attributes.
                    **kwargs: Keyword arguments specifying attributes to exclude from being loaded from the serialized data.
        
                Returns:
                    object: The newly created and initialized object.
        """
        obj = TYPES[obj_pickled['obj_type']].__new__(TYPES[obj_pickled['obj_type']])
        attrs_from_dict(obj, obj_pickled, except_attrs = kwargs)
        return obj
    
    def load(self, filename:str, **kwargs):
        """
        Loads a previously serialized equation candidate from a file.
        
                This method retrieves a pickled equation candidate from the specified file
                and prepares it for further evaluation within the EPDE framework. The loaded
                object is then passed to the `use_pickles` method for processing and
                integration into the equation discovery workflow.
        
                Args:
                    filename: The name of the file containing the pickled equation candidate.
                    **kwargs: Keyword arguments to pass to the `use_pickles` method, allowing
                        for customization of the candidate's evaluation.
        
                Returns:
                    The result of calling `use_pickles` with the unpickled equation candidate
                    and any provided keyword arguments. This typically involves further
                    processing or evaluation of the candidate within the EPDE framework.
        """
        with open(filename, mode = 'rb') as file:
            obj_pickled = pickle.load(file)
        return self.use_pickles(obj_pickled, **kwargs)
    
    def loads(self, byteobj, **kwargs):
        """
        Deserialize a pickled object from a byte string and prepare it for equation discovery.
        
                This method deserializes a Python object (typically representing data or
                initial conditions) from a byte string using the `pickle` module. The
                deserialized object is then passed to the `use_pickles` method for
                further processing, ensuring it's in a suitable format for the
                equation discovery process. This step is crucial for loading pre-processed
                data or configurations into the EPDE framework.
        
                Args:
                    byteobj: The byte string representing the pickled object.
                    **kwargs: Additional keyword arguments to pass to `use_pickles`.
        
                Returns:
                    The deserialized and processed Python object, ready for use in
                    equation discovery.
        """
        obj_pickled = pickle.loads(byteobj)
        return self.use_pickles(obj_pickled, **kwargs)