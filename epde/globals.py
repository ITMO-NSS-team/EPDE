#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from dataclasses import dataclass
import warnings

from epde.cache.cache import Cache
from symnet import pool_terms


def init_caches(set_grids: bool = False):
    """
    Initialization global variables for keeping input data, values of grid and useful tensors such as evaluated terms

    Args:
        set_grids (`bool`): flag about using grid data

    Returns:
        None
    """
    global tensor_cache, grid_cache, initial_data_cache
    tensor_cache = Cache()
    initial_data_cache = Cache()
    if set_grids:
        grid_cache = Cache()
    else:
        grid_cache = None


def set_time_axis(axis: int):
    """
    Setting global of time axis
    """
    global time_axis
    time_axis = axis


def init_eq_search_operator(operator):
    global eq_search_operator
    eq_search_operator = operator


def init_sys_search_operator(operator):
    global sys_search_operator
    sys_search_operator = operator


def delete_cache():
    global tensor_cache, grid_cache
    try:
        del tensor_cache
    except NameError:
        print('Failed to delete tensor cache due to its inexistance')
    try:
        del grid_cache
    except NameError:
        print('Failed to delete grid cache due to its inexistance')


@dataclass
class VerboseManager:
    """
    Manager for output in text form
    """
    plot_DE_solutions : bool
    show_iter_idx : bool
    iter_fitness : bool
    iter_stats : bool
    show_warnings : bool
    
def init_verbose(plot_DE_solutions : bool = False, show_iter_idx : bool = True, 
                 show_iter_fitness : bool = False, show_iter_stats : bool = False, 
                 show_warnings : bool = False):
    """
    Method for initialized of manager for output in text form

    Args:
        plot_DE_solutions (`bool`): optional 
            display solutions of a differential equation, default - False
        show_iter_idx (`bool`): optional
            display the index of each iteration EA, default - False
        show_iter_fitness (`bool`): optional
            display the fitness of each iteration EA, default - False
        show_iter_stats (`bool`): optional
            display statistical properties of the population in each iteration EA, default - False
        show_warnings (`bool`): optional
            display warnings arising during the operation of the algorithm, default - False
    """
    global verbose
    if not show_warnings:
        warnings.filterwarnings("ignore")
    verbose = VerboseManager(plot_DE_solutions, show_iter_idx, show_iter_fitness, 
                             show_iter_stats, show_warnings)


def init_symnet_pool(max_factors_in_term, families, u, derivs, shape, names, grids, max_deriv_order, mmf):
    global sympool
    sympool = pool_terms.PoolTerms(max_factors_in_term, families, mmf=mmf)
    sympool.set_initial_distr(u, derivs, shape, names, families=families, grids=grids, max_deriv_order=max_deriv_order)
    # temporary_cf = [0.3, 1.5, 0.3, 0.3, 1.5, 0.3, 1.5, 0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,]
    # for i, key in enumerate(sympool.pool_dict.keys()):
    #     sympool.pool_dict[key] = temporary_cf[i]
