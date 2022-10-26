'''

Superclasses and examples of objects, that should be used in the optimization procedures, 
using moeadd algorithm.

Contains:
------------

**moe_population_constructor** : superclass for population constructor object, dedicated 
to the creation of initial population for the evolutionary algorithm.

**moe_evolutionary_operator** : superclass for evolutionary operator, dedicated to altering the 
population during evolutionary search. Specific implementations must contain both mutation and 
crossover suboperators.

**gaussian_mutation** : an example of the mutation suboperator for problems, where 
genes candidate genes are represented by flowting point strings, that operates by addition 
of an increment, randomly selected from the normal distribution. 

**mixing_xover** : crossover for problems, involving candidate solutions with genes, 
represented by strings of floating point values. The offspring genes' elements are selected in 
the interval between the parents' gene values.

**plot_pareto** : supplementary function to plot the candidate solutions of 2D-problem 
in form of non-dominated levels, using matplotlib tools.

'''

import numpy as np
from copy import deepcopy
from abc import ABC, abstractproperty, abstractmethod
import matplotlib.pyplot as plt

from epde.moeadd.moeadd_strategy import *
from epde.moeadd.moeadd import *
from epde.moeadd.moeadd_supplementary import *


class MOEPopulationConstructor(ABC):
    '''
    
    Abstract class of the creator of new moeadd solutions, utilized in its initialization phase. 
    Shall be overwritten to be properly used for each specific case.
    
    Methods:
    ---------
    
    __init__(*args) 
        In the __init__ method, you should be able to set the parameters of the constructor.
        
    create(*creation_args)
        Method, dedicated to the creation (oftenly randmized) of new candidate solutions.
        
    Example of the subclass:
    ------------------------
    
        >>> class test_population_constructor(object):
        >>>     def __init__(self, bitstring_len = 2, vals_range = [-4, 4]):
        >>>         self.bs_len = bitstring_len; self.vals_range = vals_range
        >>>         
        >>>     def create(self, *args):
        >>>         created_solution = solution_array(x = np.random.uniform(low = self.vals_range[0], 
        >>>                                                                 high = self.vals_range[1], 
        >>>                                                                 size = self.bs_len), 
        >>>                              obj_funs=[optimized_fun_1, optimized_fun_2])
        >>>         return created_solution        

    '''
    def __init__(self, *constr_args):
        pass
        
    @abstractmethod
    def create(self, *creation_args):
        return None


class MOEEvolutionaryOperator(ABC):
    '''
    
    Abstract class of the moeadd evolutionary operator. The subclass implementations shall
    have mutation and crossover methods, that produce correspondingly one new solution and 
    a list of new solutions;
    
    Methods:
    -------
        
    mutation(solution)
        return a new solution, created by alteration of an existing one
        
    crossover(parents_pool)
        returns a list of new solutions, created from the parents pool. Parents pool already 
        constains the selected individuals, therefore, no new selection required.
    
    '''
    def __init__(self, xover_lambda, mut_lambda):
        pass

    @abstractmethod
    def mutation(self, solution):
        return None

    @abstractmethod
    def crossover(self, parents_pool):
        return [None for parent in parents_pool]
    

def gaussian_mutation(solution_x):
    '''
        Basic Gaussian mutation, that can be used inside the moeadd evolutionary operator, 
        when it works with string of real values. More complicated ones can be declared in its image. 
        
        Arguments:
        ---------
        
        solution_x : np.array, 
            values (genotype) of the moeadd solution. Represented by the *moeadd_solution.vals* attribute or the 
            same attribute of its subclass object.        
            
    '''
    return solution_x + np.random.normal(size = solution_x.size)


def mixing_xover(parents):
    '''
        Basic crossover operator, that can be used inside the moeadd evolutionary operator, 
        when it works with string of real values. More complicated ones can be declared in its image. 
        
        Arguments:
        ---------
        
        parents : list of 2 moeadd_solution, or its subclass objects, 
            parent solutions of the many-objective optimization algorithm.

        Returns:
        --------
        
        offsprings : list of 2 moeadd_solution, or its subclass objects,
            offspring solutions of the many-objective optimization algorithm, 
            with values, creating in the interval between their parent ones.            

    '''

    proportion = np.random.uniform(low = 1e-6, high = 0.5-1e-6)
    offsprings = [deepcopy(parent) for parent in parents]
    offsprings[0].precomputed_value = False; offsprings[1].precomputed_value = False
    offsprings[0].precomputed_domain = False; offsprings[1].precomputed_domain = False

    offsprings[0].vals = parents[0].vals + proportion * (parents[1].vals - parents[0].vals)
    offsprings[1].vals = parents[0].vals + (1 - proportion) * (parents[1].vals - parents[0].vals)
    return offsprings


def plot_pareto(levels, weights = None, max_level = None, logscale = (False, False)):
    '''
    
    Vizualization method to demonstrate the pareto levels of 2D-problem on the plane via matplotlib
    
    Arguments:
    ----------
    
    levels : src.moeadd.pareto_levels obj
        object, containing pareto levels. Ususally obtained from *src.moeadd.moeadd_optimizer* attribute *src.moeadd.moeadd_optimizer.pareto_levels*.
        
    weights : np.ndarray or None, optional
        Contains weights from the moeadd algorithm to be visualized. If None, no weights 
        vectors will be visualized. 
    
    max_level : int or None, optonal, default None
        Number of layers to be visualized during on the plot. If None, all
        layers will be shown.
        
    logscale : tuple of 2 bool values, optional, default ``(False, False)``
        Indication, of will the axis use logscale on the plot, if all values are positive. 
        First element of tuple - if the x-axis will use logscale. Second - y-scale.
        
    '''
    if max_level is None:
        max_level = np.inf
    else:
        max_level += 1
    assert levels.population[0].obj_fun.size == 2
    coords = [[(solution.obj_fun[1], solution.obj_fun[0]) for solution in levels.levels[front_idx]] for front_idx in np.arange(min((len(levels.levels), max_level)))]
    coords_arrays = []
    for coord_set in coords:
        coords_arrays.append(np.array(coord_set))
    coords_arrays
    colors = ['r', 'k', 'b', 'y', 'g'] + ['m' for idx in np.arange(len(coords_arrays) - 5)]
    
    obj_funs_min = np.array([min([sol.obj_fun[ofi] for sol in levels.population]) #Некрасиво
                        for ofi in range(levels.population[0].obj_fun.size)])
    obj_funs_min = np.flip(obj_funs_min)
    obj_funs_max = np.array([max([sol.obj_fun[ofi] for sol in levels.population]) 
                        for ofi in range(levels.population[0].obj_fun.size)])
    obj_funs_max = np.flip(obj_funs_max)
    
    boundaries_min = obj_funs_min - 0.5*obj_funs_min
    for bound_idx, bound_min in enumerate(boundaries_min):
        if bound_min == 0: boundaries_min[bound_idx] -= 0.5
    boundaries_max = obj_funs_max + 0.1*(obj_funs_max - obj_funs_min)
    
    
    fig, ax = plt.subplots()
    for front_idx in np.arange(len(coords_arrays)):
        ax.scatter(coords_arrays[front_idx][:, 0], coords_arrays[front_idx][:, 1], color = colors[front_idx], s = 20)
        if boundaries_min[0] > 0 and logscale[0]:
            ax.set_xscale('log')
        if boundaries_min[1] > 0 and logscale[1]:
            ax.set_yscale('log')
        
        plt.xlim((boundaries_min[0], boundaries_max[0]))
        plt.ylim((boundaries_min[1], boundaries_max[1]))

    plt.grid()
    if not (weights is None):
        for weight_idx in np.arange(weights.shape[0]):
            vector_coors = weights[weight_idx, :]
            ax.plot([0, vector_coors[0]], [0, vector_coors[1]], color = 'k')
    fig.show()