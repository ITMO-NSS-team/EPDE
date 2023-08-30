"""

Main classes and functions of the moeadd optimizer.

"""

import numpy as np
import time
import warnings

from typing import Union
from copy import deepcopy
from functools import reduce

import epde.globals as global_var
from epde.optimizers.moeadd.population_constr import SystemsPopulationConstructor
from epde.interface.logger import Logger

from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
from epde.optimizers.moeadd.supplementary import fast_non_dominated_sorting, ndl_update, Equality, Inequality
from scipy.spatial import ConvexHull    

def clear_list_of_lists(inp_list) -> list:
    '''
    Delete elements-lists with len(0) from the list
    '''
    return [elem for elem in inp_list if len(elem) > 0]


class ParetoLevels(object):
    '''
    
    The representation of Pareto levels, comprised of a finite number of objects in the 
    objective function space. Introduced to be used in methods of the moeadd.optimizer class
    
    Attributes:    
        population (`list`): List with the elements - canidate solutions of the case-specific subclass of 
            ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
        levels (`list`): List with the elements - lists of solutions, representing non-dominated levels. 
            The 0-th element - the current Pareto frontier.
        unplaced_candidates (`list`): candidates, that dont using in structure
        _sorting_method (`callable`): The method of population separation into non-dominated levels.
        _update_method (`callable`): The method of point addition into the population and onto the non-dominated levels.
    
    Notes:
        The initialization of objects of this class is held automatically in the __init__ of 
        moeadd optimizer, thus no extra interactions of a user with this class are necessary.
    
    '''
    def __init__(self, population, sorting_method = fast_non_dominated_sorting, 
                 update_method = ndl_update, initial_sort : bool = False):
        """
        Args:
            population (`list`): List with the elements - canidate solutions of the case-specific subclass of 
                ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
            sorting_method (`callable`): optional, default - ``src.moeadd.moeadd_supplementary.fast_non_dominated_sorting``
                The method of population separation into non-dominated levels
            update_method (`callable`): optional, defalut - ``src.moeadd.moeadd_supplementary.ndl_update``
                The method of point addition into the population and onto the non-dominated levels.
        """
        self._sorting_method = sorting_method
        self.population = [] #population
        self._update_method = update_method
        self.unplaced_candidates = population # tabulation deleted
        
    @property
    def levels(self):
        return self._levels     #sort(self.population)
    
    @levels.setter
    def levels(self, value : list):
        self._levels = value
    
    def __len__(self):
        return len(self.population)
    
    def __iter__(self):
        return ParetoLevelsIterator(self)
    
    def initial_placing(self):
        """
        Method for adding candidates into the structure who were not previously in it
        """
        while self._unplaced_candidates:
            self.population.append(self._unplaced_candidates.pop())
        if any([any([candidate == other_candidate for other_candidate in self.population[:idx] + self.population[idx+1:]])
                for idx, candidate in enumerate(self.population)]):
            print([candidate.text_form for candidate in self.population])
            raise Exception('Duplicating initial candidates')
        self.levels = self.sort()

    def sort(self):
        """
        Sorting of the population into Pareto non-dominated levels.
        """
        # self.levels = self._sorting_method(self.population)
        return self._sorting_method(self.population)
    
    @property
    def unplaced_candidates(self):
        return self._unplaced_candidates
    
    @unplaced_candidates.setter
    def unplaced_candidates(self, candidates : Union[list, set, tuple]):
        
        # ADD EXTRA CHECKS IF NECESSARY
        self._unplaced_candidates = candidates
    
    def update(self, point):
        """
        Addition of a candidate solution point into the pareto levels and the population list.

        Args:
            point (`MOEADDSolution`): The point, added into the candidate solutions pool.

        Returns:
            None
        """
        # print(f'IN MOEADD UPDATE {point.text_form}')        
        self.levels = self._update_method(point, self.levels)
        self.population.append(point)
 
    def delete_point(self, point):
        """
        Deletion of a candidate solution point from the pareto levels and the population list.
        
        Args:
            point (`MOEADDSolution`): The point, removed from the candidate solutions pool.

        Returns:
            None
        """        
        new_levels = []
        for level in self.levels:
            temp = []
            for element in level:
                if element != point:
                    temp.append(element)
            if not len(temp) == 0:
                new_levels.append(temp)

        population_cleared = []

        for elem in self.population:
            if elem != point:
                population_cleared.append(elem)
                
        if len(population_cleared) != sum([len(level) for level in new_levels]):
            print(len(population_cleared), len(self.population), sum([len(level) for level in new_levels]))
            print('initial population', [solution.vals for solution in self.population], len([solution.vals for solution in self.population]), '\n')
            print('cleared population', [solution.vals for solution in population_cleared], len([solution.vals for solution in self.population]), '\n')
            print(point.vals)
            raise Exception('Deleted something extra')
        self.levels = new_levels
        self.population = population_cleared

    def get_stats(self):
        return np.array([[element.obj_fun for element in level] for level in self.levels])

    def fit_convex_hull(self):
        """

        """
        if len(self.levels) > 1:
            warnings.warn('Algorithm has not converged to a single Pareto level yet!')
        points = np.vstack([sol.obj_fun for sol in self.population])
        points = np.concatenate((points, np.max(points, axis = 0).reshape((1, -1))))
        points_unique = np.unique(points, axis = 0)
        
        self.hull = ConvexHull(points = points_unique, qhull_options='Qt')

    def get_by_complexity(self, complexity):
        """
        Method for getting solutions with choosing complexity

        Args:
            complexity (`int`): number indicating the complexity of the solution

        Returns:
            matching_solutions (`list`): solutions with input complexity
        """
        matching_solutions = [solution for solution in self.levels[0] 
                              if solution.matches_complexitiy(complexity)]
        return matching_solutions        

class ParetoLevelsIterator(object):
    """
    Class for iteration by object of Pareto Levels
    """
    def __init__(self, pareto_levels):
        self._levels = pareto_levels
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._levels.population):
            res = self._levels.population[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration


class MOEADDOptimizer(object):
    """
    Solving multiobjective optimization problem (minimizing set of functions) with an 
    evolutionary approach. In this class, the unconstrained variation of the problem is 
    considered.

    Attributes:
        abbreviated_search_executed (`boolean`): flag about executing abbreviated search
        weights (`np.ndarray`): Weight vectors, introduced to decompose the optimization problem into 
            several subproblems by dividing Pareto frontier into a numeber of sectors.
        pareto_levels (`ParetoLevels`): Pareto levels object, containing the population of candidate solution as a list of 
            points and as a list of levels.
        neighborhood_lists (`list`): keeping neighbours for each solution. stored as lists with all solutions, which are sorted by the owner of the `list` (first element) 
        best_obj (`np.array`): The best achievable values of objective functions. Should be introduced with 
            ``self.pass_best_objectives`` method.
        sector_processer (`MOEADDSectorProcesser`): keeping evolutionary process
    
    Example:
    --------
    
    >>> pop_constr = test_population_constructor()
    >>> optimizer = moeadd_optimizer(pop_constr, 40, 40, 
    >>>                              None, delta = 1/50., 
    >>>                              neighbors_number = 5)
    >>> operator = test_evolutionary_operator(mixing_xover,
    >>>                                      gaussian_mutation)
    >>> optimizer.set_evolutionary(operator=operator)
    >>> optimizer.pass_best_objectives(0, 0)
    >>> optimizer.optimize(simple_selector, 0.95, (4,), 100, 0.75)
    
    In that case, we solve the optimization problem with two objective functions. The population
    constructor is defined with the imported dummy class ``test_population_constructor``, 
    and evolutionary operator contains mutation and crossover suboperators.
    
    """
    def __init__(self, population_instruct, weights_num, pop_size, solution_params, delta, neighbors_number, 
                 nds_method = fast_non_dominated_sorting, ndl_update = ndl_update): # logger: Logger = None, 
        """
        Initialization of the evolutionary optimizer is done with the introduction of 
        initial population of candidate solutions, divided into Pareto non-dominated 
        levels (sets of solutions, such, as none of the solution of a level dominates 
        another on the same level), and creation of set of weights with a proximity list 
        defined for each of them.
        
        Args:
            pop_constructor (``):
            weights_num (`int`): Number of the weight vectors, dividing the objective function values space. Often, shall be same, as the population size.
            pop_size (`int`): The size of the candidate solution population.
            solution_params (`dict`): The dicitionary with the solution parameters, passed into each new created solution during the initialization.
            delta (`float`): parameter of uniform spacing between the weight vectors; *H = 1 / delta* should be integer - a number of divisions along an objective coordinate axis.
            neighbors_number (`int`): number of neighboring weight vectors to be considered during the operation of evolutionary operators as the "neighbors" of the processed sectors.
            nds_method (`callable`): default - ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
                Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article 
                *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
                vol. 6, no. 2, pp. 182–197, Apr. 2002.*
            ndl_update (`callable`): default - ``moeadd.moeadd_supplementary.ndl_update``
                Method of adding a new solution point into the objective functions space, introduced 
                to minimize the recalculation of the non-dominated levels for the entire population. 
                The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
                update approach for steady-state evolutionary multiobjective optimization,” 
                Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
                MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
        """
        assert weights_num == pop_size, 'Each individual in population has to correspond to a sector'
        self.abbreviated_search_executed = False
        soluton_creation_attempts_softmax = 10
        soluton_creation_attempts_hardmax = 100

        pop_constructor = SystemsPopulationConstructor(**population_instruct)
        
        assert type(solution_params) == type(None) or type(solution_params) == dict, 'The solution parameters, passed into population constructor must be in dictionary'
        population = []
        for solution_idx in range(pop_size):
            solution_gen_idx = 0
            while True:
                if type(solution_params) == type(None): solution_params = {}
                temp_solution = pop_constructor.create(**solution_params)
                #TODO: check domain belonging
                temp_solution.set_domain(solution_idx)
                if not np.any([temp_solution == solution for solution in population]):
                    population.append(temp_solution)
                    print(f'New solution accepted, confirmed {len(population)}/{pop_size} solutions.')
                    break
                if solution_gen_idx == soluton_creation_attempts_softmax and global_var.verbose.show_warnings:
                    print('solutions tried:', solution_gen_idx)
                    warnings.warn('Too many failed attempts to create unique solutions for multiobjective optimization. Change solution parameters to allow more diversity.')
                if solution_gen_idx == soluton_creation_attempts_hardmax:
                    self.abbreviated_search(population, sorting_method = nds_method, update_method = ndl_update)
                    return None
                solution_gen_idx += 1
        self.pareto_levels = ParetoLevels(population, sorting_method = nds_method, update_method = ndl_update,
                                          initial_sort = False)
        
        self.weights = []; weights_size = len(population[0].obj_funs) #np.empty((pop_size, len(optimized_functionals)))
        for weights_idx in range(weights_num):
            while True:
                temp_weights = self.weights_generation(weights_size, delta)
                if temp_weights not in self.weights:
                    self.weights.append(temp_weights)
                    break
                else:
                    print(temp_weights, self.weights) # Ошибка в задании obj_fun для системы уравнений
        self.weights = np.array(self.weights)

        self.neighborhood_lists = []
        for weights_idx in range(weights_num):
            self.neighborhood_lists.append([elem_idx for elem_idx, _ in sorted(
                    list(zip(np.arange(weights_num), [np.linalg.norm(self.weights[weights_idx, :] - self.weights[weights_idx_inner, :]) for weights_idx_inner in np.arange(weights_num)])), 
                    key = lambda pair: pair[1])][:neighbors_number+1]) # срез листа - задаёт регион "близости"

        self.best_obj = None
        self._hist = []

    def abbreviated_search(self, population, sorting_method, update_method):
        """
        Searching data by pareto levels with enterned sorting and updating methods.

        Args:
            population (`list`): List with the elements - canidate solutions of the case-specific subclass of 
                ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
            sorting_method (`callable`): The method of population separation into non-dominated levels
            update_method (`callable`): The method of point addition into the population and onto the non-dominated levels.

        Returns:
            None
        """
        self.pareto_levels = ParetoLevels(population, sorting_method=sorting_method, update_method=update_method)
        if global_var.verbose.show_warnings:
            if len(population) == 1:
                warnings.warn('The multiobjective optimization algorithm has been able to create only a single unique solution. The search has been abbreviated.')
            else:
                warnings.warn(f'The multiobjective optimization algorithm has been able to create only {len(population)} unique solution. The search has been abbreviated.')
        self.abbreviated_search_executed = True
            
    @staticmethod
    def weights_generation(weights_num, delta) -> list:
        """
        Method to calculate the set of vectors to divide the problem of Pareto frontier
        discovery into several subproblems of Pareto frontier sector discovery, where
        each sector is defined by a weight vector.
        
        Args:
            weights_num (`int`): Number of the weight vectors, dividing the objective function values space.
            delta (`float`): Parameter of uniform spacing between the weight vectors; *H = 1 / delta*        
                should be integer - a number of divisions along an objective coordinate axis.
        
        Returns:
            weights (`list`): weight vectors (`np.ndarrays`), introduced to decompose the optimization problem into 
                several subproblems by dividing Pareto frontier into a number of sectors.
        """
        weights = np.empty(weights_num)
        assert 1./delta == round(1./delta) # check, if 1/delta is integer number
        m = np.zeros(weights_num)
        for weight_idx in np.arange(weights_num):
            weights[weight_idx] = np.random.choice([div_idx * delta for div_idx in np.arange(1./delta + 1e-8 - np.sum(m[:weight_idx + 1]))])
            m[weight_idx] = weights[weight_idx]/delta
        weights[-1] = 1 - np.sum(weights[:-1])
        
        weights = np.abs(weights)
        return list(weights)

    
    def pass_best_objectives(self, *args) -> None:
        """
        Setter of the `moeadd_optimizer.best_obj` attribute. 
        
        Args:
            args (`np.ndarray|list`): The values of the objective functions for the many-objective optimization problem.
        
        Returns:
            None
        """
        if len(self.pareto_levels.population) != 0:
            print('comparing lengths', len(args), len(self.pareto_levels.population[0].obj_funs))
            assert len(args) == len(self.pareto_levels.population[0].obj_funs)
            self.best_obj = np.empty(len(self.pareto_levels.population[0].obj_funs))
        elif len(self.pareto_levels.unplaced_candidates) != 0:
            self.best_obj = np.empty(len(self.pareto_levels.unplaced_candidates[0].obj_funs))
        else:
            raise IndexError('No candidates added into the Pareto levels while they must not be empty.')
        for arg_idx, arg in enumerate(args):
            self.best_obj[arg_idx] = arg if isinstance(arg, int) or isinstance(arg, float) else arg() # Переделать под больше elif'ов
    

    def set_sector_processer(self, processer: MOEADDSectorProcesser) -> None:
        """
        Setter of the `moeadd_optimizer.sector_processer` attribute.
        
        Args:
            processer (`MOEADDSectorProcesser`): The operator, which defines the evolutionary process
        """
        self.sector_processer = processer

    def set_strategy(self, strategy_director):
        builder = strategy_director.builder
        builder.assemble(True)
        self.set_sector_processer(builder.processer)
    
        
    def optimize(self, epochs):
        """
        Method for the main unconstrained evolutionary optimization. Can be applied repeatedly to 
        the population, if the previous results are insufficient. The output of the 
        optimization shall be accessed with the ``optimizer.pareto_level`` object and 
        its attributes ``.levels`` or ``.population``.
        
        Args:        
            epochs (`int`): Maximum number of iterations, during that the optimization will be held.
        
        Note:
            that if the algorithm converges to a single Pareto frontier, the optimization is stopped.
        
        """
        if not self.abbreviated_search_executed:
            self.hist = []
            assert not type(self.best_obj) == type(None)
            for epoch_idx in np.arange(epochs):
                if global_var.verbose.show_iter_idx:
                    print(f'Multiobjective optimization : {epoch_idx}-th epoch.')
                for weight_idx in np.arange(len(self.weights)):
                    if global_var.verbose.show_iter_idx:
                        print(f'During MO : processing {weight_idx}-th weight.')                    
                    sp_kwargs = self.form_processer_args(weight_idx)
                    self.sector_processer.run(population_subset = self.pareto_levels, 
                                              EA_kwargs = sp_kwargs)
                stats = self.pareto_levels.get_stats()
                self._hist.append(stats)
                if global_var.verbose.iter_fitness:
                    print(f'after epoch {epoch_idx} obtained OF: mean = {np.mean(stats[0], axis = 0)}, \
                           var = {np.mean(stats[0], axis = 0)}')    

    def form_processer_args(self, cur_weight : int): # TODO: inspect the most convenient input format
        """
        Forming arguments of the processer 
        """
        return {'weight_idx' : cur_weight, 'weights' : self.weights, 'best_obj' : self.best_obj, 
                'neighborhood_vectors' : self.neighborhood_lists}


    def get_hist(self, best_only: bool = True):
        if best_only:
            return [elem[0] for elem in self._hist]
        else:
            return self._hist