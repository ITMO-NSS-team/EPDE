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

from epde.moeadd.moeadd_strategy_elems import MOEADDSectorProcesser
from epde.moeadd.moeadd_supplementary import fast_non_dominated_sorting, ndl_update, Equality, Inequality        
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
    
    Parameters:
    ------------
    
    population : list
        List with the elements - canidate solutions of the case-specific subclass of 
        ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
        
    sorting_method : function, optional, default - ``src.moeadd.moeadd_supplementary.fast_non_dominated_sorting``
        The method of population separation into non-dominated levels.
        
    update_method : function, optional, defalut - ``src.moeadd.moeadd_supplementary.ndl_update``
        The method of point addition into the population and onto the non-dominated levels.
    
    Attributes:
    -----------
    
    population : list
        List with the elements - canidate solutions of the case-specific subclass of 
        ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
        
    levels : list
        List with the elements - lists of solutions, representing non-dominated levels. 
        The 0-th element - the current Pareto frontier.
        
    _sorting_method : function
        The method of population separation into non-dominated levels.
        
    _update_method : function
        The method of point addition into the population and onto the non-dominated levels.
    
    Notes:
    -------
    
    The initialization of objects of this class is held automatically in the __init__ of 
    moeadd optimizer, thus no extra interactions of a user with this class are necessary.
    
    '''
    def __init__(self, population, sorting_method = fast_non_dominated_sorting, 
                 update_method = ndl_update, initial_sort : bool = False):
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
        while self._unplaced_candidates:
            self.population.append(self._unplaced_candidates.pop())
        if any([any([candidate == other_candidate for other_candidate in self.population[:idx] + self.population[idx+1:]])
                for idx, candidate in enumerate(self.population)]):
            print([candidate.text_form for candidate in self.population])
            raise Exception('Duplicating initial candidates')
        self.levels = self.sort()

    def sort(self):
        '''
        
        Sorting of the population into Pareto non-dominated levels.
        
        '''
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
        '''
        
        Addition of a candidate solution point into the pareto levels and the population list.

        Arguments:
        ----------
        
        point : the case-specific subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
            The point, added into the candidate solutions pool.
            
        '''
        # print(f'IN MOEADD UPDATE {point.text_form}')        
        self.levels = self._update_method(point, self.levels)
        self.population.append(point)

    def delete_point(self, point):
        '''
        
        Deletion of a candidate solution point from the pareto levels and the population list.
        
        Arguments:
        -----------
        
        point : the case-specific subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
            The point, removed from the candidate solutions pool.
        
        '''        
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

    def fit_convex_hull(self):
        if len(self.levels) > 1:
            warnings.warn('Algorithm has not converged to a single Pareto level yet!')
        points = np.vstack([sol.obj_fun for sol in self.population])
        points = np.concatenate((points, np.max(points, axis = 0).reshape((1, -1))))
        

class ParetoLevelsIterator(object):
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
    '''
    Solving multiobjective optimization problem (minimizing set of functions) with an 
    evolutionary approach. In this class, the unconstrained variation of the problem is 
    considered.
    
    Initialization of the evolutionary optimizer is done with the introduction of 
    initial population of candidate solutions, divided into Pareto non-dominated 
    levels (sets of solutions, such, as none of the solution of a level dominates 
    another on the same level), and creation of set of weights with a proximity list 
    defined for each of them.
    
    
    Parameters:
    ---------
    
    pop_constructor : obj of moeadd_stc.moe_population_constructor class
        The problem-specific population constructor object with the ``create`` method, 
        that returns a new (often randomly) generated solution. The template for the 
        class is presented in the moeadd_stc.moe_population_constructor abstract class.
        
    weights_num : int
        Number of the weight vectors, dividing the objective function values space. Often, shall
        be same, as the population size.
        
    pop_size : int
        The size of the candidate solution population.
        
    solution_params : dict or None
        The dicitionary with the solution parameters, passed into each new created solution during the 
        initialization. 
        
    delta : float
        parameter of uniform spacing between the weight vectors; *H = 1 / delta*        
        should be integer - a number of divisions along an objective coordinate axis.
        
    neighbors_number : int, *> 0*
        number of neighboring weight vectors to be considered during the operation 
        of evolutionary operators as the "neighbors" of the processed sectors.
        
    nds_method : function, optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
        Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article 
        *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist
        multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
        vol. 6, no. 2, pp. 182–197, Apr. 2002.*
        
    ndl_update : function, optional, defalut ``moeadd.moeadd_supplementary.ndl_update``
        Method of adding a new solution point into the objective functions space, introduced 
        to minimize the recalculation of the non-dominated levels for the entire population. 
        The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
        update approach for steady-state evolutionary multiobjective optimization,” 
        Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
        MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
    
    
    Attributes:
    ----------
    
    weights : np.ndarray
        Weight vectors, introduced to decompose the optimization problem into 
        several subproblems by dividing Pareto frontier into a numeber of sectors.
        
    pareto_levels : ``pareto_levels`` class object
        Pareto levels object, containing the population of candidate solution as a list of 
        points and as a list of levels.
        
    best_obj : np.array
        The best achievable values of objective functions. Should be introduced with 
        ``self.pass_best_objectives`` method.
        
    evolutionary_operator : object of subclass of ``moe_evolutionary_operator``
        The operator, which defines the evolutionary process.
    
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
    
    '''
    def __init__(self, pop_constructor, weights_num, pop_size, solution_params, delta, neighbors_number, 
                 nds_method = fast_non_dominated_sorting, ndl_update = ndl_update):

        assert weights_num == pop_size, 'Each individual in population has to correspond to a sector'
        self.abbreviated_search_executed = False
        soluton_creation_attempts_softmax = 10
        soluton_creation_attempts_hardmax = 100
        
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

    def abbreviated_search(self, population, sorting_method, update_method):
        self.pareto_levels = ParetoLevels(population, sorting_method=sorting_method, update_method=update_method)
        if global_var.verbose.show_warnings:
            if len(population) == 1:
                warnings.warn('The multiobjective optimization algorithm has been able to create only a single unique solution. The search has been abbreviated.')
            else:
                warnings.warn(f'The multiobjective optimization algorithm has been able to create only {len(population)} unique solution. The search has been abbreviated.')
        self.abbreviated_search_executed = True
            
    @staticmethod
    def weights_generation(weights_num, delta) -> list:
        '''
        
        Method to calculate the set of vectors to divide the problem of Pareto frontier
        discovery into several subproblems of Pareto frontier sector discovery, where
        each sector is defined by a weight vector.
        
        Arguments:
        ----------
        
        weights_num : int
            Number of the weight vectors, dividing the objective function values space.
            
        delta : float
            Parameter of uniform spacing between the weight vectors; *H = 1 / delta*        
            should be integer - a number of divisions along an objective coordinate axis.
        
        Returns:
        ---------
        
        Weights : np.ndarray
            Weight vectors, introduced to decompose the optimization problem into 
            several subproblems by dividing Pareto frontier into a number of sectors.
        
        '''
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
        '''
        
        Setter of the `moeadd_optimizer.best_obj` attribute. 
        
        Arguments:
        ----------
        
        *args : np.array/list of float values
            The values of the objective functions for the many-objective optimization 
            problem.
        
        '''
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
    

    def set_sector_processer(self, processer : MOEADDSectorProcesser) -> None:
        '''
        
        Setter of the `moeadd_optimizer.sector_processer` attribute.
        
        Arguments:
        ----------
        
        operator : object of subclass of ``MOEADDSectorProcesser``
            The operator, which defines the evolutionary process
        
        '''
        
        # добавить возможность теста оператора
        self.sector_processer = processer
    
        
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, epochs):
        '''
        
        Method for the main unconstrained evolutionary optimization. Can be applied repeatedly to 
        the population, if the previous results are insufficient. The output of the 
        optimization shall be accessed with the ``optimizer.pareto_level`` object and 
        its attributes ``.levels`` or ``.population``.
        
        Parameters:
        -----------
        
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)
        
        neighborhood_selector_params : tuple/list or None
            Iterable, which will be passed into neighborhood_selector, as 
            an arugument. *None*, is no additional arguments are required inside
            the selector.
            
        epochs : int
            Maximum number of iterations, during that the optimization will be held.
            Note, that if the algorithm converges to a single Pareto frontier, 
            the optimization is stopped.
        
        '''
        if not self.abbreviated_search_executed:
            assert not type(self.best_obj) == type(None)
            for epoch_idx in np.arange(epochs):
                if global_var.verbose.show_moeadd_epochs:
                    print(f'Multiobjective optimization : {epoch_idx}-th epoch.')
                for weight_idx in np.arange(len(self.weights)):
                    if global_var.verbose.show_moeadd_epochs:
                        print(f'During MO : processing {weight_idx}-th weight.')                    
                    sp_kwargs = self.form_processer_args(weight_idx)
                    self.sector_processer.run(population_subset = self.pareto_levels, 
                                              EA_kwargs = sp_kwargs)
                        
                # if len(self.pareto_levels.levels) == 1:
                #     break
                    
    def form_processer_args(self, cur_weight : int): # TODO: inspect the most convenient input format
        return {'weight_idx' : cur_weight, 'weights' : self.weights, 'best_obj' : self.best_obj, 
                'neighborhood_vectors' : self.neighborhood_lists}


class MOEADDOptimizerConstrained(MOEADDOptimizer):
    #   TODO: rewrite in a better way, according to the new operators
    '''
    Solving multiobjective optimization problem (minimizing set of functions) with an 
    evolutionary approach. Here, the constrained variation of the problem is 
    considered.
    
    Initialization of the evolutionary optimizer is done with the introduction of 
    initial population of candidate solutions, divided into Pareto non-dominated 
    levels (sets of solutions, such, as none of the solution of a level dominates 
    another on the same level), and creation of set of weights with a proximity list 
    defined for each of them.
    
    Parameters:
    ---------
    
    pop_constructor : obj of ``moeadd_stc.moe_population_constructor`` class
        The problem-specific population constructor object with the ``create`` method, 
        that returns a new (often randomly) generated solution. The template for the 
        class is presented in the moeadd_stc.moe_population_constructor abstract class.
        
    weights_num : int
        Number of the weight vectors, dividing the objective function values space. Often, shall
        be same, as the population size.
        
    pop_size : int
        The size of the candidate solution population.
        
    solution_params : dict or None
        The dicitionary with the solution parameters, passed into each new created solution during the 
        initialization. 
        
    delta : float
        parameter of uniform spacing between the weight vectors; *H = 1 / delta*        
        should be integer - a number of divisions along an objective coordinate axis.
        
    neighbors_number : int, *> 0*
        number of neighboring weight vectors to be considered during the operation 
        of evolutionary operators as the "neighbors" of the processed sectors.
        
    nds_method : function, optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
        Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article 
        *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist
        multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
        vol. 6, no. 2, pp. 182–197, Apr. 2002.*
        
    ndl_update : function, optional, defalut ``moeadd.moeadd_supplementary.ndl_update``
        Method of adding a new solution point into the objective functions space, introduced 
        to minimize the recalculation of the non-dominated levels for the entire population. 
        The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
        update approach for steady-state evolutionary multiobjective optimization,” 
        Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
        MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
    
    
    Attributes:
    ----------
    
    weights : np.ndarray
        Weight vectors, introduced to decompose the optimization problem into 
        several subproblems by dividing Pareto frontier into a numeber of sectors.
        
    pareto_levels : ``pareto_levels`` class object
        Pareto levels object, containing the population of candidate solution as a list of 
        points and as a list of levels.
        
    best_obj : np.array
        The best achievable values of objective functions. Should be introduced with 
        ``self.pass_best_objectives`` method.
        
    evolutionary_operator : object of subclass of ``moe_evolutionary_operator``
        The operator, which defines the evolutionary process.
    
    constraints : tuple 
        Contains constriants, defined as ``src.moeadd_supplementary.Inequality``, or  
        ``src.moeadd_supplementary.Equality`` objects. Definition shall be done with 
        self.set_constraint method.
    
    Example:
    --------
    
    >>> pop_constr = test_population_constructor()
    >>> optimizer = moeadd_optimizer_constrained(pop_constr, 40, 40, 
    >>>                              None, delta = 1/50., 
    >>>                              neighbors_number = 5)
    >>> operator = test_evolutionary_operator(mixing_xover,
    >>>                                      gaussian_mutation)
    >>> constr_1 = Inequality(lambda x: x[0] + 4)
    >>> constr_2 = Equality(lambda x: x[1] + 4)
    >>> optimizer.set_constraints(constr_1, constr_2)
    >>> optimizer.set_evolutionary(operator=operator)
    >>> optimizer.pass_best_objectives(0, 0)
    >>> optimizer.optimize(simple_selector, 0.95, (4,), 100, 0.75)
    
    In that case, we solve the optimization problem with two objective functions. The population
    constructor is defined with the imported dummy class ``test_population_constructor``, 
    and evolutionary operator contains mutation and crossover suboperators. The constraints are
    presented by an inequality :math:`g_1(\mathbf{x}) = x_0 + 4 >= 0` and :math:`g_2(\mathbf{x}) = x_1 + 4 = 0`. 
    
    '''
    def set_constraints(self, *args) -> None:
        '''
        
        Definition of constraints.
        
        Parameters:
        -----------
        
        args: ``src.moeadd_supplementary.Inequality``, or ``src.moeadd_supplementary.Equality`` objects
            Constraint objects, where equality is viewed as g(x) == 0, and inequatlity is considered in format g(x) >= 0

        Example:
        ---------
        
        >>> constr_1 = Inequality(lambda x: x[0] + 4)
        >>> constr_2 = Equality(lambda x: x[1] - 2)
        >>> 
        >>> optimizer.set_constraints(constr_1, constr_2)
        
        '''
        self.constraints = args


    def constraint_violation(self, solution) -> float:
        '''
        
        Method to obtain the constraint violation of a particular solution.
        
        Parameters:
        -----------
        
        solution : object of subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
            The solution, for which the constriant violations are calculated. 
            
        Returns:
        --------
        
        constraint_violation : float
            Value, equals to the sum of all constraint violations. == 0, if the solution 
            satisfies all constraints. 
        
        '''
        summ = 0
        x = solution.vals
        for constraint in self.constraints:
            summ += constraint(x)
        return summ

    def tournament_selection(self, candidate_1, candidate_2):
        '''
        Selection of an individual with lower value of constraint violation. 
        
        Parameters:
        -----------
        
        candidate_1, candidate_2 : objects of subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
            The compared individuals.
            
        Returns:
        --------
        
        candidate : object of subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
            The individual with lower value of constraint violation between two input ones. 
            If the values of **candidate_1**, and **candidate_2** are equal, the 
            **candidate** is chosen among them.
        '''
        if self.constraint_violation(candidate_1) < self.constraint_violation(candidate_2):
            return candidate_1
        elif self.constraint_violation(candidate_1) > self.constraint_violation(candidate_2):
            return candidate_2
        else:
            return np.random.choice((candidate_1, candidate_2))


    # def update_population(self, offspring, PBI_penalty):
    #     '''
        
    #     Update population to get the pareto-nondomiated levels with the worst element removed. 
    #     Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection). 
    #     Additionally, the constraint violations are considered in the selection of the 
    #     "worst" individual.
        
    #     '''        
    #     self.pareto_levels.update(offspring)
    #     cv_values = np.zeros(len(self.pareto_levels.population))
    #     for sol_idx, solution in enumerate(self.pareto_levels.population):
    #         cv_val = self.constraint_violation(solution)
    #         if cv_val > 0:
    #             cv_values[sol_idx] = cv_val 
    #     if sum(cv_values) == 0:
    #         if len(self.pareto_levels.levels) == 1:
    #             worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
    #         else:
    #             if self.pareto_levels.levels[len(self.pareto_levels.levels) - 1] == 1:
    #                 domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
    #                 reference_solution = self.pareto_levels.levels[len(self.pareto_levels.levels) - 1][0]
    #                 reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
    #                 if len(domain_solutions[reference_solution_domain] == 1):
    #                     worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
    #                 else:
    #                     worst_solution = reference_solution
    #             else:
    #                 last_level_by_domains = population_to_sectors(self.pareto_levels.levels[len(self.pareto_levels.levels)-1], self.weights)
    #                 most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
    #                 crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]
    
    #                 if len(crowded_domains) == 1:
    #                     most_crowded_domain = crowded_domains[0]
    #                 else:
    #                     PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self.weights[domain_idx], self.best_obj, PBI_penalty) 
    #                                                         for sol_obj in last_level_by_domains[domain_idx]])
    #                     PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
    #                     most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                        
    #                 if len(last_level_by_domains[most_crowded_domain]) == 1:
    #                     worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)                            
    #                 else:
    #                     PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, self.weights[most_crowded_domain], self.best_obj, PBI_penalty), 
    #                                            last_level_by_domains[most_crowded_domain]), dtype = float)
    #                     worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
    #     else:
    #         infeasible = [solution for solution, _ in sorted(list(zip(self.pareto_levels.population, cv_values)), key = lambda pair: pair[1])]
    #         infeasible.reverse()
    #         infeasible = infeasible[:np.nonzero(cv_values)[0].size]
    #         deleted = False
    #         domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
            
    #         for infeasable_element in infeasible:
    #             domain_idx = [domain_idx for domain_idx, domain in enumerate(domain_solutions) if infeasable_element in domain][0]
    #             if len(domain_solutions[domain_idx]) > 1:
    #                 deleted = True
    #                 worst_solution = infeasable_element
    #                 break
    #         if not deleted:
    #             worst_solution = infeasible[0]
        
    #     self.pareto_levels.delete_point(worst_solution)

            
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, epochs, PBI_penalty):
        '''
        
        Method for the main unconstrained evolutionary optimization. Can be applied repeatedly to 
        the population, if the previous results are insufficient. The output of the 
        optimization shall be accessed with the ``optimizer.pareto_level`` object and 
        its attributes ``.levels`` or ``.population`` .
        
        Parameters:
        -----------
        
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
        
        neighborhood_selector_params : tuple/list or None
            Iterable, which will be passed into neighborhood_selector, as 
            an arugument. *None*, is no additional arguments are required inside
            the selector.
            
        epochs : int
            Maximum number of iterations, during that the optimization will be held.
            Note, that if the algorithm converges to a single Pareto frontier, 
            the optimization is stopped.
            
        PBI_penalty :  float, optional, default 1.
            The penalty parameter, used in penalty based intersection 
            calculation.
        
        
        '''
        if not self.abbreviated_search_executed:        
            self.train_hist = []
            for epoch_idx in np.arange(epochs):
                if global_var.verbose.show_moeadd_epochs:
                    print(f'Multiobjective optimization : {epoch_idx}-th epoch.')                
                for weight_idx in np.arange(len(self.weights)):
                    if global_var.verbose.show_moeadd_epochs:
                        print(f'During MO : processing {weight_idx}-th weight.')     
                    obj_fun = np.array([solution.obj_fun for solution in self.pareto_levels.population])
                    self.train_hist.append(np.mean(obj_fun, axis=0))
                    parent_idxs = self.mating_selection(weight_idx, self.weights, self.neighborhood_lists, self.pareto_levels.population,
                                                   neighborhood_selector, neighborhood_selector_params, delta)
                    if len(parent_idxs) % 2:
                        parent_idxs = parent_idxs[:-1]
                    np.random.shuffle(parent_idxs) 
                    parents_selected = [self.tournament_selection(self.pareto_levels.population[int(parent_idxs[2*p_metaidx])], 
                                                                  self.pareto_levels.population[int(parent_idxs[2*p_metaidx+1])]) for 
                                        p_metaidx in np.arange(int(len(parent_idxs)/2.))]
                    
                    offsprings = self.evolutionary_operator.crossover(parents_selected) # В объекте эволюционного оператора выделять кроссовер
                    if type(offsprings) == list or type(offsprings) == tuple:
                        for offspring_idx, offspring in enumerate(offsprings):
                            if global_var.verbose.show_moeadd_epochs:
                                print(f'Offsping {offspring_idx}.')
                            attempt = 1; attempt_limit = 2
                            while True:
                                temp_offspring = self.evolutionary_operator.mutation(offspring)
                                if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                                    self.update_population(temp_offspring, PBI_penalty)
                                    break
                                elif attempt >= attempt_limit:
                                    break
                                attempt += 1
                            
                    else:
                        if global_var.verbose.show_moeadd_epochs:
                            print(f'Offsping {offsprings}.')   
                        attempt = 1; attempt_limit = 2                            
                        while True:
                            temp_offspring = self.evolutionary_operator.mutation(offsprings)
                            if not any([temp_offspring == solution for solution in self.pareto_levels.population]):
                                self.update_population(temp_offspring, PBI_penalty)
                                break
                            elif attempt >= attempt_limit:
                                break
                            attempt += 1                            
                if len(self.pareto_levels.levels) == 1:
                    break
