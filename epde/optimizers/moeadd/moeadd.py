"""

Main classes and functions of the moeadd optimizer.

"""

import numpy as np
import warnings
from itertools import chain

from typing import Union, List, Callable
from functools import reduce

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

# from copy import deepcopy
# from functools import reduce

import epde.globals as global_var
from epde.structure.main_structures import SoEq
from epde.optimizers.moeadd.population_constr import SystemsPopulationConstructor
from epde.optimizers.moeadd.vis import ParetoVisualizer

from epde.optimizers.moeadd.solution_template import MOEADDSolution
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
from epde.optimizers.moeadd.supplementary import fast_non_dominated_sorting, ndl_update, Equality, Inequality, acute_angle
from scipy.spatial import ConvexHull

def clear_list_of_lists(inp_list) -> list:
    '''
    Delete elements-lists with len(0) from the list
    '''
    return [elem for elem in inp_list if len(elem) > 0]


class ObjFunNormalizer(object):
    def __init__(self, obj_worst_vals: np.ndarray):
        self._worst_vals = obj_worst_vals

    def __call__(self, obj_vals: np.ndarray):
        assert obj_vals.size == self._worst_vals.size, 'Passed objective values have different length, than stored max ones.'
        return obj_vals / self._worst_vals


def checkWeightAssignmentUniqueness(arg: np.ndarray):
    assert arg.ndim == 2 and arg.shape[0] == arg.shape[1], 'weights assignment array has to be a square matrix'
    for widx in range(arg.shape[0]):
        assert np.sum(arg[widx, :]) == 1, f'Rows of weight assignment matrixes have to contain a single "1", instead got {arg[widx, :]}.'
    for sidx in range(arg.shape[1]):
        assert np.sum(arg[:, sidx]) == 1, f'Columns of solution assignment in weight matrixes have to contain a single "1", instead got {arg[:, sidx]}.'


def randomSolutionAssignment(weights: np.ndarray, solutions: List[MOEADDSolution]):
    assert len(solutions) == weights.shape[0], 'Solutions do not match weights in length.'
    indexes = np.random.permutation(len(solutions))

    for idx, solution in enumerate(solutions):
        solution.set_domain(indexes[idx])


def marriageSolutionAssignment(weights: np.ndarray, solutions: List[MOEADDSolution]):
    assert len(solutions) == weights.shape[0], f'Solutions do not match weights in length: {len(solutions)} vs {weights.shape[0]}.'

    acute_angles = np.empty((weights.shape[0], weights.shape[0]))

    for i, weight in enumerate(weights):
        weight_full = [item for item in weight for _ in solutions[0].vals]
        for j, solution in enumerate(solutions):
            acute_angles[i, j] = acute_angle(weight_full, solution.obj_fun)
            # acute_angles[i, j] = acute_angle(weight, solution.obj_fun)

    w_preferences = np.argsort(acute_angles, axis = 1)

    s_decisions = np.full(shape = len(solutions), fill_value = -1, dtype = int)
    s_cur_pref = np.full(shape = len(solutions), fill_value = 0.)

    matches = np.zeros_like(w_preferences) # Match flags: 0 - no match, 1 - suggested match

    while not np.isclose(np.sum(matches), weights.shape[0]):
        for i in range(weights.shape[0]):
            if sum(matches[i]) > 0:
                continue

            for j in range(w_preferences.shape[1]):
                if s_decisions[w_preferences[i, 0]] == -1:
                    matches[i, ...] = 0
                    matches[i, w_preferences[i, 0]]  = 1
                    s_decisions[w_preferences[i, 0]] = i
                    s_cur_pref[w_preferences[i, 0]]  = acute_angles[i, w_preferences[i, 0]]
                    break

                elif (s_cur_pref[w_preferences[i, 0]] > acute_angles[i, w_preferences[i, 0]]):
                    w_preferences[s_decisions[w_preferences[i, 0]]] = np.roll(w_preferences[s_decisions[w_preferences[i, 0]]], shift = -1, axis = 0)
                    w_preferences[s_decisions[w_preferences[i, 0]], -1] = -1
                    assert matches[s_decisions[w_preferences[i, 0]], w_preferences[i, 0]] == 1, 'Boop! Possible error'
                    matches[s_decisions[w_preferences[i, 0]], w_preferences[i, 0]] = 0

                    matches[i, ...]  = 0
                    matches[i, w_preferences[i, 0]]  = 1
                    s_decisions[w_preferences[i, 0]] = i
                    s_cur_pref[w_preferences[i, 0]]  = acute_angles[i, w_preferences[i, 0]]
                    break

                else:
                    w_preferences[i] = np.roll(w_preferences[i], shift = -1, axis = 0)
                    w_preferences[i, -1] = -1

    print('acute_angles\n', acute_angles)
    print('matches\n', matches)

    checkWeightAssignmentUniqueness(matches)
    for sol_idx, solution in enumerate(solutions):
        weight_idx = np.where(matches[:, sol_idx] == 1)[0][0]
        print(f'Assigned weight {weight_idx} for {sol_idx}')
        solution.set_domain(weight_idx)


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
    _weights_assigned = False

    def __init__(self,
                 population, weights: np.ndarray = None, 
                 sorting_method: Callable = fast_non_dominated_sorting, 
                 update_method: Callable = ndl_update,
                 weights_assigner: Callable = marriageSolutionAssignment): # , initial_sort = False
        """
        Args:
            population (`list`): List with the elements - canidate solutions of the case-specific subclass of 
                ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
            sorting_method (`callable`): optional, default - ``src.moeadd.moeadd_supplementary.fast_non_dominated_sorting``
                The method of population separation into non-dominated levels
            update_method (`callable`): optional, defalut - ``src.moeadd.moeadd_supplementary.ndl_update``
                The method of point addition into the population and onto the non-dominated levels.
        """
        assert weights.ndim == 2, 'Weights must be represented as a 2D np.ndarray, where 1st dim - index of the weight vector, 2nd - component'
        self._sorting_method = sorting_method
        self._update_method = update_method
        self.set_weights(weights)

        self._weights_assigner = weights_assigner
        self.population = []
        self.unplaced_candidates = population

        self.normalizer = None
        self.history = set()
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        from epde.loader import attrs_from_dict
        supported_attrs = ['population']
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
            
        if attribute == 'population':
            self.population = []
            for system_elem in value:
                system = SoEq.__new__(SoEq)
                attrs_from_dict(system, system_elem, except_attrs)
                self.population.append(system)
        self.levels = self.sort()
    
    def attrs_from_dict(self, attributes, except_keys = ['obj_type']):
        self.__dict__ = {key : item for key, item in attributes.items()
                         if key not in except_keys}
    
    def set_normalizer(self):
        objectives = np.stack(reduce(lambda x, y: x.extend(y) or x,
                                     [[elem.obj_fun for elem in self.population]]), axis = 0)

        self.normalizer = ObjFunNormalizer(np.max(objectives, axis = 0))

    @property
    def levels(self):
        return self._levels
    
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
        self.levels = self.sort()

    def sort(self):
        """
        Sorting of the population into Pareto non-dominated levels.
        """
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
        population_cleared = []
        point_system = point.terms_labels
        for level in self.levels:
            temp = []
            for element in level:
                if element.terms_labels != point_system:
                    temp.append(element)
                    population_cleared.append(element)
            if not len(temp) == 0:
                new_levels.append(temp)

        if len(population_cleared) != sum([len(level) for level in new_levels]):
            print(len(population_cleared), len(self.population), sum([len(level) for level in new_levels]))
            print('initial population', [solution.vals for solution in self.population], len([solution.vals for solution in self.population]), '\n')
            print('cleared population', [solution.vals for solution in population_cleared], len([solution.vals for solution in self.population]), '\n')
            print(point.vals)
            raise Exception('Deleted something extra')
        self.levels = new_levels
        self.population = population_cleared

    def get_stats(self):
        return np.array(flatten_chain([[element.obj_fun for element in level] 
                                       for level in self.levels]))

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

    def set_weights(self, weights):
        if weights is None:
            print(f'Setting ParetoLevels attribule weights with None: this should be a placeholder, expect futher logs.')
        #if neccessary, implement additional logic into setter
        self._weights = weights

    def associate_weights(self):
        if not self._weights_assigned:
            if self._weights is None:
                raise AttributeError('Weights should not be None in the assignment phase.')
            
            if len(self.population) == 0:
                self._weights_assigner(self._weights, self.unplaced_candidates)
            else:
                self._weights_assigner(self._weights, self.population)
            
            self._weights_assigned = True


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
    >>>                              None, H = 15,
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
    def __init__(self, population_instruct, pop_size, solution_params,
                 H: int, neighbors_number: int, nds_method = fast_non_dominated_sorting, 
                 ndl_update = ndl_update, passed_population: Union[List, ParetoLevels] = None,
                 best_sol_vals = None, weights_assigner: str = 'marriage'):
        """
        Initialization of the evolutionary optimizer is done with the introduction of 
        initial population of candidate solutions, divided into Pareto non-dominated 
        levels (sets of solutions, such, as none of the solution of a level dominates 
        another on the same level), and creation of set of weights with a proximity list 
        defined for each of them.
        
        Parameters
        ----------
        population_instruct : dict
            Parameters of the individual creation.
        best_obj : List[int]
            List of best obtaiable values for each criteria in the optimization problem.
        pop_size : int 
            The size of the candidate solution population.
        solution_params : dict
            The dicitionary with the solution parameters, passed into each new created solution during the initialization.
        H : float
            The parameter of uniform spacing between the weight vectors; *H = 1 / delta* should be integer - a number of divisions along an objective coordinate axis.
        neighbors_number : int 
            The number of neighboring weight vectors to be considered during the operation of evolutionary operators as the "neighbors" of the processed sectors.
        nds_method : callable, optional
            Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article 
            *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
            vol. 6, no. 2, pp. 182-197, Apr. 2002.* The default is ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
        ndl_update : callable, optional
            Method of adding a new solution point into the objective functions space, introduced 
            to minimize the recalculation of the non-dominated levels for the entire population. 
            The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
            update approach for steady-state evolutionary multiobjective optimization,” 
            Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
            MI, USA, Tech. Rep. COIN No. 2014014, 2014.* The default - ``moeadd.moeadd_supplementary.ndl_update``
        passed_population : list or ParetoLevels object, optional
            Pre-generated initial guess of the population. Can be passed as a ``list`` of candidate solutions or as a ``ParetoLevels`` object.
            If the number of passed candidates is lower, than the required size of the population, additional solutions will be created.
        solution_assignment : str, optional
            Method of initial weight allocation to candidate solutions. Existing options are 'random' for legacy random allocation and 
            'marriage' for Gale-Shapley algorithm. The default - 'marriage'
        """
        self.abbreviated_search_executed = False

        assert (type(solution_params) == type(None) or
                 type(solution_params) == dict), 'The solution parameters, passed into population constructor must be in dictionary'

        weights_size = len(best_sol_vals)
        self.weights = np.array(self.weights_generation(weights_size, H))

        pop_constructor = SystemsPopulationConstructor(**population_instruct)

        if (passed_population is None) or isinstance(passed_population, list):
            population = [] if passed_population is None else passed_population
            psize = len(population)
            for solution_idx in range(psize):
                pop_constructor.applyToPassed(population[solution_idx], **solution_params)

            for solution_idx in range(pop_size - psize):
                solution_gen_idx = 0
                if type(solution_params) == type(None): solution_params = {}
                temp_solution = pop_constructor.create(**solution_params)

                for equation in temp_solution.vals:
                    while len(equation.terms_labels) != len(equation.structure):
                        temp_solution.vals[equation.main_var_to_explain].randomize()
                        temp_solution.vals[equation.main_var_to_explain].reset_saved_state()
                population.append(temp_solution)

                solution_gen_idx += 1
            if weights_assigner == 'marriage':
                weights_assigner_method = marriageSolutionAssignment
            elif weights_assigner == 'random':
                weights_assigner_method = randomSolutionAssignment
            else:
                warnings.warn(f'Get unimplemented method of weights assignment {weights_assigner_method}')
            self.pareto_levels = ParetoLevels(population, weights=self.weights,
                                              sorting_method = nds_method, 
                                              update_method = ndl_update,
                                              weights_assigner=weights_assigner_method)
        else:
            if not isinstance(passed_population, ParetoLevels):
                raise TypeError(f'Incorrect type of the population passed. Expected ParetoLevels object, instead got \
                                 {type(passed_population)}')
            self.pareto_levels = passed_population

        self.neighborhood_lists = []
        weights_num = self.weights.shape[0]
        for weights_idx in range(weights_num):
            self.neighborhood_lists.append([elem_idx for elem_idx, _ in sorted(
                    list(zip(np.arange(weights_num), [np.linalg.norm(self.weights[weights_idx, :] - self.weights[weights_idx_inner, :]) for weights_idx_inner in np.arange(weights_num)])), 
                    key = lambda pair: pair[1])][:neighbors_number+1]) # срез листа - задаёт регион "близости"

        self.best_obj = best_sol_vals
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
    def weights_generation_old(weights_num, delta) -> list:
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
        assert 1. / delta == round(1. / delta)  # check, if 1/delta is integer number
        m = np.zeros(weights_num)
        for weight_idx in np.arange(weights_num):
            weights[weight_idx] = np.around(np.random.choice(
                [div_idx * delta for div_idx in np.arange(1. / delta + 1e-8 - np.sum(m[:weight_idx + 1]))]), 2)
            m[weight_idx] = weights[weight_idx] / delta
        weights[-1] = np.around(1 - np.sum(weights[:-1]), 2)

        weights = np.abs(weights)
        return list(weights)

    @staticmethod
    def weights_generation(weights_num,  H) -> list:
        """
        Method to calculate the set of vectors to divide the problem of Pareto frontier
        discovery into several subproblems of Pareto frontier sector discovery, where
        each sector is defined by a weight vector.
        
        Args:
            weights_num (`int`): Number of the weight vectors, dividing the objective function values space.
            H (`float`): Parameter of uniform spacing between the weight vectors; *H = 1 / delta*
                should be integer - a number of divisions along an objective coordinate axis.
        
        Returns:
            weights (`list`): weight vectors (`np.ndarrays`), introduced to decompose the optimization problem into 
                several subproblems by dividing Pareto frontier into a number of sectors.
        """

        def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
            if depth == len(ref_dir) - 1:
                ref_dir[depth] = beta / (1.0 * n_partitions)
                ref_dirs.append(ref_dir[None, :])
            else:
                for i in range(beta + 1):
                    ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                    das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)

        if H == 0:
            return np.full((1, weights_num), 1 / weights_num)
        else:
            ref_dirs = []
            ref_dir = np.full(weights_num, np.nan)
            das_dennis_recursion(ref_dirs, ref_dir, H, H, 0)
            return np.concatenate(ref_dirs, axis=0)

    @staticmethod
    def get_number_of_points(weights_num, H):
        def factorial(n: int):
            if (n == 0 or n == 1):
                return 1
            else:
                return n * factorial(n - 1)

        def binomial_coefficient(n, k):
            return factorial(n) / (factorial(k) * factorial(n - k))

        return int(binomial_coefficient(H + weights_num - 1, weights_num - 1))

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
        
    def plot_pareto(self, dimensions:list, **visualizer_kwargs):
        assert len(dimensions) == 2, 'Current approach supports only two dimensional plots'
        visualizer = ParetoVisualizer(self.pareto_levels)
        visualizer.plot_pareto_mt(dimensions = dimensions, **visualizer_kwargs)