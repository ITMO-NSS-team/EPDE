"""

Main classes and functions of the moeadd optimizer.

"""

import numpy as np
import warnings
from itertools import chain

from typing import Union, List

def flatten_chain(matrix):
    """
    Flattens a 2D matrix (list of lists) into a 1D list.
    
    This operation is crucial for preparing data structures
    used in the equation discovery process, where a flattened
    representation is often required for efficient processing
    by evolutionary algorithms and other optimization techniques.
    
    Args:
        matrix (list[list]): A 2D list (list of lists) that needs to be flattened.
    
    Returns:
        list: A 1D list containing all elements from the input matrix.
    """
    return list(chain.from_iterable(matrix))

# from copy import deepcopy
# from functools import reduce

import epde.globals as global_var
from epde.structure.main_structures import SoEq
from epde.optimizers.moeadd.population_constr import SystemsPopulationConstructor
from epde.optimizers.moeadd.vis import ParetoVisualizer

from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
from epde.optimizers.moeadd.supplementary import fast_non_dominated_sorting, ndl_update, Equality, Inequality
from scipy.spatial import ConvexHull    

def clear_list_of_lists(inp_list) -> list:
    """
    Removes empty lists from a list of lists.
    
        This function is used to clean up the equation structures during the evolutionary process,
        ensuring that only valid and meaningful equations are considered for further evaluation and optimization.
        Empty lists can arise from various genetic operations or simplification steps, and removing them
        helps to maintain the integrity and efficiency of the search.
    
        Args:
            inp_list (list): A list of lists.
    
        Returns:
            list: A new list containing only the lists from the input list that have a length greater than zero.
    """
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
                 update_method = ndl_update, initial_sort = False):
        """
        Args:
                    population (`list`): List of candidate solutions, each representing a potential differential equation model.
                    sorting_method (`callable`): Method used to rank solutions based on their non-domination level. Defaults to fast non-dominated sorting.
                    update_method (`callable`): Method for adding new solutions to the Pareto levels. Defaults to NDL update.
                    initial_sort (`bool`): Flag indicating whether to perform an initial sorting of the population. Defaults to False.
        
                Returns:
                    None
        
                Why:
                    This class organizes a population of candidate differential equation models into Pareto levels based on their performance metrics. The sorting and update methods allow for efficient management and evolution of these levels during the equation discovery process.
        """
        self._sorting_method = sorting_method
        self.population = []
        self._update_method = update_method
        self.unplaced_candidates = population
        self.history = set()
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Manually reconstructs a specific attribute of the object.
        
                This method facilitates the restoration of object attributes, primarily the 'population', from a serialized state. It ensures that the population is correctly initialized with `SoEq` objects, populated with their respective data, to maintain the state of discovered equations. This is crucial when loading previously discovered equation sets. After reconstruction, the levels are sorted to reflect the updated population.
                
                Args:
                    attribute (str): The attribute to reconstruct. Currently, only 'population' is supported.
                    value (list): The data used to reconstruct the attribute. For 'population', it's a list of dictionaries, each representing an equation.
                    except_attrs (dict): A dictionary of attributes to exclude during the reconstruction.
                
                Returns:
                    None. The method modifies the object's state in place.
                
                Class Fields (initialized or modified):
                    population (list): A list of `SoEq` objects representing the population. Initialized or overwritten during 'population' attribute reconstruction.
                    levels (list): A sorted list of levels, updated by the `sort` method.
        """
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
        """
        Populates the object's attributes from a dictionary, excluding specified keys.
        
                This method streamlines the assignment of attributes to the object
                by directly mapping dictionary key-value pairs to the object's
                internal dictionary, excluding any keys specified in `except_keys`.
                This ensures that the object's state is efficiently updated based on
                the provided data, which is essential for representing different
                Pareto levels and their corresponding equation structures.
        
                Args:
                    attributes (dict): A dictionary containing attribute names and their values.
                    except_keys (list, optional): A list of keys to exclude from being set as attributes.
                        Defaults to ['obj_type'].
        
                Returns:
                    None: This method modifies the object's attributes in place.
        """
        self.__dict__ = {key : item for key, item in attributes.items()
                         if key not in except_keys}
    
    @property
    def levels(self):
        """
        Returns the Pareto levels of the solutions.
        
                These levels represent the non-dominated sets of solutions at each stage of the Pareto optimization process.
                They are used to assess the trade-offs between different objectives and to guide the search towards the Pareto front.
        
                Returns:
                    list: The Pareto levels, where each level contains a set of non-dominated solutions.
        """
        return self._levels
    
    @levels.setter
    def levels(self, value : list):
        """
        Sets the levels to define the Pareto front. These levels determine the different sets of non-dominated solutions.
        
                Args:
                    value (list): A list representing the levels.
        
                Returns:
                    None
        
                Initializes:
                    _levels (list): The levels of the object.
        
                Why:
                    This method is essential for defining the Pareto front, which is a crucial step in multi-objective optimization. By setting the levels, we can identify different sets of non-dominated solutions, allowing for a more comprehensive analysis of the trade-offs between objectives.
        """
        self._levels = value
    
    def __len__(self):
        """
        Returns the number of individuals currently present in the population.
        
                This reflects the size of the population being evolved and optimized
                to discover governing differential equations.
        
                Args:
                    self: The ParetoLevels instance.
        
                Returns:
                    int: The number of individuals in the population.
        """
        return len(self.population)
    
    def __iter__(self):
        """
        Returns an iterator for traversing the Pareto levels.
        
                This allows iterating through the discovered equation structures,
                organized by their Pareto dominance, to explore the trade-offs
                between different model complexities and accuracies.
        
                Args:
                    None
        
                Returns:
                    ParetoLevelsIterator: An iterator that yields Pareto levels,
                                          each containing a set of non-dominated equations.
        """
        return ParetoLevelsIterator(self)
    
    def initial_placing(self):
        """
        Places initial candidates into the population for the Pareto optimization process.
        
        This method moves all candidates from the unplaced pool to the main population,
        preparing them for the initial sorting into Pareto levels. This ensures that
        all provided initial guesses are considered during the equation discovery process.
        
        Args:
            None
        
        Returns:
            None
        """
        while self._unplaced_candidates:
            self.population.append(self._unplaced_candidates.pop())
        # if any([any([candidate == other_candidate for other_candidate in self.population[:idx] + self.population[idx+1:]])
                # for idx, candidate in enumerate(self.population)]):
            # print([candidate.text_form for candidate in self.population])
            # raise Exception('Duplicating initial candidates')
        self.levels = self.sort()

    def sort(self):
        """
        Sorts the population into Pareto non-dominated levels to identify a diverse set of equation candidates that represent different trade-offs between accuracy and complexity. This ensures that the evolutionary process explores a wide range of potential solutions.
        
                Args:
                    self: The ParetoLevels object containing the population to be sorted.
        
                Returns:
                    A list of Pareto levels, where each level contains a set of non-dominated individuals (equations).
        """
        # self.levels = self._sorting_method(self.population)
        return self._sorting_method(self.population)
    
    @property
    def unplaced_candidates(self):
        """
        Returns the set of unplaced candidates.
        
        This property provides access to the candidates that haven't been assigned yet,
        allowing the algorithm to efficiently explore the search space of possible equation structures.
        These candidates represent potential terms or components that can be added to the equation.
        
        Args:
            self: The object instance.
        
        Returns:
            set: The set of unplaced candidates.
        """
        return self._unplaced_candidates
    
    @unplaced_candidates.setter
    def unplaced_candidates(self, candidates : Union[list, set, tuple]):
        """
        Initializes the set of candidates that have not yet been assigned to a Pareto level.
        
        This initialization is crucial for the algorithm to keep track of which candidates still need to be evaluated and assigned during the Pareto front construction process.
        
        Args:
            candidates (Union[list, set, tuple]): A collection of candidates that have not yet been placed on any Pareto level.
        
        Returns:
            None
        """
        
        # ADD EXTRA CHECKS IF NECESSARY
        self._unplaced_candidates = candidates
    
    def update(self, point):
        """
        Adds a candidate solution to the Pareto levels and population.
        
        This method integrates a new solution into the existing set of non-dominated solutions,
        refining the approximation of the Pareto front. By maintaining a diverse and well-distributed
        set of solutions, it contributes to the exploration of the solution space and the
        identification of optimal trade-offs between conflicting objectives.
        
        Args:
            point (`MOEADDSolution`): The candidate solution to be added.
        
        Returns:
            None
        """
        self.levels = self._update_method(point, self.levels)
        self.population.append(point)
 
    def delete_point(self, point):
        """
        Deletes a candidate solution from the Pareto levels and population.
        
                This ensures that the Pareto levels and population accurately reflect the current set of non-dominated solutions after a candidate has been processed or determined to be unsuitable. This is important for maintaining the integrity of the Pareto front approximation during the evolutionary search process.
        
                Args:
                    point (`MOEADDSolution`): The solution to be removed.
        
                Returns:
                    None
        """
        new_levels = []
        history = []
        for level in self.levels:
            temp = []
            for element in level:
                if not np.allclose(element.obj_fun, point.obj_fun) or any(np.allclose(element.obj_fun, h) for h in history):
                    temp.append(element)
                history.append(element.obj_fun)
            if not len(temp) == 0:
                new_levels.append(temp)

        population_cleared = []
        history = []
        for elem in self.population:
            if not np.allclose(elem.obj_fun, point.obj_fun) or any(np.allclose(elem.obj_fun, h) for h in history):
                population_cleared.append(elem)
            history.append(elem.obj_fun)
                
        if len(population_cleared) != sum([len(level) for level in new_levels]):
            print(len(population_cleared), len(self.population), sum([len(level) for level in new_levels]))
            print('initial population', [solution.vals for solution in self.population], len([solution.vals for solution in self.population]), '\n')
            print('cleared population', [solution.vals for solution in population_cleared], len([solution.vals for solution in self.population]), '\n')
            print(point.vals)
            raise Exception('Deleted something extra')
        self.levels = new_levels
        self.population = population_cleared

    def get_stats(self):
        """
        Collect objective function values from all discovered equations.
                This is done to evaluate and compare the performance 
                of different equation candidates found at various levels of the search process.
                
                Returns:
                    np.ndarray: A NumPy array containing the objective function values
                    from all equation candidates across all levels.
        """
        return np.array(flatten_chain([[element.obj_fun for element in level] 
                                       for level in self.levels]))

    def fit_convex_hull(self):
        """
        Fits a convex hull to the Pareto front of the population.
        
                This method calculates the convex hull of the objective function values of the solutions in the population.
                It is used to approximate the Pareto front, which represents the set of non-dominated solutions.
                The convex hull provides a computationally efficient way to estimate the extent and shape of the Pareto front,
                facilitating analysis and visualization of the trade-offs between different objectives.
        
                Args:
                    None
        
                Returns:
                    None
        """
        if len(self.levels) > 1:
            warnings.warn('Algorithm has not converged to a single Pareto level yet!')
        points = np.vstack([sol.obj_fun for sol in self.population])
        points = np.concatenate((points, np.max(points, axis = 0).reshape((1, -1))))
        points_unique = np.unique(points, axis = 0)
        
        self.hull = ConvexHull(points = points_unique, qhull_options='Qt')

    def get_by_complexity(self, complexity):
        """
        Retrieves solutions that satisfy a specific complexity requirement.
        
        This method filters the solutions within the first Pareto level to identify those that align with the given complexity.
        This is useful for exploring the trade-offs between model complexity and accuracy, allowing users to select solutions that are both parsimonious and well-fitting.
        
        Args:
            complexity (int): The desired complexity level of the solutions.
        
        Returns:
            list: A list of solutions from the first Pareto level that match the specified complexity.
        """
        matching_solutions = [solution for solution in self.levels[0] 
                              if solution.matches_complexitiy(complexity)]
        return matching_solutions        

class ParetoLevelsIterator(object):
    """
    Class for iteration by object of Pareto Levels
    """

    def __init__(self, pareto_levels):
        """
        Initializes the iterator with a sequence of Pareto levels.
        
        This iterator facilitates traversal through a set of Pareto levels,
        enabling access to increasingly refined approximations of the Pareto front.
        The Pareto front represents the set of optimal trade-offs between
        competing objectives in the equation discovery process.
        
        Args:
            pareto_levels: A sequence of Pareto levels, each representing a
                successive refinement of the Pareto front.
        
        Returns:
            None
        
        Class Fields:
            _levels: The sequence of Pareto levels to iterate over.
            _idx: The index of the current Pareto level in the sequence.
        """
        self._levels = pareto_levels
        self._idx = 0

    def __next__(self):
        """
        Returns the next individual from the population.
        
        Iterates through the population of the current level. This is done to explore the solution space level by level, ensuring that diverse and non-dominated solutions are considered during the evolutionary process.
        
        Args:
            self: The object instance.
        
        Returns:
            Individual: The next individual in the population.
        
        Raises:
            StopIteration: If the end of the population is reached.
        """
        if self._idx < len(self._levels.population):
            res = self._levels.population[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration


class MOEADDOptimizer(object):
    """
    Solves multi-objective optimization problems using an evolutionary algorithm. This class focuses on unconstrained problem variations. It aims to find a set of solutions that represent the best trade-offs between multiple objective functions.
    
    
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

    def __init__(self, population_instruct, weights_num, pop_size, solution_params,
                 delta: float, neighbors_number: int,
                 nds_method = fast_non_dominated_sorting, ndl_update = ndl_update, 
                 passed_population: Union[List, ParetoLevels] = None):
        """
        Initializes the MOEADDOptimizer, creating an initial population of candidate solutions and a set of weight vectors.
        
                The population is divided into Pareto non-dominated levels, ensuring diversity and exploration of the objective space.
                Weight vectors guide the search process by decomposing the multi-objective problem into a set of single-objective subproblems,
                each associated with a specific region of the objective space. This initialization is crucial for the optimizer to effectively
                explore the solution space and identify a well-distributed Pareto front.
        
                Args:
                    population_instruct (dict): Parameters for individual creation within the population.
                    weights_num (int): The number of weight vectors used to divide the objective function space.  Often matches the population size.
                    pop_size (int): The size of the candidate solution population.
                    solution_params (dict): Parameters passed to each newly created solution during initialization.
                    delta (float): Parameter controlling the uniform spacing between weight vectors; *H = 1 / delta* should be an integer.
                    neighbors_number (int): The number of neighboring weight vectors considered during evolutionary operations.
                    nds_method (callable, optional): Method for non-dominated sorting of candidate solutions. Defaults to ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``.
                    ndl_update (callable, optional): Method for efficiently updating non-dominated levels when adding new solutions. Defaults to ``moeadd.moeadd_supplementary.ndl_update``.
                    passed_population (Union[List, ParetoLevels], optional): Initial population to use. If None, a new population is created. Defaults to None.
        
                Returns:
                    None
        """
        assert weights_num == pop_size, 'Each individual in population has to correspond to a sector'
        self.abbreviated_search_executed = False
        soluton_creation_attempts= {'softmax' : 10,
                                    'hardmax' : 100}

        assert (type(solution_params) == type(None) or
                 type(solution_params) == dict), 'The solution parameters, passed into population constructor must be in dictionary'

        pop_constructor = SystemsPopulationConstructor(**population_instruct)

        if (passed_population is None) or isinstance(passed_population, list):
            population = [] if passed_population is None else passed_population
            psize = len(population)
            for solution_idx in range(psize):
                population[solution_idx].set_domain(solution_idx)
                pop_constructor.applyToPassed(population[solution_idx], **solution_params)

            for solution_idx in range(pop_size - psize):
                solution_gen_idx = 0
                while True:
                    if type(solution_params) == type(None): solution_params = {}
                    temp_solution = pop_constructor.create(**solution_params)
                    temp_solution.set_domain(psize + solution_idx)
                    if not np.any([temp_solution == solution for solution in population]):
                        population.append(temp_solution)
                        print(f'New solution accepted, confirmed {len(population)}/{pop_size} solutions.')
                        break
                    if solution_gen_idx == soluton_creation_attempts['softmax'] and global_var.verbose.show_warnings:
                        print('solutions tried:', solution_gen_idx)
                        warnings.warn('Too many failed attempts to create unique solutions for multiobjective optimization.\
                                      Change solution parameters to allow more diversity.')
                    if solution_gen_idx == soluton_creation_attempts['hardmax']:
                        population.append(temp_solution)
                        print(f'New solution accepted, despite being a dublicate of another solution.\
                              Confirmed {len(population)}/{pop_size} solutions.')
                        break
                    solution_gen_idx += 1
            self.pareto_levels = ParetoLevels(population, sorting_method = nds_method, update_method = ndl_update,
                                              initial_sort = False)
        else:
            if not isinstance(passed_population, ParetoLevels):
                raise TypeError(f'Incorrect type of the population passed. Expected ParetoLevels object, instead got \
                                 {type(passed_population)}')
            self.pareto_levels = passed_population
                                 
        self.weights = []; weights_size = len(population[0].obj_funs) #np.empty((pop_size, len(optimized_functionals)))
        for weights_idx in range(weights_num):
            temp_weights = self.weights_generation(weights_size, delta)
            while temp_weights in self.weights:
                temp_weights = self.weights_generation(weights_size, delta)
            self.weights.append(temp_weights)
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
        Initializes the Pareto levels based on the provided population and methods, then warns if population is too small.
        
        This method constructs the Pareto levels structure, which is essential for multi-objective optimization.
        It uses the provided sorting and update methods to organize the population into non-dominated fronts.
        If the population size is small, a warning is issued, indicating that the search may be limited due to insufficient diversity.
        
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
        Method to generate a set of weight vectors used to guide the search towards different regions of the Pareto front. These vectors define subproblems, each focusing on a specific trade-off between objectives, thereby promoting diversity in the solutions found.
        
                Args:
                    weights_num (`int`): The number of weight vectors to generate, determining the granularity of the Pareto front exploration.
                    delta (`float`): A parameter controlling the spacing between weight vectors, influencing the uniformity of the search. It is required that *H = 1 / delta* is an integer.
        
                Returns:
                    `list`: A list of weight vectors (`np.ndarrays`), each representing a different preference for the objectives and guiding the evolutionary search towards diverse solutions on the Pareto front.
        """
        weights = np.empty(weights_num)
        assert 1./delta == round(1./delta) # check, if 1/delta is integer number
        m = np.zeros(weights_num)
        for weight_idx in np.arange(weights_num):
            weights[weight_idx] = np.around(np.random.choice([div_idx * delta for div_idx in np.arange(1./delta + 1e-8 - np.sum(m[:weight_idx + 1]))]), 2)
            m[weight_idx] = weights[weight_idx]/delta
        weights[-1] = np.around(1 - np.sum(weights[:-1]), 2)
        
        weights = np.abs(weights)
        return list(weights)

    
    def pass_best_objectives(self, *args) -> None:
        """
        Updates the current best objective values observed during the optimization process.
        
                This method is crucial for tracking the progress of the MOEA/D algorithm, ensuring that the best-performing objective values are recorded for comparison and selection during the evolutionary search. It iterates through the provided objective values and updates the internal `best_obj` attribute with the improved values. This ensures that the algorithm maintains a record of the best solutions encountered so far, guiding the search towards the Pareto front.
        
                Args:
                    *args (`np.ndarray|list`): The values of the objective functions for a candidate solution in the many-objective optimization problem. The number of arguments must match the number of objective functions.
        
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
        Sets the sector processor for the MOEADDOptimizer.
        
        This operator is responsible for driving the evolutionary search within each sector, 
        guiding the optimization process towards identifying suitable equation structures.
        
        Args:
            processer (`MOEADDSectorProcesser`): The sector processor to be used.
        
        Returns:
            `None`
        """
        self.sector_processer = processer


    def set_strategy(self, strategy_director):
        """
        Sets the sector processing strategy.
        
        This method configures the sector processor using a strategy director, enabling dynamic adaptation of the optimization process to different problem characteristics. By employing the director's builder to assemble a specific processing strategy, the optimizer can tailor its search behavior to enhance performance and convergence.
        
        Args:
            strategy_director: The director responsible for building the strategy.
        
        Returns:
            None.
        """
        builder = strategy_director.builder
        builder.assemble(True)
        self.set_sector_processer(builder.processer)
    
        
    def optimize(self, epochs):
        """
        Method for performing evolutionary optimization to refine the Pareto levels. This process iteratively improves the set of non-dominated solutions by exploring different equation structures and parameter values. The optimization output, representing the discovered equations and their corresponding fitness values, can be accessed through the ``optimizer.pareto_level`` object and its attributes ``.levels`` or ``.population``.
        
                Args:
                    epochs (`int`): The maximum number of evolutionary optimization iterations.
        
                Returns:
                    None. The optimization results are stored within the `pareto_levels` attribute of the optimizer.
        
                Note:
                    The optimization process terminates early if the algorithm converges to a single Pareto frontier, indicating that further iterations are unlikely to yield significant improvements.
        
                Why:
                    This optimization refines the Pareto levels by iteratively processing subsets of the population, guided by different weight vectors. This approach allows the algorithm to explore the search space more effectively and identify a diverse set of non-dominated solutions, each representing a potential equation structure that fits the data well.
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
        """
        Returns the search history, allowing inspection of the optimization process.
        
                This function provides a way to track the progress of the equation discovery process. By examining the history, users can understand how the search evolved and identify potential areas for improvement in the search strategy.
        
                Args:
                    best_only (bool, optional): If True, returns only the best equation scores found so far. If False, returns the complete history, including equation scores and corresponding parameters at each step. Defaults to True.
        
                Returns:
                    list: The history of the search. If `best_only` is True, returns a list of the best equation scores. Otherwise, returns the complete history as a list of tuples, where each tuple contains the best equation score and the corresponding parameters.
        """
        if best_only:
            return [elem[0] for elem in self._hist]
        else:
            return self._hist
        
    def plot_pareto(self, dimensions:list, **visualizer_kwargs):
        """
        Plots the Pareto front for two objectives, visualizing the trade-offs achieved by the multi-objective evolutionary algorithm.
        
                This method leverages the ParetoVisualizer to display the Pareto front,
                representing the non-dominated solutions found during the optimization
                process. By plotting the front, users can analyze the trade-offs between
                different objectives and select solutions that best meet their specific
                requirements. It currently supports only two-dimensional plots.
        
                Args:
                    dimensions: A list containing the two objective dimensions to be plotted.
                    visualizer_kwargs: Keyword arguments to be passed to the
                        ParetoVisualizer's plotting function, allowing customization of the plot's appearance.
        
                Returns:
                    None. The method displays a plot of the Pareto front.
        """
        assert len(dimensions) == 2, 'Current approach supports only two dimensional plots'
        visualizer = ParetoVisualizer(self.pareto_levels)
        # visualizer.plot_pareto(dimensions = dimensions, **visualizer_kwargs)
        visualizer.plot_pareto_mt(dimensions = dimensions, **visualizer_kwargs)
