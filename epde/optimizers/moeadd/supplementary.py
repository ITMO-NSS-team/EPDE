"""

Supplementary procedures for the moeadd optimizer. 

Contains:
---------

**check_dominance(target, compared_with) -> bool** : Function to check, if one
solution is dominated by another;

**ndl_update(new_solution, levels) -> list** : Function to add a new solution into
the existing levels;

**fast_non_dominated_sorting(population) -> list** : Sorting of a population into
the non-dominated levels. The faster alternative to the **slow_non_dominated_sorting**.
Computational complexity :math:`O(MN^2)`, where *N* is the population size, and *M* is 
the number of objective functions.

**slow_non_dominated_sorting(population) -> list** : Sorting of a population into
the non-dominated levels. Computational complexity :math:`O(MN^3)`, where *N* is 
the population size, and *M* is the number of objective functions.

**acute_angle(vector_a, vector_b) -> float** : calculation of acute angle between two vectors.

**Constraint** abstract class with **Equality** and **Inequality** implementations.

"""

from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod

from epde.supplementary import rts

def check_dominance(target, compared_with) -> bool:
    """
    Determines if one solution dominates another based on their objective function values.
    
    This function is crucial for identifying Pareto-optimal solutions within the evolutionary process.
    By comparing objective values, the algorithm can effectively navigate the search space towards
    solutions that represent the best trade-offs between different objectives.
    
    Args:
        target (`src.moeadd.moeadd_solution_template.MOEADDSolution`): A solution to be evaluated for dominance.
        compared_with (`src.moeadd.moeadd_solution_template.MOEADDSolution`): The solution against which the target is compared.
    
    Returns:
        `bool`: True if the `target` solution dominates the `compared_with` solution (i.e., it is at least as good in all objectives and strictly better in at least one), False otherwise.
    """
    flag = False

    sdn = 5 # Number of significant digits
    for obj_fun_idx in range(len(target.obj_fun)):
        if rts(target.obj_fun[obj_fun_idx], sdn) <= rts(compared_with.obj_fun[obj_fun_idx], sdn):
            if rts(target.obj_fun[obj_fun_idx], sdn) < rts(compared_with.obj_fun[obj_fun_idx], sdn):
                flag = True
        else:
            return False
    return flag


def ndl_update(new_solution, levels) -> list:   # efficient_ndl_update
    """
    Computationally-cheap method of adding new solution into the existing Pareto levels.

    Args:
    """
    Efficiently integrates a new solution into existing Pareto levels by considering dominance relationships.
    
        This method strategically places a new solution within the non-dominated levels, 
        ensuring the Pareto optimality is maintained. It identifies where the new solution 
        fits best based on dominance checks with existing solutions in each level.
    
        Args:
            new_solution (`src.moeadd.moeadd_solution_template.MOEADDSolution`): The solution to be added.
            levels (`list`): A list of lists, where each sublist represents a non-dominated level containing
                `src.moeadd.moeadd_solution_template.MOEADDSolution` objects.
    
        Returns:
            `list`: A new list of lists representing the updated non-dominated levels, with the `new_solution`
            integrated appropriately. The structure mirrors the input `levels` argument.
    
        Notes:
            The idea for this method was introduced in *K. Li, K. Deb, Q. Zhang, and S. Kwong, 
            “Efficient non-domination level update approach for steady-state evolutionary 
            multiobjective optimization,” Dept. Electr. Comput. Eng., Michigan State Univ., 
            East Lansing, MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
    """
        new_solution (`src.moeadd.moeadd_solution_template.MOEADDSolution`):  case-specific subclass object
            The solution, that is to be added onto the non-dominated levels.
        levels (`list`): List of lists of ``src.moeadd.moeadd_solution_template.MOEADDSolution`` case-specific subclass 
            object, representing the input non-dominated levels.

    Returns:
        new_levels (`list`): List of lists of ``src.moeadd.moeadd_solution_template.MOEADDSolution`` case-specific subclass 
            object, containing the solution from input parameter *level* with *new_solution*, added to it.

    Notes:
        The idea for this method was introduced in *K. Li, K. Deb, Q. Zhang, and S. Kwong, 
        “Efficient non-domination level update approach for steady-state evolutionary 
        multiobjective optimization,” Dept. Electr. Comput. Eng., Michigan State Univ., 
        East Lansing, MI, USA, Tech. Rep. COIN No. 2014014, 2014.*

    """
    moving_set = {new_solution}
    new_levels = deepcopy(levels)  # levels# CAUSES ERRORS DUE TO DEEPCOPY

    for level_idx in np.arange(len(levels)):
        moving_set_new = set()
        for ms_idx, moving_set_elem in enumerate(moving_set):
            if np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]):
                moving_set_new.add(moving_set_elem)
            elif (not np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]) and
                  not np.any([check_dominance(moving_set_elem, solution) for solution in new_levels[level_idx]])):
                new_levels[level_idx].append(moving_set_elem)
            elif np.all([check_dominance(moving_set_elem, solution) for solution in levels[level_idx]]):
                temp_levels = new_levels[level_idx:]
                new_levels[level_idx:] = []
                new_levels.append([moving_set_elem,])
                new_levels.extend(temp_levels)  # ; completed_levels = True
            else:
                dominated_level_elems = [level_elem for level_elem in new_levels[level_idx] if check_dominance(
                    moving_set_elem, level_elem)]
                non_dominated_level_elems = [
                    level_elem for level_elem in new_levels[level_idx] if not check_dominance(moving_set_elem, level_elem)]
                non_dominated_level_elems.append(moving_set_elem)
                new_levels[level_idx] = non_dominated_level_elems

                for element in dominated_level_elems:
                    moving_set_new.add(element)
        moving_set = moving_set_new
        if not len(moving_set):
            break
    if len(moving_set):
        new_levels.append(list(moving_set))
    if len(new_levels[len(new_levels)-1]) == 0:
        _ = new_levels.pop()
    return new_levels


def fast_non_dominated_sorting(population) -> list:
    """
    Procedure to classify solutions into distinct non-dominated sets based on their objective values.
    
        This method efficiently identifies Pareto-optimal solutions by iteratively assigning individuals to different fronts.
        It maintains a count of dominating solutions for each individual and a list of solutions dominated by each individual.
        This approach reduces computational complexity compared to naive methods, making it suitable for larger populations.
        This function is a faster alternative to the ``slow_non_dominated_sorting``, but requires 
        a little more memory to store indexes of elements, dominated by every solution. This 
        method was introduced in *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast 
        and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
        vol. 6, no. 2, pp. 182–197, Apr. 2002.* The computational complexity of the method is 
        :math:`O(MN^2)`, where *N* is the population size, and *M* is the number of objective 
        functions in comparisson with :math:`O(MN^3)` of the straightforward way.
    
        Args:
            population (list): A list of individuals, where each individual is represented by its objective values.
    
        Returns:
            list: A list of lists, where each inner list represents a non-dominated front. The first front (index 0) contains the Pareto-optimal solutions.
    """

    levels = []
    ranks = np.empty(len(population))
    # Число элементов, доминирующих над i-ым кандидиатом
    domination_count = np.zeros(len(population))
    # Индексы элементов, над которыми доминирует i-ый кандидиат
    dominated_solutions = [[] for elem_idx in np.arange(len(population))]
    current_level_idxs = []
    for main_elem_idx in np.arange(len(population)):
        for compared_elem_idx in np.arange(len(population)):
            if main_elem_idx == compared_elem_idx:
                continue
            if check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                domination_count[main_elem_idx] += 1
            elif check_dominance(population[main_elem_idx], population[compared_elem_idx]):
                dominated_solutions[main_elem_idx].append(compared_elem_idx)
        if domination_count[main_elem_idx] == 0:
            current_level_idxs.append(main_elem_idx)
            ranks[main_elem_idx] = 0
    levels.append([population[elem_idx] for elem_idx in current_level_idxs])

    level_idx = 0
    while len(current_level_idxs) > 0:
        new_level_idxs = []
        for main_elem_idx in current_level_idxs:
            for dominated_elem_idx in dominated_solutions[main_elem_idx]:
                domination_count[dominated_elem_idx] -= 1
                if domination_count[dominated_elem_idx] == 0:
                    ranks[dominated_elem_idx] = level_idx + 1
                    new_level_idxs.append(dominated_elem_idx)

        if len(new_level_idxs):
            levels.append([population[elem_idx]
                          for elem_idx in new_level_idxs])
        level_idx += 1
        current_level_idxs = new_level_idxs  # deepcopy(new_level_idxs)
    return levels


def slow_non_dominated_sorting(population) -> list:
    """
    Procedure to classify a population into non-dominated levels, identifying Pareto fronts.
    
        This function iteratively identifies and separates individuals into distinct non-dominated levels.
        Each level represents a Pareto front, containing solutions that are not dominated by any other
        solution in the population (excluding those already assigned to previous levels).
        This process continues until all individuals are assigned to a level.
        The computational complexity of this sorting algorithm is :math:`O(MN^3)` in the worst case,
        where *N* is the population size and *M* is the number of objective functions.
        This classification helps to identify the best trade-offs within the population,
        which is crucial for constructing equation structures that accurately represent the underlying dynamics
        of the system being modeled.
    
        Args:
            population (`list`): The input population, represented as a list of individuals.
    
        Returns:
            levels (`list`): List of lists of population elements. The outer index represents the Pareto front
                number (e.g., 0-th is the current Pareto frontier), while the inner index indicates the
                position of an element within that level.
    """
    locked_idxs = []
    levels = []
    levels_elems = 0
    while len(population) > levels_elems:
        processed_idxs = []
        for main_elem_idx in np.arange(len(population)):
            if not main_elem_idx in locked_idxs:
                dominated = False
                for compared_elem_idx in np.arange(len(population)):
                    if main_elem_idx == compared_elem_idx or compared_elem_idx in locked_idxs:
                        continue
                    if check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                        dominated = True
                if not dominated:
                    processed_idxs.append(main_elem_idx)
        locked_idxs.extend(processed_idxs)
        levels_elems += len(processed_idxs)
        levels.append([population[elem_idx] for elem_idx in processed_idxs])
    return levels


def acute_angle(vector_a, vector_b) -> float:
    """
    Calculates the acute angle between two vectors.
    
        This function is crucial for determining the relationships between different terms and vectors within the equation discovery process. By calculating the angles, the algorithm can assess the similarity and independence of various components, aiding in the selection of the most relevant terms for the final equation.
    
        Args:
            vector_a (np.ndarray): The first vector, representing a term or component in the equation.
            vector_b (np.ndarray): The second vector, representing another term or component in the equation.
    
        Returns:
            float: The acute angle in radians between the two vectors. This value is used to evaluate the correlation and redundancy between terms.
    """
    cos_val = np.dot(vector_a, vector_b)/(np.sqrt(np.dot(vector_a, vector_a))*np.sqrt(np.dot(vector_b, vector_b)))
    if np.abs(cos_val) > 1.:
        cos_val = np.sign(cos_val)
    return np.arccos(cos_val)


class Constraint(ABC):
    """
    Represents a constraint on the search space for equation discovery. It defines conditions that candidate equations must satisfy to be considered valid. This class serves as a base for implementing various types of constraints, such as limiting the number of terms, restricting the types of operations, or enforcing specific structural properties.
    
    
        The abstract class for the constraint. Noteable subclasses: Inequality & Equality.
    """

    @abstractmethod
    def __init__(self, *args):
        """
        Initializes the constraint object.
        
        This abstract method serves as a blueprint for initializing specific constraint types
        within the equation discovery process. Subclasses should override this method to
        define the constraint's parameters and behavior.
        
        Args:
            *args: Variable length argument list. Arguments passed to the constraint.
        
        Returns:
            None.
        
        Why:
        This initialization is a part of defining constraints on the discovered equations.
        Constraints are used to guide the search process towards more physically meaningful
        or mathematically stable solutions.
        """
        pass

    @abstractmethod
    def __call__(self, *args):
        """
        Applies the constraint to the provided arguments.
        
        This abstract method serves as the core logic for evaluating the constraint
        defined by a subclass against a given set of arguments. Subclasses must
        implement this method to define the specific constraint behavior.
        
        Args:
            *args: Variable-length argument list representing the values to which
                the constraint is applied.
        
        Returns:
            None. This method's behavior is defined by its side effects within
            subclasses, such as modifying internal state or raising exceptions
            if the constraint is violated.
        
        Why:
        This method enables the framework to evaluate candidate equation structures
        by checking if they satisfy specific conditions or limitations imposed
        during the equation discovery process.
        """
        pass


class Inequality(Constraint):
    """
    Represents an inequality constraint. Assumes the format :math:`g(x) >= 0`. Subclass of Constraint.
    
    
    Args:
        g (`function (lambda function)`): The constraint function, which shall penalize the candidate solution, if the value of
            :math:`g(x) >= 0` is not fulfilled (is less, than 0). The penalty is equal to 
            the absolute value of constraint violation.
    
    Methods:
        __call__(self, x) : returns float
            Overloaded call operator returns the value of constaint violation.
    """


    def __init__(self, g):
        """
        Initializes the Inequality object with a graph representing inequality relationships.
        
        This graph is used to efficiently manage and traverse inequality constraints
        during the equation discovery process, ensuring that discovered equations
        adhere to the specified relationships.
        
        Args:
            g: A graph object representing inequality relationships between terms.
        
        Returns:
            None.
        
        Class Fields:
            _g (object): The graph object associated with this instance,
                         representing inequality constraints.
        """
        self._g = g

    def __call__(self, x) -> float:
        """
        Evaluates the constraint violation for a given candidate solution.
        
        This method determines the degree to which a candidate solution violates the defined inequality constraint. It returns a non-zero value only when the constraint is not satisfied, quantifying the extent of the violation. This is crucial for the evolutionary algorithm to penalize solutions that do not adhere to the problem's constraints, guiding the search towards feasible regions.
        
        Args:
            x (`np.ndarray`): The candidate solution's values (gene representation).
        
        Returns:
            `float`: The constraint violation value. Returns the absolute value of g(x) if g(x) < 0, otherwise 0.
        """
        return - self._g(x) if self._g(x) < 0 else 0


class Equality(Constraint):
    """
    Represents an equality constraint. Assumes the format :math:`h(x) = 0`. Subclass of Constraint.
    
    
    Args:
        h (`function (lambda function)`): The constraint function, which shall be penalized, if the value does not match with
            the const
    
    Methods:
        __call__(self, x) : returns float
            Overloaded call operator returns the value of constaint violation.
    """


    def __init__(self, h):
        """
        Initializes the object's height, a crucial parameter for comparing instances within the equation discovery process.
        
        The height attribute influences how the evolutionary algorithm evaluates and selects equation candidates.
        
        Args:
            h (float): The height value to be stored.
        
        Returns:
            None.
        
        Class Fields:
            _h (float): The height of the object.
        """
        self._h = h

    def __call__(self, x) -> float:
        """
        Evaluates the constraint violation of a candidate solution for equality constraints.
        
        This method quantifies how well a candidate solution satisfies the equality constraint 
        defined by :math:`h(x) = 0`. It returns the absolute value of :math:`h(x)`, 
        representing the magnitude of the violation. This value is used by the evolutionary 
        algorithm to guide the search towards solutions that better satisfy the constraint.
        
        Args:
            x (`np.ndarray`): Values of the candidate solution, representing its gene.
        
        Returns:
            `float`: The constraint violation value, which is the absolute value of :math:`h(x)`.
                     A value of 0 indicates that the constraint is perfectly satisfied.
        """
        return np.abs(self._h(x))
