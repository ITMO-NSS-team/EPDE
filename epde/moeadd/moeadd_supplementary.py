"""

Supplementary procedures for the moeadd optimizer. 

Contains:
---------

**check_dominance(target, compared_with) -> bool** : Function to check, if one
solution is dominated by another;

**NDL_update(new_solution, levels) -> list** : Function to add a new solution into
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
from abc import ABC, abstractproperty, abstractmethod
    

#def check_dominance(target, compared_with) -> bool:
#    return (all([target.obj_fun[obj_fun_idx] <= compared_with.obj_fun[obj_fun_idx] for obj_fun_idx in np.arange(target.obj_fun.size)]) and 
#            any([target.obj_fun[obj_fun_idx] < compared_with.obj_fun[obj_fun_idx] for obj_fun_idx in np.arange(target.obj_fun.size)]))
   
def check_dominance(target, compared_with) -> bool:
    '''
    
    Function to check, if one solution is dominated by another.
    
    Arguments:
    ----------
    
    target : ``src.moeadd.moeadd_stc.moeadd_solution`` case-specific subclass object
        The individual solution on the pareto levels, compared with the other element.
        
    compared_with : ``src.moeadd.moeadd_stc.moeadd_solution`` case-specific subclass object
        The individual solution on the pareto levels, with with the target is compared.
    
    Returns:
    --------
    
    domiated : bool
        Function returns True, if the **compared_with** dominates (has at least one objective
        functions with less values, while the others are the same) the **target**; 
        False in all other cases.
    
    '''
    flag = False
    for obj_fun_idx in range(len(target.obj_fun)):
        if target.obj_fun[obj_fun_idx] <= compared_with.obj_fun[obj_fun_idx]:
            if target.obj_fun[obj_fun_idx] < compared_with.obj_fun[obj_fun_idx]:
                flag = True
        else:
            return False
    return flag
 
def NDL_update(new_solution, levels) -> list:   # efficient_NDL_update
    '''
    
    Computationally-cheap method of adding new solution into the existing Pareto levels.
    
    Parameters:
    -----------
    
    new_solution : ``src.moeadd.moeadd_stc.moeadd_solution`` case-specific subclass object
        The solution, that is to be added onto the non-dominated levels.
        
    levels : list
        List of lists of ``src.moeadd.moeadd_stc.moeadd_solution`` case-specific subclass 
        object, representing the input non-dominated levels.
        
    Returns:
    --------
    
    new_levels : list
        List of lists of ``src.moeadd.moeadd_stc.moeadd_solution`` case-specific subclass 
        object, containing the solution from input parameter *level* with *new_solution*, 
        added to it.
    
    Notes:
    ------
    
    The idea for this method was introduced in *K. Li, K. Deb, Q. Zhang, and S. Kwong, 
    “Efficient non-domination level update approach for steady-state evolutionary 
    multiobjective optimization,” Dept. Electr. Comput. Eng., Michigan State Univ., 
    East Lansing, MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
    
    '''
    moving_set = {new_solution}
    new_levels = deepcopy(levels) #levels#
    for level_idx in np.arange(len(levels)):
        moving_set_new = set()
        for ms_idx, moving_set_elem in enumerate(moving_set):
            if np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]):
                moving_set_new.add(moving_set_elem)
            elif (not np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]) and 
                  not np.any([check_dominance(moving_set_elem, solution) for solution in new_levels[level_idx]])):
                new_levels[level_idx].append(moving_set_elem)#; completed_levels = True
            elif np.all([check_dominance(moving_set_elem, solution) for solution in levels[level_idx]]):
                temp_levels = new_levels[level_idx:]
                new_levels[level_idx:] = []
                new_levels.append([moving_set_elem,]); new_levels.extend(temp_levels)#; completed_levels = True
            else:
                dominated_level_elems = [level_elem for level_elem in new_levels[level_idx] if check_dominance(moving_set_elem, level_elem)]
                non_dominated_level_elems = [level_elem for level_elem in new_levels[level_idx] if not check_dominance(moving_set_elem, level_elem)]
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
    '''
    
    Procedure of separating points from the general population into non-dominated levels.
    This function is a faster alternative to the ``slow_non_dominated_sorting``, but requires 
    a little more memory to store indexes of elements, dominated by every solution. This 
    method was introduced in *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast 
    and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
    vol. 6, no. 2, pp. 182–197, Apr. 2002.* The computational complexity of the method is 
    :math:`O(MN^2)`, where *N* is the population size, and *M* is the number of objective 
    functions in comparisson with :math:`O(MN^3)` of the straightforward way.
    
    
    Parameters:
    -----------
    
    population : list
        The input population, represented as a list of individuals.
        
    Returns: 
    ----------
    
    levels : list
        List of lists of population elements. The outer index is the number of a layer 
        (e.g. 0-th is the current Pareto frontier), while the inner is the index of an element on a level.
    
    '''
    
    levels = []; ranks = np.empty(len(population))
    domination_count = np.zeros(len(population)) # Число элементов, доминирующих над i-ым кандидиатом
    dominated_solutions = [[] for elem_idx in np.arange(len(population))] # Индексы элементов, над которыми доминирует i-ый кандидиат
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
            current_level_idxs.append(main_elem_idx); ranks[main_elem_idx] = 0
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
    
        if len(new_level_idxs): levels.append([population[elem_idx] for elem_idx in new_level_idxs])
        level_idx += 1
        current_level_idxs = new_level_idxs#deepcopy(new_level_idxs)
    return levels

    
def slow_non_dominated_sorting(population) -> list:
    '''
    
    Procedure of separating points from the general population into non-dominated levels.
    Operates in the straightforward way: each layer is comprised of elements, that are 
    not dominated by any other element in the population, except the ones, already put into
    the output levels. Computational complexity of this variant of sorting in worst scenario is
    :math:`O(MN^3)`, where *N* is the population size, and *M* is the number of objective functions.
    
    Parameters:
    -----------
    
    population : list
        The input population, represented as a list of individuals.
        
    Returns: 
    ----------
    
    levels : list
        List of lists of population elements. The outer index is the number of a layer 
        (e.g. 0-th is the current Pareto frontier), while the inner is the index of an element on a level.
    
    '''
    locked_idxs = []
    levels = []; levels_elems = 0
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
        locked_idxs.extend(processed_idxs); levels_elems += len(processed_idxs)
        levels.append([population[elem_idx] for elem_idx in processed_idxs])
    return levels
 
    
def acute_angle(vector_a, vector_b) -> float:
    return np.arccos(np.dot(vector_a, vector_b)/(np.sqrt(np.dot(vector_a, vector_a))*np.sqrt(np.dot(vector_b, vector_b))))


class Constraint(ABC):
    '''
    
    The abstract class for the constraint. Noteable subclasses: Inequality & Equality.
    
    '''
    @abstractmethod
    def __init__(self, *args):
        pass
    
    @abstractmethod
    def __call__(self, *args):
        pass
    
    
class Inequality(Constraint):
    '''
        
    The class of the constrain (subclass of Constraint), representing the inequality. 
    The format of inequality is assumed in format :math:`g(x) >= 0`.
    
    Parameters:
    ----------
    
    g : function (lambda function)
        The constraint function, which shall penalize the candidate solution, if the value of
        :math:`g(x) >= 0` is not fulfilled (is less, than 0). The penalty is equal to 
        the absolute value of constraint violation.
    
    Methods:
    ---------
    
    __call__(self, x) : returns float
        Overloaded call operator returns the value of constaint violation.
    
    '''    
    def __init__(self, g):
        self._g = g
        
    def __call__(self, x) -> float:
        '''
        Method to evaluate the constraint violation of the candidate solution.
        
        Parameters:
        ---------
        
        x : np.ndarray
            Values (.vals attribute) of the candidate solution, that represent its gene.
        
        Returns:
        --------
        
        cv : float
            Constraint violation value. If the value of :math:`g(x) >= 0` is not 
            fulfilled (is less, than 0), than returns :math:`|g(x)|`, else 0.
        
        '''
        return - self._g(x) if self._g(x) < 0 else 0
    

class Equality(Constraint):
    '''
        
    The class of the constrain (subclass of Constraint), representing the inequality. 
    The format of inequality is assumed in format :math:`h(x) = 0`.
    
    Parameters:
    ----------
    
    h : function (lambda function)
        The constraint function, which shall be penalized, if the value does not match with
        the const
    
    Methods:
    ---------
    
    __call__(self, x) : returns float
        Overloaded call operator returns the value of constaint violation.
        
    '''  
    def __init__(self, h):
        self._h = h
        
    def __call__(self, x) -> float:
        '''
        Method to evaluate the constraint violation of the candidate solution.
        
        Parameters:
        ---------
        
        x : np.ndarray
            Values (.vals attribute) of the candidate solution, that represent its gene.
        
        Returns:
        --------
        
        cv : float
            Constraint violation value. If the value of :math:`h(x) = 0` is not 
            fulfilled, than returns :math:`|g(x)|`, else 0.
        
        '''        
        return np.abs(self._h(x))    