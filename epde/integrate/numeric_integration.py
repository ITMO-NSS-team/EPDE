from typing import Union, List, Dict, Tuple, Callable
import copy
from functools import reduce
from warnings import warn
import os

from ctypes import PyDLL, py_object

import torch
import numpy as np
from scipy.integrate import ode, solve_ivp

from epde.solver.data import Domain, Conditions
from epde.structure.main_structures import SoEq
from epde.integrate.bop import PregenBOperator, BOPElement, get_max_deriv_orders
from epde.integrate.interface import SystemSolverInterface

def get_terms_der_order(equation: Dict, variable_idx: int) -> np.ndarray:
    """
    Get the highest derivative orders of a specified variable within each term of the equation.
        This information is crucial for determining the complexity and structure of the equation,
        which guides the search for the optimal equation form.
    
        Args:
            equation (Dict): A dictionary representing the equation, where keys are term indices and values are dictionaries containing term information (variables, derivatives, powers).
            variable_idx (int): The index of the variable for which to find the highest derivative orders.
    
        Returns:
            np.ndarray: A NumPy array containing the highest derivative order of the specified variable in each term of the equation.
    """
    term_max_ord = np.zeros(len(equation))
    for term_idx, term_dict in enumerate(equation.values()):

        if isinstance(term_dict['var'], list) and len(term_dict['var']) > 1:
            max_ord = 0
            for arg_idx, deriv_ord in enumerate(term_dict['term']):
                if isinstance(term_dict['pow'][arg_idx], (int, float)) and term_dict['var'][arg_idx] == variable_idx:
                    max_ord = max(max_ord, len([var for var in deriv_ord if var is not None]))
            term_max_ord[term_idx] = max_ord
        elif isinstance(term_dict['var'], int):
            if isinstance(term_dict['pow'], (int, float)) and term_dict['var'] == variable_idx:
                term_max_ord[term_idx] = max(0, len([var for var in term_dict['term'] if var is not None]))
        elif isinstance(term_dict['var'], list) and len(term_dict['var']) == 1:
            if isinstance(term_dict['var'][0], (int, float)):
                term_var = term_dict['var'][0]
            elif isinstance(term_dict['var'][0], (list, tuple)):
                term_var = term_dict['var'][0][0]
            if (isinstance(term_dict['pow'], (int, float)) or (isinstance(term_dict['pow'], (list, tuple)) 
                                                               and len(term_dict['pow']) == 1)) and term_var == variable_idx:
                term_max_ord[term_idx] = max(0, len([var for var in term_dict['term'] if var is not None and var != [None,]]))
        pass

    return term_max_ord

def get_higher_order_coeff(equation: Dict, orders: np.ndarray, var: int) -> Tuple[List]:
    """
    Extracts coefficients corresponding to the highest derivative order with respect to a specified variable from the equation.
    
    This function identifies terms in the equation associated with the highest derivative order and isolates them as 'denominator terms'. The remaining terms are considered 'numerator terms'. The 'denominator terms' are then transformed by effectively nullifying the influence of the specified variable, preparing them for further analysis or simplification of the equation.
    
    Args:
        equation: A dictionary representing the equation, where keys are term indices and values are dictionaries containing term information about each term.
        orders: A NumPy array containing the derivative order of each term in the equation, corresponding to the term indices in the equation dictionary.
        var: The variable index with respect to which the higher-order coefficients are extracted.
    
    Returns:
        A list containing two lists: the first list contains the transformed 'denominator terms' (higher-order coefficients), and the second list contains the remaining 'numerator terms'.
    """
    def transform_term(term: Dict, deriv_key: list, var: int) -> Dict:
        term_filtered = copy.deepcopy(term)
        if (isinstance(term['var'], int) and term['var'] == var) or (isinstance(term['var'], list) 
                                                                     and len(term['var']) == 1 and term['var'][0] == var):
            term_filtered['term'] = [None,]
            term_filtered['pow'] = 0
        else:
            term_idx = [idx for idx, der_var in enumerate(term_filtered['term'])
                        if der_var == deriv_key and term_filtered['var'][idx] == var][0]
            term_filtered['term'][term_idx] = [None,]
            term_filtered['pow'][term_idx] = 0
        return term_filtered            

    denom_terms = []
    numer_terms = []
    for term_idx, term in enumerate(equation.values()):
        if orders[term_idx] == np.max(orders):
            denom_terms.append(transform_term(term, deriv_key=[0,]*int(np.max(orders)), var=var))
        else:
            numer_terms.append(term)
    return [denom_terms, numer_terms]

def get_eq_order(equation, variables: List[str]):
    """
    Identifies the variable with the highest derivative order in a given equation.
    
    This function is crucial for determining the complexity and structure of the
    differential equation. By identifying the variable with the highest derivative,
    we can understand which variable's dynamics are most influential in the equation.
    This information is essential for equation discovery and simplification.
    
    Args:
      equation: The equation to analyze, represented as a symbolic expression or string.
      variables: A list of variable names present in the equation.
    
    Returns:
      A tuple containing:
        - The index of the variable with the highest derivative order in the `variables` list.
        - A NumPy array representing the derivative orders of each term in the equation
          with respect to the identified variable.
    """
    eq_var = 0; eq_orders = np.zeros(len(equation))
    for var_idx in range(len(variables)):
        orders = get_terms_der_order(equation=equation, variable_idx=var_idx)
        if np.max(orders) > np.max(eq_orders):
            eq_var = var_idx; eq_orders = orders
    return eq_var, eq_orders

def replace_operator(term: Dict, variables: List):
    """
    Replaces symbolic variable representations within a term with numerical indices based on a provided variable list.
    
        This transformation is crucial for efficiently representing and manipulating equations within the system,
        allowing for streamlined processing during equation discovery and optimization. By converting symbolic
        variables into numerical indices, the system can leverage optimized numerical operations for tasks such as
        equation evaluation and comparison.
    
        Args:
            term (Dict): A dictionary representing a term in the equation, containing information about the variable, its power, and any derivatives.
            variables (List): A list of tuples, where each tuple represents a variable and its differentiation order, used to map symbolic variables to numerical indices.
    
        Returns:
            Dict: A modified term dictionary with symbolic variables replaced by their corresponding numerical indices from the `variables` list.
    """
    term_ = copy.deepcopy(term)
    if isinstance(term_['var'], list) and len(term_['var']) > 1:
        for arg_idx, deriv_ord in enumerate(term_['term']):
            if isinstance(term_['var'][arg_idx], (tuple, list)):
                continue
            term_['var'][arg_idx]  = variables.index((term_['var'][arg_idx], deriv_ord))
            term_['term'][arg_idx] = [None,]
    elif isinstance(term['var'], int) or (isinstance(term_['var'], list) and len(term_['var']) == 1):
        if isinstance(term['var'], int):
            term_var = term_['var']
        else:
            term_var = term_['var'][0]
        if isinstance(term['pow'], (int, float)):
            term_['var']  = variables.index((term_var, term_['term']))
            term_['term'] = [None,]
    return term_

class ImplicitEquation(object):
    """
    Represents an implicit equation system for solving differential equations.
    
            This class takes a system of equations, a grid, and a list of variables
            to set up the problem for numerical solution. It parses the equations,
            determines the order of each variable, and constructs dynamics operators.
    
            Class Methods:
            - __init__
            - parse_cond
            - __call__
            - grid_dict
            - create_first_ord_eq
            - merge_coeff
            - term_callable
    
            Attributes:
                grid_dict: A NumPy array representing the grid.
                _dynamics_operators: A list of dynamics operators, each corresponding to a variable.
                _var_order: A list of tuples, where each tuple contains a variable and its derivative order.
                _vars_with_eqs: A dictionary mapping an index to a tuple containing a variable and its order within the system of equations.
    """

    def __init__(self, system: List, grid: np.ndarray, variables: List[str]):
        """
        Initializes the `ImplicitEquation` with a system of equations, a grid, and a list of variables.
        
                This constructor prepares the system for analysis by determining the order of each variable within the equations and constructing corresponding dynamics operators. These operators are essential for representing the relationships between variables and their derivatives, enabling the system to be solved or analyzed.
        
                Args:
                    system: A list of equations representing the system.
                    grid: A NumPy array representing the grid.
                    variables: A list of variable names.
        
                Fields:
                    grid_dict: A NumPy array representing the grid.
                    _dynamics_operators: A list of dynamics operators, each corresponding to a variable.
                    _var_order: A list of tuples, where each tuple contains a variable and its derivative order.
                    _vars_with_eqs: A dictionary mapping an index to a tuple containing a variable and its order within the system of equations.
        
                Returns:
                    None
        """
        self.grid_dict = grid

        # print(f'Solved system is {system}')

        self._dynamics_operators = []
        self._var_order = []
        self._vars_with_eqs = {}

        for var, order in [get_eq_order(equation, variables) for equation in system]:
            self._var_order.extend([(var, [None,])] + [(var, [0,]*(idx+1)) for idx in range(int(np.max(order))-1)])
            if len(self._vars_with_eqs) == 0:
                self._vars_with_eqs[int(np.max(order)) - 1] = (var, order)
            else:
                self._vars_with_eqs[list(self._vars_with_eqs.keys())[-1] + int(np.max(order))] = (var, order)

        for var_idx, var in enumerate(self._var_order):
            if var_idx in self._vars_with_eqs.keys():
                operator = get_higher_order_coeff(equation = system[self._vars_with_eqs[var_idx][0]],
                                                  orders = self._vars_with_eqs[var_idx][1], 
                                                  var = self._vars_with_eqs[var_idx][0])
                operator[0] = [replace_operator(denom_term, self._var_order) for denom_term in operator[0]]
                operator[1] = [replace_operator(numer_term, self._var_order) for numer_term in operator[1]]
            else:
                operator = [None, self.create_first_ord_eq(var_idx + 1)]
            self._dynamics_operators.append(operator)

    def parse_cond(self, conditions: List[Union[BOPElement, dict]]):
        """
        Parses boundary conditions to create a condition vector aligned with dynamic operators.
        
                This method processes a list of boundary conditions, which can be specified
                as `BOPElement` objects or dictionaries. It maps these conditions to a
                numerical vector, ensuring each dynamic operator has a corresponding
                boundary value. This is crucial for setting up the problem before solving.
        
                Args:
                    conditions: A list of boundary conditions. Each condition can be
                        either a `BOPElement` or a dictionary. `BOPElement` should contain
                        information about variable index and term, dictionary should contain
                        'bnd_op' and 'bnd_val' keys.
        
                Returns:
                    A NumPy array representing the condition values for each dynamic operator.
                    The array has the same length as the list of dynamic operators.
        """
        cond_val = np.full(shape = len(self._dynamics_operators), fill_value=np.inf)
        for cond in conditions:
            if isinstance(cond, BOPElement):
                assert isinstance(cond.variables, int), 'Boundary operator has to contain only a single variable.'
                try:
                    var = self._var_order.index((cond.variables, cond.term))
                except ValueError:
                    print(f'Missing {cond.variables, cond.term} from the list of variables {self._var_order}')
                    raise RuntimeError()
                cond_val[var] = cond.values
            else:
                op_form = list(cond['bnd_op'].values())[0]
                term_key = [op_key for op_key in list(op_form.keys()) if op_key not in ['coeff', 'pow', 'var']][0]
                try:
                    # print(f'term key {term_key}')
                    var = self._var_order.index((op_form['var'], op_form[term_key]))
                except ValueError:
                    print(f'Missing {(op_form["var"], op_form[term_key])} from the list of variables {self._var_order}')
                    raise RuntimeError()
                cond_val[var] = cond['bnd_val']

        assert np.sum(np.inf == cond_val) == 0, 'Not enough initial conditions were passed.'
        return cond_val

    def __call__(self, t, y):
        """
        Evaluates the dynamics operators for a given state and time.
        
                This method calculates the values of the dynamics operators based on the
                provided state `y` and time `t`. It iterates through each operator,
                calculates the numerator and denominator using the `term_callable` method,
                and computes the resulting value. A `ZeroDivisionError` is raised if any
                denominator is close to zero. This evaluation is crucial for assessing
                the equation's behavior at specific points in the search space, guiding
                the evolutionary algorithm towards better solutions.
        
                Args:
                    t (float): The time value.
                    y (np.ndarray): The state vector.
        
                Returns:
                    np.ndarray: An array containing the evaluated values of the dynamics
                        operators.
        """
        values = np.empty(len(self._dynamics_operators))
        for idx, operator in enumerate(self._dynamics_operators):
            if operator[0] is None:
                denom = 1
            else:
                denom = [self.term_callable(term, t, y) for term in operator[0]]
                if np.isclose(denom, 0):
                    raise ZeroDivisionError('Denominator in the dynamics operator is close to zero.')
            numerator = [self.term_callable(term, t, y) for term in operator[1]]
            values[idx] = -1*np.sum(numerator)/np.sum(denom)
        return values

    @property
    def grid_dict(self):
        """
        Returns a dictionary containing the grid values, rounded for discrete representation. This is useful for tasks such as visualization or when the equation discovery process benefits from discrete representations of the data domain.
        
                Args:
                    None
        
                Returns:
                    dict: A dictionary where keys represent grid points and values are the corresponding rounded values.
        """
        return self._grid_rounded

    @grid_dict.setter
    def grid_dict(self, grid_points):
        """
        Generates a dictionary mapping rounded grid points to their indices.
        
                This method calculates the grid step size, rounds the grid points to
                a precision determined by the grid step, and creates a dictionary
                where the keys are the rounded grid point values and the values are
                their corresponding indices in the original grid points array. This
                dictionary is used to efficiently find the grid index closest to a
                given value during the equation discovery process.
        
                Args:
                    grid_points: A NumPy array of grid point values.
        
                Returns:
                    dict: A dictionary where keys are rounded grid point values and
                        values are their indices.
        
                Initializes:
                    _grid_step (float): The step size between grid points, calculated
                        as the difference between the second and first grid points.
                    _grid_rounded (dict): A dictionary mapping rounded grid point values
                        to their indices in the original `grid_points` array.
        """
        self._grid_step = grid_points[1] - grid_points[0]
        digits = np.floor(np.log10(self._grid_step/2.)-1)
        self._grid_rounded = {np.round(grid_val, -int(digits)): idx 
                              for idx, grid_val in np.ndenumerate(grid_points)}

    def create_first_ord_eq(self, var: int) -> List[Tuple]:
        """
        Creates a first-order equation term for a specified variable.
        
                This is a fundamental building block for constructing more complex differential equations.
                It ensures that each term adheres to the defined structure, which is essential for the evolutionary process.
        
                Args:
                    var (int): The index of the variable to create the first-order term for.
        
                Returns:
                    List[Tuple]: A list containing a dictionary representing the first-order equation term.
                                 The dictionary includes the coefficient, term, power, and variable index.
        """
        return [{'coeff' : -1.,
                 'term'  : [None,],
                 'pow'   : 1,
                 'var'   : var},]

    def merge_coeff(self, coeff: np.ndarray, t: float):
        """
        Merges coefficients based on time, leveraging the temporal grid.
        
                This method retrieves a coefficient value for a given time `t`. It first attempts to
                directly access the coefficient from the internal grid dictionary. If the time `t`
                is not explicitly present in the grid, it interpolates the coefficient value
                between the nearest neighboring grid points. This ensures a continuous representation
                of the coefficient over time, even when data is only available at discrete points.
                This is important to correctly represent coefficient values at any given point in time
                during the equation discovery process.
        
                Args:
                    coeff (np.ndarray): Coefficient array representing the coefficient values at grid points.
                    t (float): Time value for which to retrieve/interpolate the coefficient.
        
                Returns:
                    float: The coefficient value at time `t`, either retrieved directly from the grid
                           or interpolated between neighboring grid points.
        """
        try:
            return self.grid_dict[t]
        except KeyError:
            for grid_loc, grid_idx in self.grid_dict.items():
                if grid_loc < t and grid_loc + self._grid_step > t:
                    # print('Search in ', grid_loc, grid_loc + self._grid_step)
                    left_loc, right_loc = grid_loc, grid_loc + self._grid_step
                    left_idx, right_idx = grid_idx[0], grid_idx[0] + 1
                    break
            val = coeff[left_idx] + (t - left_loc) / (right_loc - left_loc) * (coeff[right_idx] - coeff[left_idx])
            return val

    def term_callable(self, term: Dict, t, y):
        """
        Evaluates a single term in the equation based on its definition.
        
                This method calculates the value of a term in the equation, considering
                its coefficient, variables, and powers. It handles different types of
                coefficients (callables, neural networks, arrays, or constants) and
                variables (indices or tuples of indices). It also supports callable or
                neural network powers. This flexibility is crucial for representing a wide range of equation structures that the evolutionary algorithm might discover.
        
                Args:
                    term: A dictionary defining the term, containing keys 'coeff', 'var', and 'pow'.
                    t: The current time value.
                    y: The current state vector.
        
                Returns:
                    float: The calculated value of the term.
        """
        def call_ann_token(token_nn: torch.nn.Sequential, arguments: list,
                           t: float, y: np.ndarray):
            return token_nn[torch.from_numpy(y[tuple(arguments)]).reshape((-1, 1))].detach().numpy() # Hereby, the ANN does not explicitly depend on time

        if isinstance(term['coeff'], Callable):
            k = term['coeff'](t)
        elif isinstance(term['coeff'], torch.nn.Sequential):
            k = term['coeff'](torch.from_numpy(t).reshape((1, 1).float()))
        elif isinstance(term['coeff'], np.ndarray):
            k = self.merge_coeff(term['coeff'], t)
        else:
            k = term['coeff']
        
        if not isinstance(term['var'], (list, tuple)) or len(term['pow']) == 1:
            term_var = [term['var'],]
        else:
            term_var = term['var']
        if isinstance(term['var'], (list, tuple)):
            term_pow = term['pow']
        else:
            term_pow = [term['pow'],]
        
        values = []
        for var_idx, var in enumerate(term_var):  
            if isinstance(var, int):
                if isinstance(term_pow[var_idx], (int, float)):
                    val = y[var]**term_pow[var_idx]
                elif isinstance(term_pow[var_idx], torch.nn.Sequential):
                    val = call_ann_token(term_pow[var_idx], var, t, y)
                else:                
                    val = term_pow[var_idx](y[var])
            elif isinstance(var, (tuple, list)):
                if isinstance(term_pow[var_idx], torch.nn.Sequential):
                    val = call_ann_token(term_pow[var_idx], var, t, y)
                elif isinstance(term_pow[var_idx], (int, float)):
                    assert len(var) == 1, 'Incorrect number of arguments'
                    val = y[list(var)]**term_pow[var_idx]
                    if isinstance(val, np.ndarray): val = val[0]
                    
                    # print(values[-1], type(values[-1]))
                else:               
                    val = term_pow[var_idx](*y[list(var)])
                    if isinstance(val, torch.Tensor):
                        val = val.item()

            values.append(val)
            pass

        return reduce(lambda x, z: x*z, values, k)


class OdeintAdapter(object):
    """
    Adapts the odeint solver for use with EPDE framework.
    
        This class provides an interface to solve systems of ordinary differential
        equations using the odeint solver. It handles the conversion of EPDE
        systems into a format suitable for odeint and manages the solution process.
    
        Class Methods:
        - __init__
        - solve_system
        - build_ode_problem
        - solve
    """

    def __init__(self, method: str = 'Radau'):
        """
        Initializes the solver adapter with a specified numerical integration method.
        
                This constructor configures the adapter to use a particular method for solving the discovered differential equations.
                The choice of the integration method can significantly impact the accuracy and efficiency of the equation solving process,
                allowing the framework to explore a wider range of equation candidates and refine the discovered models.
        
                Args:
                    method (str): The numerical integration method to be used (default: 'Radau').
        
                Returns:
                    None
        
                Class Fields:
                    _solve_method (str): The solving method to be used. Initialized with the value of the `method` parameter.
        """
        self._solve_method = method
        pass # TODO: implement hyperparameters setup, according to the problem specifics

    def solve_epde_system(self, system: Union[SoEq, dict], grids: list=None, boundary_conditions=None,
                          mode='NN', data=None, vars_to_describe = ['u'], *args, **kwargs):
        """
        Solves a system of equations, preparing it for the numerical solver.
                
                This method takes a system of equations, either in a structured format
                or as a list of solver forms, and prepares it for solution. It sets up
                the necessary boundary conditions, either using provided conditions or
                generating default ones based on the input data and grids. This ensures
                the system is properly configured before being passed to the solver.
        
                Args:
                    system (Union[SoEq, dict]): The system of equations to solve. Can be a SoEq object or a list of solver forms.
                    grids (list, optional): The spatial and temporal grids on which to solve the system. Defaults to None.
                    boundary_conditions (optional): Boundary conditions for the system. If None, default conditions are generated. Defaults to None.
                    mode (str, optional): The mode for adapting the system. Defaults to 'NN'.
                    data (optional): Data used for generating default boundary conditions. Defaults to None.
                    vars_to_describe (list, optional): List of variables to describe in the solution. Defaults to ['u'].
                    *args: Additional positional arguments.
                    **kwargs: Additional keyword arguments.
        
                Returns:
                    The solution to the EPDE system.
        
                WHY: This method prepares the system of equations for the solver by handling different input formats,
                setting up boundary conditions, and extracting relevant variables. This ensures that the solver receives
                a properly configured system, leading to accurate and reliable solutions.
        """
        if isinstance(system, SoEq):
            system_interface = SystemSolverInterface(system_to_adapt=system)
            system_solver_forms = system_interface.form(grids = grids, mode = mode)
        elif isinstance(system, list):
            system_solver_forms = system
        else:
            raise TypeError('Incorrect input into the Odeint Adapter.')

        if boundary_conditions is None:
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system_solver_forms])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions

        if isinstance(system, SoEq):
            vars_to_describe = system.vars_to_describe
            
        return self.solve(equations = [sf_labeled[1] for sf_labeled in system_solver_forms], domain = grids[0], 
                          boundary_conditions = boundary_conditions, vars = vars_to_describe)
        # Add condition parser and control function args parser



    def solve(self, equations, domain: Union[Domain, np.ndarray],
              boundary_conditions: List[BOPElement] = None, vars: List[str] = ['x',], *args, **kwargs):
        """
        Solves the implicit equation defined by the given equations, domain, and boundary conditions using numerical integration.
        
                This method leverages `scipy.integrate.solve_ivp` to find a numerical solution to the system of ordinary differential equations defined by `equations`. The solution is computed over the specified `domain`, starting from the provided `boundary_conditions`. The method returns the solution trajectory evaluated at points within the domain. This is a crucial step in the equation discovery process, as it allows us to evaluate how well a candidate equation fits the observed data by comparing the numerical solution to the data.
        
                Args:
                    equations: A list of equations that define the implicit equation.
                    domain: The domain over which to solve the equation. It can be a
                        `Domain` object or a NumPy array.
                    boundary_conditions: A list of boundary conditions for the equation.
                    vars: A list of variable names used in the equations.
        
                Returns:
                    A tuple containing:
                      - An integer status code (0 for success).
                      - A NumPy array representing the solution of the equation.
        """
        if not isinstance(equations, list):
            raise RuntimeError('Equations have to be passed as a list.')
        self._implicit_equation = ImplicitEquation(equations, domain, vars)
        if isinstance(domain, Domain): 
            grid = domain.build().detach().numpy().reshape(-1)
        else:
            grid = domain.detach().numpy().reshape(-1)

        initial_cond = self._implicit_equation.parse_cond(boundary_conditions)
        solution = solve_ivp(fun = self._implicit_equation, t_span = (grid[0], grid[-1]), y0=initial_cond,
                             t_eval = grid, method = self._solve_method)
        if not solution.success:
            warn(f'Numerical solution of ODEs has did not converge. The error message is {solution.message}')
        return 0, solution.y.T

def loadSpectralSolver(path_to_so: str):
    """
    Loads a spectral solver library from a shared object file.
    
        This method loads a spectral solver library (a .so file) using `ctypes.PyDLL`.
        It configures the argument and return types for the solver function within the library
        to ensure compatibility with the evolutionary algorithm's data structures.
        This is necessary for evaluating candidate equations by numerically solving them
        and comparing the solutions to the observed data.
    
        Args:
            path_to_so: The path to the shared object file (.so) containing the spectral solver.
    
        Returns:
            The loaded solver library as a `ctypes.PyDLL` object, or `None` if loading fails.
    """
    try:
        # TODO:
        # Using PyDLL (particularly .so file), that is inported via ctypes is a kludge 
        # and shall be refactored ASAP.
        solverlib = PyDLL(path_to_so)
        solverlib.argtypes = [py_object, py_object, py_object, py_object]
        solverlib.restype  =  py_object
        return solverlib
    except:
        return None

class SpectralAdapter(object):
    """
    Adapts a spectral solver to a unified interface.
    
        This class serves as an adapter for spectral solvers, providing a consistent
        interface for solving equations using different spectral methods. It handles
        the setup, execution, and result retrieval from the underlying solver.
    
        Class Methods:
        - __init__
        - set_parameters
        - solve
        - get_result
        - get_named_result
        - get_raw_result
        - is_finished
        - has_errors
        - get_error_message
        - get_solver_statistics
        - get_solution
        - get_named_solution
        - get_raw_solution
        - get_internal_data
        - set_internal_data
    """

    def __init__(self, path_to_so: str = None, **kwargs):
        """
        Initializes the SpectralSolverWrapper, preparing the solver for equation discovery.
        
                This involves loading the spectral solver library and configuring it
                for subsequent operations such as basis function evaluation and
                spectral coefficient calculation, which are crucial steps in
                transforming data into a spectral representation suitable for
                equation learning.
        
                Args:
                    path_to_so: Path to the spectral solver shared object file.
                    **kwargs: Arbitrary keyword arguments to be stored as parameters.
        
                Returns:
                    None
        
                Class Fields:
                    params (dict): A dictionary storing arbitrary keyword arguments passed during initialization.
                    spectral_solver: The loaded spectral solver object.
        """
        self.params = kwargs

        if path_to_so is None:
            path_to_so = os.path.abspath(os.getcwd())
        
        self.spectral_solver = loadSpectralSolver(path_to_so)
        if self.spectral_solver is None:
            raise RuntimeError("Failed to load spectral solver.")
        
    def solve_epde_system(self, system: Union[SoEq, dict], grids: list=None, boundary_conditions=None,
                          mode='NN', data=None, vars_to_describe = ['u'], *args, **kwargs):
        """
        Solves an EPDE system using the specified solver to find the best equation representation.
                
                This method takes a system of equations and transforms it into a format suitable
                for the solver. It then uses the solver to find the solution that best fits
                the data, effectively identifying the underlying differential equation. Default
                boundary conditions are generated if none are provided. This is done to ensure
                the solver has all the necessary information to find a valid solution.
                
                Args:
                    system: The system of equations to solve. Can be a SoEq object or a list
                        of solver forms.
                    grids: The spatial and temporal grids on which to solve the system.
                    boundary_conditions: The boundary conditions for the system. If None,
                        default boundary conditions are generated.
                    mode (str): The mode of operation for the solver. Defaults to 'NN'.
                    data: The data to use for solving the system. Defaults to None.
                    vars_to_describe (list): A list of variables to describe. Defaults to ['u'].
                    *args: Additional positional arguments.
                    **kwargs: Additional keyword arguments.
                
                Returns:
                    The solution to the EPDE system.
        """
        if isinstance(system, SoEq):
            system_interface = SystemSolverInterface(system_to_adapt=system)
            system_solver_forms = system_interface.form(grids = grids, mode = mode)
        elif isinstance(system, list):
            system_solver_forms = system
        else:
            raise TypeError('Incorrect input into the Odeint Adapter.')

        if boundary_conditions is None:
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system_solver_forms])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions

        if isinstance(system, SoEq):
            vars_to_describe = system.vars_to_describe
            
        return self.solve(equations = [sf_labeled[1] for sf_labeled in system_solver_forms], domain = grids[0], 
                          boundary_conditions = boundary_conditions, vars = vars_to_describe)
    
    def solve(self, equations, domain: Union[Domain, np.ndarray],
              boundary_conditions: List[BOPElement] = None, vars: List[str] = ['x',], *args, **kwargs):
        """
        Solves a system of equations using a spectral solver.
        
                This method leverages a spectral solver to find solutions that satisfy the given equations within the specified domain and boundary conditions.
                It prepares the domain by converting it into a suitable grid format for the solver.
                The method returns the solution obtained from the spectral solver.
        
                Args:
                    equations: The system of equations to solve.
                    domain: The domain over which to solve the equations. Can be a
                        Domain object or a numpy array.
                    boundary_conditions: A list of boundary conditions to apply. Defaults to None.
                    vars: A list of variable names. Defaults to ['x'].
        
                Returns:
                    tuple: A tuple containing:
                        - An integer status code (0 for success).
                        - The solution obtained from the spectral solver.
        
                Why: This method automates the process of finding solutions for differential equations,
                which is a crucial step in discovering governing equations from data.
                By using a spectral solver, it efficiently finds solutions that fit the observed data within the given constraints.
        """
        if not isinstance(equations, list):
            raise RuntimeError('Incorrect type of equations passed into odeint solver.')

        if isinstance(domain, Domain): 
            grid = domain.build().detach().numpy().reshape(-1) # todo: verify torch tensor application
        else:
            grid = domain.detach().numpy().reshape(-1) # todo: verify torch tensor application

        solution = self.spectral_solver(equations, grid, boundary_conditions, self.params)

        return 0, solution 