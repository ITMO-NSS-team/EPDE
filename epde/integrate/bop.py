from typing import Union, List, Dict
from types import FunctionType
from functools import singledispatch

import numpy as np
import torch

import epde.globals as global_var
from epde.structure.main_structures import SoEq

VAL_TYPES = Union[FunctionType, int, float, torch.Tensor, np.ndarray]


def get_max_deriv_orders(system_sf: List[Union[Dict[str, Dict]]], variables: List[str] = ['u',]) -> dict:
    """
    Computes the maximum derivative orders for each variable in a system of equations.
    
        This method analyzes a system of symbolic forms representing equations to determine the highest derivative order present for each variable with respect to each axis. This information is crucial for constructing suitable numerical schemes and discretizations when solving or analyzing the identified differential equations.
    
        Args:
            system_sf: A list of symbolic forms, where each element represents an equation.
                Each equation can be a dictionary or a list.
            variables: A list of variable names to analyze. Defaults to ['u'].
    
        Returns:
            dict: A dictionary where keys are variable names and values are NumPy arrays
                representing the maximum derivative orders for each axis.
                For example, {'u': array([2., 0.])} indicates that the maximum
                derivative order of variable 'u' is 2 along the first axis and 0
                along the second axis.
    """
    def count_factor_order(factor_code, deriv_ax):
        if factor_code is None or isinstance(factor_code, tuple):
            return 0
        else:
            if isinstance(factor_code, list):
                return factor_code.count(deriv_ax)
            elif isinstance(factor_code, int):
                return 1 if factor_code == deriv_ax else 0
            else:
                raise TypeError(f'Incorrect type of the input. Got {type(factor_code), factor_code}, expecting int or list')

    @singledispatch
    def get_equation_requirements(equation_sf, variables=['u',]):
        raise NotImplementedError(
            'Single-dispatch called in generalized form')

    @get_equation_requirements.register
    def _(equation_sf: dict, variables=['u',]) -> dict:  # dict = {u: 0}):
        dim = global_var.grid_cache.get('0').ndim
        if len(variables) == 1:
            var_max_orders = np.zeros(dim)
            for term in equation_sf.values():
                if isinstance(term['pow'], list):
                    for deriv_factor in term['term']:
                        orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                            in np.arange(dim)])
                        var_max_orders = np.maximum(var_max_orders, orders)
                else:
                    orders = np.array([count_factor_order(term['term'], ax) for ax
                                        in np.arange(dim)])
                    var_max_orders = np.maximum(var_max_orders, orders)
            return {variables[0]: var_max_orders}
        else:
            var_max_orders = {var_key: np.zeros(dim) for var_key in variables}
            for term_key, symb_form in equation_sf.items():
                if isinstance(symb_form['var'], list):
                    assert len(symb_form['term']) == len(symb_form['var'])
                    for factor_idx, deriv_factor in enumerate(symb_form['term']):
                        var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                in np.arange(dim)])
                        if isinstance(symb_form['var'][factor_idx], int):
                            var_key = symb_form['var'][factor_idx] #- 1
                        else:
                            var_key = 0
                            var_orders = 0 # Such tokens do not increase order of the DE
                        var_max_orders[variables[var_key]] = np.maximum(var_max_orders[variables[var_key]],
                                                                        var_orders)
                elif isinstance(symb_form['var'], int):
                    raise NotImplementedError()
                    assert len(symb_form['term']) == 1
                    for factor_idx, factor in enumerate([count_factor_order(symb_form['term'], ax) for ax
                                                        in np.arange(dim)]):
                        var_orders = np.array([count_factor_order(deriv_factor, ax) for ax
                                                in np.arange(dim)])
                        var_key = symb_form['var'][factor_idx]
                        var_max_orders[var_key] = np.maximum(var_max_orders[var_key], var_orders)
            return var_max_orders

    @get_equation_requirements.register
    def _(equation_sf: list, variables=['u',]):
        raise NotImplementedError(
            'TODO: add equation list form processing') 

    eq_forms = []
    for equation_form in system_sf:
        eq_forms.append(get_equation_requirements(equation_form, variables))

    max_orders = {var: np.maximum.accumulate([eq_list[var] for eq_list in eq_forms])[-1]
                    for var in variables}  # TODO
    return max_orders

class BOPElement(object):
    """
    Represents a boundary operator element.
    
        This class defines a boundary operator element, encapsulating its properties
        and methods for application and manipulation within a boundary value problem.
    
        Methods:
        - __init__
        - set_grid
        - operator_form
        - values
        - __call__
    """

    def __init__(self, axis: int, key: str, coeff: float = 1., term: list = [None], 
                 power: Union[Union[List[int], int]] = 1, var: Union[List[int], int] = 1,
                 rel_location: float = 0., device = 'cpu'):
        """
        Initializes a Constraint object for equation discovery.
        
        This method sets up a constraint within the equation search space,
        guiding the evolutionary algorithm towards solutions that satisfy
        specific conditions. These constraints help to narrow down the search
        and ensure that the discovered equations adhere to known physical laws
        or domain-specific knowledge.
        
        Args:
            axis: The axis along which the constraint is applied.
            key: A string identifier for the constraint.
            coeff: The coefficient of the constraint term (default: 1.0).
            term: A list representing the terms in the constraint (default: [None]).
            power: The power of the constraint term (default: 1). It can be either a list of integers or a single integer.
            var: The variable associated with the constraint (default: 1). It can be either a list of integers or a single integer.
            rel_location: The relative location of the constraint (default: 0.0).
            device: The device to use for computation (default: 'cpu').
        
        Fields:
            axis: The axis along which the constraint is applied (int).
            key: A string identifier for the constraint (str).
            coefficient: The coefficient of the constraint term (float).
            term: A list representing the terms in the constraint (list).
            power: The power of the constraint term (Union[Union[List[int], int]]).
            variables: The variable associated with the constraint (Union[List[int], int]).
            location: The relative location of the constraint (float).
            grid: Placeholder for grid information, initialized to None.
            status: A dictionary tracking the status of boundary condition setup,
                initialized with 'boundary_location_set' and 'boundary_values_set'
                both set to False.
            _device: The device to use for computation (str).
        
        Returns:
            None.
        
        Why:
            Constraints are essential for guiding the equation discovery process.
            They allow incorporating prior knowledge and physical laws, leading
            to more accurate and physically meaningful discovered equations.
        """
        self.axis = axis
        self.key = key
        self.coefficient = coeff
        self.term = term
        self.power = power
        self.variables = var
        self.location = rel_location
        self.grid = None
        
        self.status = {'boundary_location_set': False,
                       'boundary_values_set': False}
        
        self._device = device

    def set_grid(self, grid: torch.Tensor):
        """
        Sets the spatial grid for the element.
        
        This method updates the internal grid representation, which defines the
        spatial domain over which the element is defined. Setting the grid also
        marks the boundary location as set, indicating that the element's
        spatial context is fully defined. This is a crucial step for subsequent
        operations that rely on spatial information.
        
        Args:
            grid (torch.Tensor): The new grid to be used.
        
        Returns:
            None.
        
        Class Fields:
            grid (torch.Tensor): The grid representing the spatial domain.
            status (dict): A dictionary storing the status of various flags,
                including 'boundary_location_set'.
        """
        self.grid = grid
        self.status['boundary_location_set'] = True

    @property
    def operator_form(self):
        """
        Represents the term in a structured format suitable for equation discovery.
        
        This representation facilitates the application of evolutionary algorithms
        by organizing the term's components (coefficient, term itself, power, and variables)
        into a dictionary. This structured format is essential for evaluating and
        manipulating equation candidates during the search process.
        
        Returns:
            tuple: A tuple containing the key and a dictionary representing the operator form.
                The dictionary includes the coefficient ('coeff'), term, power ('pow'),
                and variables ('var') of this element.
        
        Args:
            None
        
        Returns:
            tuple: A tuple where the first element is the key of the term and the second
                element is a dictionary containing the 'coeff', term, 'pow', and 'var'
                representing the operator form of the term.
        """
        form = {
            'coeff': self.coefficient,
            self.key: self.term,
            'pow': self.power,
            'var': self.variables
        }
        return self.key, form

    @property
    def values(self):
        """
        Returns the coefficient values, evaluating them on the grid if necessary.
        
                This property provides a way to access the coefficient values, ensuring they are
                evaluated on the spatial grid if the coefficient is defined as a function. This
                dynamic evaluation is crucial for representing spatially varying coefficients
                within the discovered differential equations.
        
                Args:
                    self: The object instance.
        
                Returns:
                    torch.Tensor: The value of the coefficient, as a PyTorch tensor. If the coefficient
                        is a function, the tensor represents the function evaluated on the grid.
                        Otherwise, it returns the pre-computed value.
        """
        if isinstance(self._values, FunctionType):
            assert self.grid_set, 'Tring to evaluate variable coefficent without a proper grid.'
            res = self._values(self.grids)
            assert res.shape == self.grids[0].shape
            return torch.from_numpy(res).to(self._device)
        else:
            return self._values

    @values.setter
    def values(self, vals):
        """
        Sets the coefficient values for the basis function. These values determine the contribution of this basis function to the overall equation.
        
                The input is processed to ensure compatibility with the computational backend. Specifically, if the input is a NumPy array, it's converted to a PyTorch tensor and moved to the appropriate device for efficient computation. This ensures that all coefficients are in a consistent format for equation evaluation and optimization.
        
                Args:
                    vals (Union[Callable, int, float, torch.Tensor, np.ndarray]): The values to set for the coefficients. Can be a function, integer, float, PyTorch tensor, or NumPy array.
        
                Returns:
                    None
        
                Raises:
                    TypeError: If the input `vals` is not of a supported type.
        
                Class Fields Initialized:
                    _values (torch.Tensor or function): The values of the coefficients, stored as a PyTorch tensor or a function.
                    vals_set (bool): A flag indicating whether the values have been set.
        """
        if isinstance(vals, (FunctionType, int, float, torch.Tensor)):
            self._values = vals
            self.vals_set = True
        elif isinstance(vals, np.ndarray):
            self._values = torch.from_numpy(vals).to(self._device)
            self.vals_set = True
        else:
            raise TypeError(
                f'Incorrect type of coefficients. Must be a type from list {VAL_TYPES}.')

    def __call__(self, values: VAL_TYPES = None) -> dict:
        """
        Applies the boundary operator to construct a boundary condition for the problem.
        
                This method prepares the boundary condition by either using provided values or
                setting them internally. It determines the boundary's spatial configuration
                based on the grid or location information, then formulates the boundary
                operator and value. This is crucial for defining the constraints of the
                differential equation being solved.
        
                Args:
                    values: The boundary values. If None and boundary values have not been
                        previously set, a ValueError is raised.
        
                Returns:
                    dict: A dictionary containing the boundary location ('bnd_loc'),
                        boundary operator ('bnd_op'), boundary value ('bnd_val'),
                        variables ('variables'), and type ('type').
        """
        if not self.vals_set and values is not None:
            self.values = values
            self.status['boundary_values_set'] = True
        elif not self.vals_set and values is None:
            raise ValueError('No location passed into the BOP.')
        if self.grid is not None:
            boundary = self.grid
        elif self.grid is None and self.location is not None:
            _, all_grids = global_var.grid_cache.get_all(mode = 'torch')

            abs_loc = self.location * all_grids[0].shape[self.axis]
            if all_grids[0].ndim > 1:
                boundary = np.array(all_grids[:self.axis] + all_grids[self.axis+1:])
                if isinstance(values, FunctionType):
                    raise NotImplementedError  # TODO: evaluation of BCs passed as functions or lambdas
                boundary = torch.from_numpy(np.expand_dims(boundary, axis=self.axis)).to(self._device).float()

                boundary = torch.cartesian_prod(boundary,
                                                torch.from_numpy(np.array([abs_loc,], dtype=np.float64)).to(self._device)).float()
                boundary = torch.moveaxis(boundary, source=0, destination=self.axis).resize()
            else:
                boundary = torch.from_numpy(np.array([[abs_loc,],])).to(self._device).float() # TODO: work from here
            
        elif boundary is None and self.location is None:
            raise ValueError('No location passed into the BOP.')
            
        form = self.operator_form
        boundary_operator = {form[0]: form[1]}
        
        boundary_value = self.values
        
        return {'bnd_loc' : boundary.to(self._device), 'bnd_op' : boundary_operator, 
                'bnd_val' : boundary_value.to(self._device), 
                'variables' : self.variables, 'type' : 'operator'}

class PregenBOperator(object):
    """
    Represents a boundary operator (BOperator) that pre-generates boundary conditions.
    
             Class Methods:
             - demonstrate_required_ords
             - conditions
             - max_deriv_orders
             - generate_default_bc
    """

    def __init__(self, system: SoEq, system_of_equation_solver_form: list): #, device = 'cpu'
        self.system = system
        self.equation_sf = [eq for eq in system_of_equation_solver_form]
        self.variables = list(system.vars_to_describe)

    def demonstrate_required_ords(self):
        """
        Demonstrates how to specify the maximum derivative orders for each equation in the system.
        
                This method prepares a list associating each main variable with its corresponding maximum derivative order. This information is crucial for the evolutionary process to explore suitable equation structures by limiting the order of derivatives considered for each variable, thereby guiding the search for governing differential equations.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    list: A list of tuples, where each tuple contains a main variable and its maximum derivative order.
        """
        linked_ords = list(zip([eq.main_var_to_explain for eq in self.system],
                                self.max_deriv_orders))

    @property
    def conditions(self):
        """
        Returns the boolean conditions used to define the search space.
        
                These conditions constrain the possible forms of the equation, 
                guiding the search towards more plausible or physically meaningful solutions.
        
                Returns:
                    list: The list of boolean conditions.
        """
        return self._bconds

    @conditions.setter
    def conditions(self, conds: List[BOPElement]):
        """
        Initializes the boundary conditions required for solving the discovered differential equation.
        
                This method sets up the boundary conditions necessary for numerical solvers to accurately compute solutions. It ensures that the provided boundary conditions are of the correct type and number, matching the order of derivatives present in the discovered equation.
        
                Args:
                    conds: A list of BOPElement objects, each representing a boundary condition. The order of these conditions must align with the derivative orders of the equation.
        
                Returns:
                    None
        
                Raises:
                    ValueError: If the number of provided boundary conditions does not match the expected number based on the equation's derivative orders.
                    NotImplementedError: If an attempt is made to initialize a boundary operator in-place, as this functionality is not yet implemented.
        
                Class Fields Initialized:
                    _bconds (list): A list of initialized boundary condition objects, ready for use by the numerical solver.
        """
        self._bconds = []
        if len(conds) != int(sum([value.sum() for value in self.max_deriv_orders.values()])):
            raise ValueError(
                'Number of passed boundry conditions does not match requirements of the system.')
        for condition in conds:
            if isinstance(condition, BOPElement):
                self._bconds.append(condition())
            else:
                print('condition is ', type(condition), condition)
                raise NotImplementedError(
                    'In-place initialization of boundary operator has not been implemented yet.')

    @property
    def max_deriv_orders(self):
        """
        Gets the maximum derivative orders for each variable in the equation.
        
                This property determines the highest derivative order present for each variable within the symbolic equation, which is crucial for understanding the complexity and structure of the identified differential equation. Knowing the maximum derivative orders helps in selecting appropriate numerical solvers and interpreting the equation's behavior.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    dict: A dictionary where keys are the variables and values are their
                        maximum derivative orders.
        """
        return get_max_deriv_orders(self.equation_sf, self.variables)

    def generate_default_bc(self, vals: Union[np.ndarray, dict] = None, grids: List[np.ndarray] = None,
                            allow_high_ords: bool = False, required_bc_ord: List[int] = None):
        """
        Generates default boundary conditions based on the provided parameters.
        
                This method constructs boundary condition operators (BOPElement) to enforce constraints on the solution at the domain boundaries.
                It determines the location and derivative order of these conditions based on the provided parameters and the underlying grid.
                This ensures that the discovered differential equations adhere to the physical constraints of the problem.
        
                Args:
                    vals: Values for the boundary conditions. If None, default values are used.
                    grids: List of grid tensors. If None, grids are retrieved from the cache.
                    allow_high_ords: A boolean flag to allow higher order derivatives. (Not fully implemented)
                    required_bc_ord: A dictionary specifying the required derivative orders for each variable.
                        If None, the `max_deriv_orders` attribute is used.
        
                Returns:
                    list: A list of BOPElement objects representing the generated boundary conditions.
        
                Initializes:
                    conditions (list): A list of BOPElement objects representing the generated boundary conditions.
        """
        # Implement allow_high_ords - selection of derivatives from
        if required_bc_ord is None:
            required_bc_ord = self.max_deriv_orders
        assert set(self.variables) == set(required_bc_ord.keys()), 'Some conditions miss required orders.'

        grid_cache = global_var.initial_data_cache
        tensor_cache = global_var.initial_data_cache

        if vals is None:
            val_keys = {key: (key, (1.0,)) for key in self.variables}

        if grids is None:
            _, grids = grid_cache.get_all(mode = 'torch')

        device = global_var.grid_cache._device
        # assert self._device
        device_on_cpu = (device  == 'cpu')
        relative_bc_location = {0: (), 1: (0,), 2: (0, 1),
                                3: (0., 0.5, 1.), 4: (0., 1/3., 2/3., 1.)}

        bconds = []
        tensor_shape = grids[0].shape

        def get_boundary_ind(tensor_shape, axis, rel_loc):
            return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                                       for dim_idx, shape in enumerate(tensor_shape)], indexing='ij'))

        for var_idx, variable in enumerate(self.variables):
            for ax_idx, ax_ord in enumerate(required_bc_ord[variable]):
                for loc in relative_bc_location[ax_ord]:
                    indexes = get_boundary_ind(tensor_shape, ax_idx, rel_loc=loc)

                    if device_on_cpu:
                        coords = np.array([grids[idx][indexes].detach().numpy() for idx in np.arange(len(tensor_shape))]).T
                    else:
                        coords = np.array([grids[idx][indexes].detach().cpu().numpy()
                                           for idx in np.arange(len(tensor_shape))]).T
                    if coords.ndim > 2:
                        coords = coords.squeeze()

                    if vals is None:
                        bc_values = tensor_cache.get(val_keys[variable])[indexes]
                    else:
                        bc_values = vals[indexes]

                    bc_values = np.expand_dims(bc_values, axis=0).T
                    coords = torch.from_numpy(coords).to(device).float()

                    bc_values = torch.from_numpy(bc_values).to(device).float() # TODO: set devices for all torch objs
                    operator = BOPElement(axis=ax_idx, key=variable, coeff=1, term=[None],
                                          power=1, var=var_idx, rel_location=loc, device=device)
                    operator.set_grid(grid=coords)
                    operator.values = bc_values
                    bconds.append(operator)
        print('Types of conds:', [type(cond) for cond in bconds])
        self.conditions = bconds


class BoundaryConditions(object):
    """
    Represents boundary conditions for a physical simulation.
    
        Attributes:
            dirichlet (dict): Dirichlet boundary conditions.
            neumann (dict): Neumann boundary conditions.
            robin (dict): Robin boundary conditions.
    """

    def __init__(self, grids=None, partial_operators: dict = []):
        """
        Initializes the OperatorCollection object.
        
        The OperatorCollection stores spatial information and precomputed operators
        required for efficient boundary condition application. This precomputation
        is done once and used many times during the PDE approximation.
        
        Args:
            grids: The grids to store.
            partial_operators (dict): A dictionary of partial operators.
        
        Fields:
            grids_set (bool): A boolean indicating whether grids are set.
            grids: The grids stored, if provided.
            operators (dict): The partial operators stored.
        
        Returns:
            None.
        """
        self.grids_set = (grids is not None)
        if grids is not None:
            self.grids = grids
        self.operators = partial_operators

    def form_operator(self):
        """
        Forms a list of boundary conditions for each operator.
        
                This method retrieves boundary conditions associated with each operator
                defined in the system. These conditions are essential for solving the discovered
                differential equations numerically.
        
                Args:
                    None
        
                Returns:
                    list[list]: A list of lists, where each inner list represents the boundary
                    conditions for a specific operator. The operators and their associated
                    boundary conditions are obtained from the `self.operators` dictionary.
        """
        return [list(bcond()) for bcond in self.operators.values()]
