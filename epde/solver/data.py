"""module for working with inerface for initialize grid, conditions and equation"""

from typing import List, Union
import torch
import numpy as np
import sys
import os

from epde.solver.device import check_device
from epde.solver.input_preprocessing import EquationMixin


def tensor_dtype(dtype: str):
    """
    Converts a string representation of a data type to its corresponding PyTorch data type.
    
    This function ensures that the data type used within the equation discovery process is correctly interpreted by the PyTorch backend, 
    allowing for seamless integration with tensor operations and numerical solvers.
    
    Args:
        dtype (str): A string representing the desired data type (e.g., 'float32', 'float64', 'float16').
    
    Returns:
        torch.dtype: The corresponding PyTorch data type.
    """
    if dtype == 'float32':
        dtype = torch.float32
    elif dtype == 'float64':
        dtype = torch.float64
    elif dtype == 'float16':
        dtype = torch.float16

    return dtype


class Domain():
    """
    class for grid building
    """

    def __init__(self, type='uniform'):
        """
        Initializes a new Domain instance.
        
        This method sets the domain type and prepares a dictionary to hold symbolic variables.
        The domain type influences how variables are sampled and used during the equation discovery process.
        The variable dictionary will store symbolic representations of variables used in the discovered equations.
        
        Args:
            type (str): The type of the domain, influencing variable sampling (default: 'uniform').
        
        Returns:
            None.
        """
        self.type = type
        self.variable_dict = {}
    
    @property
    def dim(self):
        """
        Returns the dimensionality of the domain.
        
                This property returns the number of variables associated with the domain,
                effectively representing the dimensionality of the search space for
                equation discovery. This is crucial for understanding the complexity
                of potential differential equations that can be constructed within this domain.
        
                Args:
                    self: The Domain instance.
        
                Returns:
                    int: The number of variables in the domain's variable dictionary.
        """
        return len(self.variable_dict)
    
    def variable(
            self,
            variable_name: str,
            variable_set: Union[List, torch.Tensor],
            n_points: Union[None, int],
            dtype: str = 'float32') -> None:
        """
        Initializes a spatial variable within the domain.
        
        This method creates a tensor representing the spatial discretization of a given variable.
        It supports both uniform sampling between specified bounds and the use of pre-defined tensor values.
        This is a crucial step in setting up the computational domain for solving differential equations.
        
        Args:
            variable_name (str): Name of the spatial variable.
            variable_set (Union[List, torch.Tensor]): Either a list [start, stop] defining the interval for uniform discretization,
                                                    or a torch.Tensor containing the pre-defined variable values.
            n_points (Union[None, int]): Number of points to use for uniform discretization. Required if `variable_set` is a list.
            dtype (str, optional): Data type of the resulting tensor. Defaults to 'float32'.
        
        Returns:
            None: The method updates the internal `variable_dict` with the created tensor.
        """
        dtype = tensor_dtype(dtype)

        if isinstance(variable_set, torch.Tensor):
            variable_tensor = check_device(variable_set)
            variable_tensor = variable_set.to(dtype)
            self.variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                n_points = n_points + 1
                start, end = variable_set
                variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
                self.variable_dict[variable_name] = variable_tensor
    
    def build(self, mode: str) -> torch.Tensor:
        """
        Generates a computational grid based on the specified mode.
        
        This grid serves as the foundation for representing the problem domain,
        enabling the application of various numerical techniques to solve
        differential equations. The grid is constructed from the domain's
        variables, and its structure depends on the chosen solution mode.
        
        Args:
            mode (str): Specifies the solution approach ('mat', 'autograd', or 'NN').
                         Determines how the grid is generated and used in subsequent calculations.
        
        Returns:
            torch.Tensor: A tensor representing the computational grid. Its shape and
                          values are determined by the domain variables and the selected mode.
        """
        var_lst = list(self.variable_dict.values())
        var_lst = [i.cpu() for i in var_lst]
        if mode in ('autograd', 'NN'):
            if len(self.variable_dict) == 1:
                grid = check_device(var_lst[0].reshape(-1, 1)) # TODO: verify the correctness of mat method grids generation
            else:
                grid = check_device(torch.cartesian_prod(*var_lst))
        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = check_device(torch.tensor(np.array(grid)))
        return grid


class Conditions():
    """
    class for adding the conditions: initial, boundary, and data.
    """

    def __init__(self):
        """
        Initializes a new instance of the Conditions class.
        
        The Conditions class manages a list of symbolic conditions that are used to filter and select equation candidates during the equation discovery process. This method initializes the list to store these conditions.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            conditions_lst (list): A list to store conditions.
        """
        self.conditions_lst = []

    def dirichlet(
            self,
            bnd: Union[torch.Tensor, dict],
            value: Union[callable, torch.Tensor, float],
            var: int = 0):
        """
        Applies a Dirichlet boundary condition to the problem.
        
        This method specifies fixed values for the solution at the given boundary points.
        It's used to constrain the solution space and ensure that the identified equation
        satisfies the known boundary behavior.
        
        Args:
            bnd (Union[torch.Tensor, dict]): Boundary points where the Dirichlet condition is applied.
                Can be a torch.Tensor or a dictionary with coordinate names as keys and their values as values.
            value (Union[callable, torch.Tensor, float]): The value(s) of the solution at the boundary points.
                If a callable is provided, it will be evaluated at the boundary points to determine the values.
            var (int, optional): The variable index for systems of equations. Defaults to 0, representing a single equation.
        
        Returns:
            None: The method updates the internal list of boundary conditions (`self.conditions_lst`) with the specified Dirichlet condition.
        """

        self.conditions_lst.append({'bnd': bnd,
                                    'bop': None,
                                    'bval': value,
                                    'var': var,
                                    'type': 'dirichlet'})

    def operator(self,
                 bnd: Union[torch.Tensor, dict],
                 operator: dict,
                 value: Union[callable, torch.Tensor, float]):
        """
        Adds an operator boundary condition to the list of conditions. This condition constrains the solution based on a differential operator applied at the boundary.
        
                Args:
                    bnd (Union[torch.Tensor, dict]): Boundary points where the condition is applied. Can be a tensor or a dictionary with coordinate names as keys and coordinate values as values.
                    operator (dict): A dictionary defining the differential operator. It specifies the terms, coefficients, powers, and variables involved in the operator.
                    value (Union[callable, torch.Tensor, float]): The value of the operator at the boundary. Can be a constant, a tensor, or a callable function that takes the boundary points as input.
        
                Returns:
                    None: This method adds the boundary condition to an internal list (`self.conditions_lst`) for later use in the equation discovery process.
        
                Why:
                    This method is crucial for incorporating boundary conditions into the equation discovery process. By specifying constraints on the solution at the boundaries, we can guide the search towards equations that not only fit the data but also satisfy known physical or mathematical constraints. This helps to refine the search space and improve the accuracy and reliability of the discovered equations.
        """
        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0
        operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'operator'})

    def periodic(self,
                 bnd: Union[List[torch.Tensor], List[dict]],
                 operator: dict = None,
                 var: int = 0):
        """
        Adds a periodic boundary condition to the problem definition. This ensures that the solution exhibits a repeating pattern across the specified boundaries. This is useful when modeling systems where the behavior at one boundary is directly related to the behavior at another, such as in wave propagation or fluid dynamics.
        
                Args:
                    bnd (Union[List[torch.Tensor], List[dict]]): A list defining the boundaries for the periodic condition.  Each element can be a tensor representing a coordinate or a dictionary specifying coordinate values (e.g., `{'x': 1, 't': [0,1]}`).
                    operator (dict, optional): A dictionary defining the operator for the periodic condition. This is used when enforcing periodicity on derivatives or other operators. Defaults to None, which implies a Dirichlet-type periodic condition (i.e., the function values are periodic).
                    var (int, optional): The index of the variable to which the periodic condition applies. This is relevant for systems of equations. Defaults to 0.
        
                Returns:
                    None: This method modifies the internal list of conditions (`self.conditions_lst`) by appending a dictionary representing the periodic boundary condition.
        """
        value = torch.tensor([0.])
        if operator is None:
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})
        else:
            try:
                var = operator[operator.keys()[0]]['var']
            except:
                var = 0
            operator = EquationMixin.equation_unify(operator)
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})

    def data(
        self,
        bnd: Union[torch.Tensor, dict],
        operator: Union[dict, None],
        value: torch.Tensor,
        var: int = 0):
        """
        Adds data conditions to the list of conditions. These conditions represent known values of the solution at specific locations, potentially influenced by differential operators. This information is crucial for guiding the equation discovery process by providing constraints that candidate equations must satisfy.
        
                Args:
                    bnd (Union[torch.Tensor, dict]): Boundary points where the solution's value is known. Can be a tensor or a dictionary mapping coordinate names to values.
                    operator (Union[dict, None]): Differential operator(s) associated with the data condition. Specifies how the solution's derivatives relate to the known value. Can be None if the value is directly known.
                    value (torch.Tensor): The known value of the solution (or the result of the operator acting on the solution) at the boundary points.
                    var (int, optional): Index of the variable if dealing with a system of equations. Defaults to 0.
        
                Returns:
                    None
        """
        if operator is not None:
            operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'data'})

    def _bnd_grid(self,
                  bnd: Union[torch.Tensor, dict],
                  variable_dict: dict,
                  dtype) -> torch.Tensor:
        """
        Builds a subgrid representing the boundary conditions for the problem.
        
                This method constructs a grid of points that satisfy the specified boundary conditions.
                It handles different types of boundary specifications, including direct tensor inputs,
                fixed coordinate values, and interval constraints on coordinate values. The resulting
                grid is used to evaluate the solution at the boundaries and enforce the boundary conditions
                during the training process.
        
                Args:
                    bnd (Union[torch.Tensor, dict]): Boundary conditions. Can be a torch.Tensor
                        representing the boundary points directly, or a dictionary where keys are
                        coordinate names and values are either:
                        - torch.Tensor: Tensor of boundary points for that coordinate.
                        - float or int: Fixed value for that coordinate.
                        - list: Interval [lower_bound, upper_bound] specifying the range for that coordinate.
                    variable_dict (dict): Dictionary containing torch.Tensors for each domain variable,
                        representing the grid coordinates.
                    dtype (dtype): The desired data type for the resulting grid.
        
                Returns:
                    torch.Tensor: A tensor representing the subgrid of points that satisfy the specified
                        boundary conditions. The shape of the tensor is (N, D), where N is the number of
                        boundary points and D is the number of dimensions.
        """

        dtype = variable_dict[list(variable_dict.keys())[0]].dtype

        if isinstance(bnd, torch.Tensor):
            bnd_grid = bnd.to(dtype)
        else:
            var_lst = []
            for var in variable_dict.keys():
                if isinstance(bnd[var], torch.Tensor):
                    var_lst.append(check_device(bnd[var]).to(dtype))
                elif isinstance(bnd[var], (float, int)):
                    var_lst.append(check_device(torch.tensor([bnd[var]])).to(dtype))
                elif isinstance(bnd[var], list):
                    lower_bnd = bnd[var][0]
                    upper_bnd = bnd[var][1]
                    grid_var = variable_dict[var]
                    bnd_var = grid_var[(grid_var >= lower_bnd) & (grid_var <= upper_bnd)]
                    var_lst.append(check_device(bnd_var).to(dtype))
            bnd_grid = torch.cartesian_prod(*var_lst).to(dtype)
        if len(bnd_grid.shape) == 1:
            bnd_grid = bnd_grid.reshape(-1, 1)
        return bnd_grid

    def build(self,
              variable_dict: dict) -> List[dict]:
        """
        Processes boundary conditions to prepare them for equation discovery.
        
        This method prepares boundary conditions by converting them into a usable format,
        ensuring compatibility with the data and specified device. It handles different
        types of boundary conditions, including periodic ones, and ensures that boundary
        values are correctly formatted as tensors with the appropriate data type and device.
        This preprocessing step is crucial for the subsequent equation discovery process.
        
        Args:
            variable_dict (dict): A dictionary containing torch.Tensors representing the domain variables.
        
        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains preprocessed information about a boundary condition.
        """
        if self.conditions_lst == []:
            return None

        try:
            dtype = variable_dict[list(variable_dict.keys())[0]].dtype
        except:
            dtype = variable_dict[list(variable_dict.keys())[0]][0].dtype # if periodic

        for cond in self.conditions_lst:
            if cond['type'] == 'periodic':
                cond_lst = []
                for bnd in cond['bnd']:
                    cond_lst.append(self._bnd_grid(bnd, variable_dict, dtype))
                cond['bnd'] = cond_lst
            else:
                cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype)
            
            if isinstance(cond['bval'], torch.Tensor):
                cond['bval'] = check_device(cond['bval']).to(dtype)
            elif isinstance(cond['bval'], (float, int)):
                cond['bval'] = check_device(
                    torch.ones_like(cond['bnd'][:,0])*cond['bval']).to(dtype)
            elif callable(cond['bval']):
                cond['bval'] = check_device(cond['bval'](cond['bnd'])).to(dtype)

        return self.conditions_lst


class Equation():
    """
    class for adding eqution.
    """

    def __init__(self):
        """
        Initializes the EquationList object.
        
        The EquationList is used to store and manage a collection of equation objects.
        This initialization creates an empty list ready to be populated with equations discovered or created during the equation discovery process.
        
        Args:
            self: The object instance.
        
        Returns:
            None
        """
        self.equation_lst = []
    
    @property
    def num(self):
        """
        Returns the number of equations.
        
        This property provides a convenient way to access the size of the equation list,
        reflecting the complexity of the discovered equation system.
        
        Args:
            self: The Equation object instance.
        
        Returns:
            int: The number of equations currently stored.
        """
        return len(self.equation_lst)
    
    def add(self, eq: dict):
        """
        Adds a new equation to the system of equations.
        
        This allows the system to represent more complex relationships and dependencies between variables.
        
        Args:
            eq (dict): A dictionary representing the equation in operator form.
        
        Returns:
            None
        """
        self.equation_lst.append(eq)
