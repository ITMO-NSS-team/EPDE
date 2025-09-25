"""Module for operatoins with operator and boundaru con-ns."""

from typing import Tuple, Union, List
import torch

from epde.solver.points_type import Points_type
from epde.solver.derivative import Derivative
from epde.solver.device import device_type, check_device
from epde.solver.utils import PadTransform

from torch.utils.data import DataLoader

def integration(func: torch.Tensor,
                grid: torch.Tensor,
                power: int = 2) \
                -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """
    Integrates a function over a grid, effectively reducing the dimensionality of the problem.
    
    This function performs numerical integration of a given function `func` over a specified `grid`.
    The integration is performed along one axis of the grid at a time.
    If the grid has more than one dimension, the integration result and the reduced grid are returned.
    If the grid has only one dimension, the final integration result and 0 are returned.
    
    Args:
        func (torch.Tensor): The function values at each grid point.  Represents the operator applied to the test function.
        grid (torch.Tensor): The grid points over which to integrate.
        power (int, optional): The power to which the function values are raised during integration. Defaults to 2.
    
    Returns:
        Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
            - If the grid has only one column, returns a tuple containing the final integration result (float) and 0.0 (float).
            - If the grid has multiple columns, returns a tuple containing:
                - A list of integration results (list) for each segment along the integrated axis.
                - The reduced grid (torch.Tensor) with the last column removed.
    
    Why:
        This function is used to reduce the dimensionality of the problem by integrating out one spatial dimension at a time.
        This is a key step in solving differential equations using numerical methods.
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    u = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            u += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** power + func[i - 1] ** power) / 2
        else:
            result.append(u)
            marker = grid[i][column]
            index.append(i)
            u = 0.
    if column == -1:
        return u, 0.
    else:
        result.append(u)
        grid = grid[index, :-1]
        return result, grid


def dict_to_matrix(bval: dict, true_bval: dict)\
    -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """
    Transforms dictionaries of boundary values into matrix representations for equation discovery.
    
    This function converts dictionaries containing predicted and true boundary values
    for different boundary types into PyTorch tensors. These tensors are structured
    as matrices, where each column represents a specific boundary type. The function
    also returns lists of boundary types and their corresponding lengths.
    
    Args:
        bval (dict): Dictionary where keys are boundary types and values are predicted
                     boundary values (torch.Tensor).
        true_bval (dict): Dictionary where keys are boundary types and values are true
                          boundary values (torch.Tensor).
    
    Returns:
        matrix_bval (torch.Tensor): Matrix with predicted boundary values; each column
                                    corresponds to a boundary type.
        matrix_true_bval (torch.Tensor): Matrix with true boundary values; each column
                                         corresponds to a boundary type.
        keys (list): List of boundary types corresponding to the columns of the matrices.
        len_list (list): List of lengths for each boundary type column before padding.
    
    Why:
        This transformation is necessary to prepare the boundary data for efficient
        processing and comparison within the equation discovery pipeline. By converting
        the dictionaries into matrix form, the framework can leverage vectorized
        operations for calculating loss functions and evaluating the fitness of
        candidate equations based on their ability to satisfy the given boundary conditions.
    """

    keys = list(bval.keys())
    max_len = max([len(i) for i in bval.values()])
    pad = PadTransform(max_len, 0)
    matrix_bval = pad(bval[keys[0]]).reshape(-1,1)
    matrix_true_bval = pad(true_bval[keys[0]]).reshape(-1,1)
    len_list = [len(bval[keys[0]])]
    for key in keys[1:]:
        bval_i = pad(bval[key]).reshape(-1,1)
        true_bval_i = pad(true_bval[key]).reshape(-1,1)
        matrix_bval = torch.hstack((matrix_bval, bval_i))
        matrix_true_bval = torch.hstack((matrix_true_bval, true_bval_i))
        len_list.append(len(bval[key]))

    return matrix_bval, matrix_true_bval, keys, len_list


class Operator():
    """
    Class for differential equation calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_operator: Union[list,dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int,
                 batch_size: int = None):
        """
        Initializes the Operator class, preparing it for the application of differential operators to a given model on a specified grid.
        
                The Operator class facilitates the evaluation of differential operators,
                handling different computational modes and data batching for efficient processing.
        
                Args:
                    grid (torch.Tensor): The spatial or temporal grid representing the domain of the problem.
                    prepared_operator (Union[list,dict]): The differential operator, preprocessed into a suitable format for computation.
                    model (Union[torch.nn.Sequential, torch.Tensor]): The model (e.g., neural network) that approximates the solution of the differential equation.
                    mode (str): Specifies the computation mode ('mat', 'NN', or 'autograd') for operator evaluation.
                    weak_form (List[callable]): A list of basis functions used when solving the equation in a weak formulation.
                    derivative_points (int): The number of points used in the numerical approximation of derivatives.
                    batch_size (int, optional): The size of mini-batches used for processing the grid. Defaults to None.
        
                Returns:
                    None
        """
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.derivative_points = derivative_points
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode in ('autograd', 'mat'):
            self.sorted_grid = self.grid
        self.batch_size = batch_size
        if self.batch_size is not None:
            self.grid_loader =  DataLoader(self.sorted_grid, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=device_type()))
            self.n_batches = len(self.grid_loader)
            del self.sorted_grid
            torch.cuda.empty_cache()
            self.init_mini_batches()
            self.current_batch_i = 0
        self.derivative = Derivative(self.model,
                                self.derivative_points).set_strategy(self.mode).take_derivative
    
    def init_mini_batches(self):
        """
        Initializes the mini-batch iterator for processing data in smaller chunks. This prepares the data loader to provide the next batch of grid points for equation discovery.
        
                Args:
                    self: The Operator instance.
        
                Returns:
                    None
        """
        self.grid_iter = iter(self.grid_loader)
        self.grid_batch = next(self.grid_iter)

    def apply_operator(self,
                       operator: list,
                       grid_points: Union[torch.Tensor, None]) -> torch.Tensor:
        """
        Applies a given operator to a grid subset, effectively evaluating a part of the equation on that subset.
        
        This process is crucial for breaking down the equation into manageable parts that can be computed and later assembled to represent the entire equation's behavior across the domain.
        
        Args:
            operator (list): A prepared list of terms representing the operator. This list is structured according to the equation's definition. See input_preprocessing.operator_prepare() for details on the expected format.
            grid_points (Union[torch.Tensor, None]): The points on the grid subset where the operator will be evaluated. This is only used in 'autograd' and 'mat' modes, where numerical derivatives are calculated.
        
        Returns:
            torch.Tensor: The result of applying the operator to the grid subset. This represents the decoded operator's contribution on that specific subset.
        """

        for term in operator:
            term = operator[term]
            dif = self.derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def _pde_compute(self) -> torch.Tensor:
        """
        Computes the residual of the partial differential equation (PDE).
        
        This method calculates the PDE residual by applying the prepared operator to the sorted grid.
        The grid can be processed in mini-batches if a batch size is specified.
        The method handles cases with single or multiple equations.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The computed PDE residual.
        """

        if self.batch_size is not None:
            sorted_grid = self.grid_batch
            try:
                self.grid_batch = next(self.grid_iter)
            except: # if no batches left then reinit
                self.init_mini_batches()
                self.current_batch_i = -1
        else:
            sorted_grid = self.sorted_grid
        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(
                self.prepared_operator[0], sorted_grid).reshape(-1,1)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], sorted_grid).reshape(-1,1))
            op = torch.cat(op_list, 1)
        return op


    def _weak_pde_compute(self) -> torch.Tensor:
        """
        Computes the weak form of the PDE residual by integrating the product of the PDE operator and test functions over the domain. This process effectively transforms the strong form of the PDE into a weak form, enabling the computation of a solution that minimizes the residual in a Galerkin sense. The method iterates through each component of the PDE operator, applies the weak form functions, and performs integration to obtain the residual.
        
                Args:
                    None
        
                Returns:
                    torch.Tensor: weak PDE residual.
        """

        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self._pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in self.weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for _ in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list).reshape(1,-1)

    def operator_compute(self):
        """
        Calculates the residual of the operator, choosing between a direct computation or a weak form computation based on the operator's configuration.
        
        This selection allows the framework to adapt the residual calculation based on the chosen representation of the differential equation.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The operator residual.
        """
        if self.weak_form is None or self.weak_form == []:
            return self._pde_compute()
        else:
            return self._weak_pde_compute()


class Bounds():
    """
    Class for boundary and initial conditions calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_bconds: Union[list, dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: List[callable],
                 derivative_points: int):
        """
        Initializes the Bounds object, preparing the boundary conditions for solving the differential equation.
        
                This involves setting up the computational grid, processing boundary conditions,
                and configuring the model (neural network, matrix, or autograd) to satisfy these conditions.
                The operator is initialized to handle the evaluation of the differential equation
                on the grid, considering the specified mode (e.g., matrix-based, neural network-based, or autograd-based)
                and weak form (if applicable). This setup ensures that the solution respects the constraints
                imposed by the boundary conditions.
        
                Args:
                    grid (torch.Tensor): The computational grid representing the domain discretization.
                    prepared_bconds (Union[list, dict]): Prepared boundary conditions, processed by the Equation class.
                    model (Union[torch.nn.Sequential, torch.Tensor]): The model (neural network, matrix, or autograd) used to represent the solution.
                    mode (str): Specifies the mode of operation ('mat', 'NN', or 'autograd').
                    weak_form (List[callable]): A list of basis functions for the weak formulation of the equation.
                    derivative_points (int): The number of points used for derivative calculation (relevant for the Derivative_mat class).
        
                Returns:
                    None
        """
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = Operator(self.grid, self.prepared_bconds,
                                       self.model, self.mode, weak_form,
                                       derivative_points)

    def _apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """
        Calculates the contribution of a set of boundary operators to the solution field.
        
        This method iterates through a list of pre-processed boundary operators, applies each operator to the grid,
        and concatenates the results to form a complete representation of the boundary conditions' influence on the field.
        This is a crucial step in incorporating boundary information into the solution obtained by the neural network.
        
        Args:
            operator_set (list): A list of boundary operators, pre-processed to be compatible with the neural network.
                                 These operators define the mathematical constraints imposed by the boundaries.
                                 See Equation_NN.operator_prepare for details on the preparation process.
        
        Returns:
            torch.Tensor: A tensor representing the aggregated effect of all boundary operators on the solution field.
                          This tensor is used to enforce the specified boundary conditions during the training or inference.
        """

        field_part = []
        for operator in operator_set:
            field_part.append(self.operator.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def _apply_dirichlet(self, bnd: torch.Tensor, var: int) -> torch.Tensor:
        """
        Applies Dirichlet boundary conditions to the model output at specified boundary points.
        
        This method extracts the model's prediction at the boundary locations and reshapes it
        to ensure compatibility with subsequent calculations, effectively enforcing the Dirichlet
        boundary condition. The specific implementation depends on the chosen mode of operation
        ('NN', 'autograd', or 'mat'), allowing for flexibility in how the model's output is accessed
        and processed.
        
        Args:
            bnd (torch.Tensor): Terms (boundary points) of prepared boundary conditions.
                For more details, refer to the input preprocessing (bnd_prepare method).
            var (int): Indicates for which dependent variable it is necessary to apply
                the boundary condition. For a single equation, this is typically 0.
        
        Returns:
            torch.Tensor: The model's output at the boundary points, reshaped for use in
                loss calculations or other downstream operations.
        """

        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                b_op_val.append(self.model[var][position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def _apply_neumann(self, bnd: torch.Tensor, bop: list) -> torch.Tensor:
        """
        Computes the value of the boundary condition based on the specified mode.
        
        This method calculates the boundary condition value by applying the derivative operator
        to the boundary points, using different approaches depending on the configured mode.
        The mode determines whether a neural network, automatic differentiation, or a matrix-based
        method is used to evaluate the operator. This is a crucial step in enforcing boundary
        constraints within the solution domain.
        
        Args:
            bnd (torch.Tensor): Terms (boundary points) of prepared boundary conditions.
            bop (list): Terms of prepared boundary derivative operator.
        
        Returns:
            torch.Tensor: Calculated boundary condition.
        """

        if self.mode == 'NN':
            b_op_val = self._apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.operator.apply_operator(bop, bnd)
        elif self.mode == 'mat':
            var = bop[list(bop.keys())[0]]['var'][0]
            b_op_val = self.operator.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                b_val.append(b_op_val[var][position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def _apply_periodic(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """
        Applies periodic boundary conditions by combining Dirichlet or Neumann conditions based on the specified mode. This ensures that the solution at one boundary matches the solution at the opposite boundary, which is crucial for modeling systems with repeating spatial or temporal domains.
        
                Args:
                    bnd (torch.Tensor): Terms (boundary points) of prepared boundary conditions.
                    bop (list): Terms of prepared boundary derivative operator. If None, Dirichlet conditions are applied.
                    var (int): Indicates for which dependent variable it is necessary to apply the boundary condition. For a single equation, it is 0.
        
                Returns:
                    torch.Tensor: Calculated boundary condition values, ensuring periodicity.
        """

        if bop is None:
            b_op_val = self._apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self._apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self._apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self._apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode in ('autograd', 'mat'):
                b_op_val = self._apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self._apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def _apply_data(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """
        Applies either Dirichlet or Neumann boundary conditions to the solution.
        
        This method determines which type of boundary condition to apply based on the provided
        derivative operator and then calculates the condition value. This ensures that known
        constraints on the solution at the boundaries are enforced, guiding the equation discovery
        process towards solutions that respect these constraints.
        
        Args:
            bnd (torch.Tensor): Terms (data points) of prepared boundary conditions.
            bop (list): Terms of prepared data derivative operator. If None, Dirichlet conditions are applied; otherwise, Neumann conditions are applied using this operator.
            var (int): Indicates for which dependent variable it is necessary to apply the data condition. For a single equation, this is 0.
        
        Returns:
            torch.Tensor: Calculated data condition.
        """
        if bop is None:
            b_op_val = self._apply_dirichlet(bnd, var).reshape(-1, 1)
        else:
            b_op_val = self._apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond: dict) -> torch.Tensor:
        """
        Calculates the value of the boundary operator based on the specified boundary condition type.
        
                This function acts as a dispatcher, selecting the appropriate method
                to apply based on the 'type' key within the boundary condition dictionary.
                This allows to handle different types of boundary conditions such as Dirichlet,
                Neumann, periodic or data-driven in a unified manner.
        
                Args:
                    bcond (dict): A dictionary containing the terms of the prepared boundary condition,
                                  as generated by the `bnd_prepare` method in the `input_preprocessing` module.
                                  This dictionary must contain a 'type' key specifying the boundary condition type
                                  (e.g., 'dirichlet', 'operator', 'periodic', 'data') and other keys
                                  relevant to that type (e.g., 'bnd', 'var', 'bop').
        
                Returns:
                    torch.Tensor: The calculated value of the boundary operator, represented as a PyTorch tensor.
                                  The specific calculation depends on the boundary condition type specified in `bcond`.
        """

        if bcond['type'] == 'dirichlet':
            b_op_val = self._apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self._apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self._apply_periodic(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        elif bcond['type'] == 'data':
            b_op_val = self._apply_data(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        return b_op_val

    def apply_bcs(self) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Applies boundary and data conditions to construct matrices of predicted and true boundary values, organized by boundary type. This is done to prepare boundary conditions for comparison with PDE solutions.
        
                Args:
                    self (Bounds): An instance of the Bounds class containing prepared boundary conditions.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor, list, list]:
                        - bval (torch.Tensor): A matrix where each column represents the predicted boundary values for a specific boundary type.
                        - true_bval (torch.Tensor): A matrix where each column represents the true boundary values for a specific boundary type.
                        - keys (list): A list of boundary types, corresponding to the columns in the `bval` and `true_bval` matrices.
                        - bval_length (list): A list containing the length of each boundary type column.
        """

        bval_dict = {}
        true_bval_dict = {}

        for bcond in self.prepared_bconds:
            try:
                bval_dict[bcond['type']] = torch.cat((bval_dict[bcond['type']],
                                                    self.b_op_val_calc(bcond).reshape(-1)))
                true_bval_dict[bcond['type']] = torch.cat((true_bval_dict[bcond['type']],
                                                    bcond['bval'].reshape(-1)))
            except:
                bval_dict[bcond['type']] = self.b_op_val_calc(bcond).reshape(-1)
                true_bval_dict[bcond['type']] = bcond['bval'].reshape(-1)

        bval, true_bval, keys, bval_length = dict_to_matrix(
                                                    bval_dict, true_bval_dict)

        return bval, true_bval, keys, bval_length
