
"""preprocessing module for operator (equation) and boundaries.
"""

from copy import deepcopy
from typing import Union
import numpy as np
import torch

from epde.solver.points_type import Points_type
from epde.solver.finite_diffs import Finite_diffs
from epde.solver.device import check_device

def lambda_prepare(val: torch.Tensor,
                   lambda_: Union[int, list, torch.Tensor]) -> torch.Tensor:
    """
    Prepares regularization parameters to match the dimensions of the operator or boundary condition tensor. This ensures that each term in the equation or boundary condition is regularized appropriately.
    
        Args:
            val (torch.Tensor): Operator tensor or boundary value tensor. The shape determines the number of terms to regularize.
            lambda_ (Union[int, list, torch.Tensor]): Regularization parameter(s). Can be a single value (applied to all terms), a list of values (one for each term), or a pre-defined tensor.
    
        Returns:
            torch.Tensor: A tensor containing the regularization parameters, reshaped to have a row for each term in the operator/boundary condition. The number of columns matches the number of terms in `val`.
    """

    if isinstance(lambda_, torch.Tensor):
        return lambda_

    if isinstance(lambda_, (int, float)):
        try:
            lambdas = torch.ones(val.shape[-1], dtype=val.dtype) * lambda_
        except:
            lambdas = torch.tensor(lambda_, dtype=val.dtype)
    elif isinstance(lambda_, list):
        lambdas = torch.tensor(lambda_, dtype=val.dtype)

    return lambdas.reshape(1,-1)

class EquationMixin:
    """
    Auxiliary class. This one contains some methods that uses in other classes.
    """


    @staticmethod
    def equation_unify(equation: dict) -> dict:
        """
        Unifies the equation format by ensuring the presence of 'var' and converting 'pow' and the differential direction to lists when necessary. This ensures that all operators within the equation have a consistent structure suitable for subsequent processing and solving.
        
                Args:
                    equation (dict): A dictionary representing the equation, where keys are operator labels and values are dictionaries containing operator parameters like 'pow', differential direction, and potentially 'var'.
        
                Returns:
                    dict: The modified equation dictionary with a unified format for all operators, ensuring 'var' is present and 'pow' and differential direction are lists when appropriate.
        """

        for operator_label in equation.keys():
            operator = equation[operator_label]
            dif_dir = list(operator.keys())[1]
            try:
                operator['var']
            except:
                if isinstance(operator['pow'], (int, float)):
                    operator[dif_dir] = [operator[dif_dir]]
                    operator['pow'] = [operator['pow']]
                    operator['var'] = [0]
                elif isinstance(operator['pow'], list):
                    operator['var'] = [0 for _ in operator['pow']]
                continue
            if isinstance(operator['pow'], (int, float)):
                operator[dif_dir] = [operator[dif_dir]]
                operator['pow'] = [operator['pow']]
                operator['var'] = [operator['var']]

        return equation

    @staticmethod
    def closest_point(grid: torch.Tensor, target_point: float) -> int:
        """
        Finds the grid point closest to a given boundary point.
        
        This is crucial for accurately applying boundary conditions when solving
        differential equations numerically. By identifying the nearest grid
        point, we can enforce the boundary condition at a location that is
        consistent with the domain discretization.
        
        Args:
            grid (torch.Tensor): The computational grid representing the domain.
            target_point (float): The location of the boundary point.
        
        Returns:
            int: The index of the grid point closest to the target point.
        """

        min_dist = np.inf
        pos = 0
        min_pos = 0
        for point in grid:
            dist = torch.linalg.norm(point - target_point)
            if dist < min_dist:
                min_dist = dist
                min_pos = pos
            pos += 1
        return min_pos

    @staticmethod
    def convert_to_double(bnd: Union[list, np.array]) -> float:
        """
        Converts input data (either a list or a NumPy array) to a double-precision floating-point representation, ensuring compatibility with PyTorch operations for equation discovery.
        
                Args:
                    bnd (Union[list, np.array]): The input data, which can be a list of arrays or a NumPy array, representing points or values.
        
                Returns:
                    Union[list, torch.Tensor]: The data converted to double precision. If the input is a list, the function recursively converts each element of the list. If the input is a NumPy array, it's converted to a PyTorch tensor with double precision.
        
                Why:
                    This conversion is crucial for maintaining numerical stability and precision during the equation discovery process, especially when dealing with complex calculations and optimizations within the PyTorch framework.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.convert_to_double(cur_bnd)
            return bnd
        elif isinstance(bnd, np.ndarray):
            return torch.from_numpy(bnd).double()
        return bnd.double()

    @staticmethod
    def search_pos(grid: torch.Tensor, bnd) -> list:
        """
        Identifies the indices of points within a grid, or finds the closest grid points if exact matches are not found.
        
        This function is crucial for mapping boundary conditions or specific points of interest
        onto the discrete grid used for equation solving. By finding the grid positions
        corresponding to these points, the framework can accurately apply boundary conditions
        and evaluate the solution at specific locations.
        
        Args:
            grid (torch.Tensor): A tensor representing the spatial grid where each row is a point in n-dimensional space.
            bnd (torch.Tensor or list of torch.Tensor): Points whose positions on the grid need to be determined.
                If a list is provided, the function recursively processes each point in the list.
        
        Returns:
            list: A list of integer indices representing the positions of the input points on the grid.
                If an exact match is not found, the index of the closest grid point is returned.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.search_pos(grid, cur_bnd)
            return bnd
        pos_list = []
        for point in bnd:
            try:
                pos = int(torch.where(torch.all(
                    torch.isclose(grid, point), dim=1))[0])
            except Exception:
                pos = EquationMixin.closest_point(grid, point)
            pos_list.append(pos)
        return pos_list

    @staticmethod
    def bndpos(grid: torch.Tensor, bnd: torch.Tensor) -> Union[list, int]:
        """
        Returns the indices of grid points that correspond to boundary conditions. This is crucial for enforcing constraints on the solution space when solving differential equations.
        
                Args:
                    grid (torch.Tensor): The spatial grid on which the solution is defined.
                    bnd (torch.Tensor): The boundary values that the solution must satisfy.
        
                Returns:
                    Union[list, int]: A list of indices representing the positions of the boundary points on the grid.
        """

        if grid.shape[0] == 1:
            grid = grid.reshape(-1, 1)
        grid = grid.double()
        bnd = EquationMixin.convert_to_double(bnd)
        bndposlist = EquationMixin.search_pos(grid, bnd)
        return bndposlist


class Equation_NN(EquationMixin, Points_type):
    """
    Class for preprocessing input data: grid, operator, bconds in unified
        form. Then it will be used for determine solution by 'NN' method.
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator:  Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str = '1',
                 boundary_order: str = '2'):
        """
        Prepares the problem setup for solving differential equations using neural networks. This involves defining the equation itself, specifying boundary conditions, and setting parameters for the numerical scheme.
        
                Args:
                    grid (torch.Tensor): Tensor representing the spatial or temporal domain where the solution is sought.
                    operator (Union[dict, list]): Definition of the differential equation to be solved.
                    bconds (list): List of boundary conditions that constrain the solution.
                    h (float, optional): Discretization parameter (grid resolution) for finite difference approximations. Defaults to 0.001.
                    inner_order (str, optional): Accuracy order for finite difference scheme within the domain. Defaults to '1'.
                    boundary_order (str, optional): Accuracy order for finite difference scheme at the boundaries. Defaults to '2'.
        
                Returns:
                    None
        """

        super().__init__(grid)
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def _operator_to_type_op(self,
                            dif_direction: list,
                            nvars: int,
                            axes_scheme_type: str) -> list:
        """
        Converts a symbolic differentiation operator into a list of finite difference approximations,
        tailored to the specified point types and differentiation direction.
        
                This conversion is essential for numerically evaluating the equation on a grid,
                where derivatives are approximated using finite differences.
        
                Args:
                    dif_direction (list): Differentiation direction, represented as a list of lists (e.g., `[[0, 0]]` for d2/dx2).
                    nvars (int): Dimensionality of the problem (number of independent variables).
                    axes_scheme_type (str): Type of finite difference scheme to use ('central' or a combination of 'f' and 'b' for forward and backward).
        
                Returns:
                    list: A list containing two lists:
                        - The first list contains the finite difference schemes corresponding to each term in the differentiation direction.
                        - The second list contains the orders of accuracy for each scheme.
        """

        if axes_scheme_type == 'central':
            scheme_variant = self.inner_order
        else:
            scheme_variant = self.boundary_order

        fin_diff_list = []
        s_order_list = []
        for term in dif_direction:
            scheme, s_order = Finite_diffs(
                term, nvars, axes_scheme_type).scheme_choose(
                scheme_variant, h=self.h)
            fin_diff_list.append(scheme)
            s_order_list.append(s_order)
        return [fin_diff_list, s_order_list]

    def _finite_diff_scheme_to_grid_list(self,
                                        finite_diff_scheme: list,
                                        grid_points: torch.Tensor) -> list:
        """
        Converts a finite difference scheme, represented as integer steps, into a list of grid points shifted according to the scheme.
        
                This transformation is crucial for evaluating differential operators numerically on a grid. By shifting the grid points according to the finite difference scheme, the method effectively approximates derivatives at each point.
        
                Args:
                    finite_diff_scheme (list): A list representing a single term in the finite difference scheme. Each element in the list corresponds to the shift along a particular axis. If element is None - that means that no shift is required
                    grid_points (torch.Tensor): The original grid points that will be shifted based on the finite difference scheme.
        
                Returns:
                    list: A list of tensors, where each tensor represents the grid points shifted according to a specific term in the finite difference scheme.
        """

        s_grid_list = []
        for shifts in finite_diff_scheme:
            if shifts is None:
                s_grid_list.append(grid_points)
            else:
                s_grid = grid_points
                for j, axis in enumerate(shifts):
                    s_grid = self.shift_points(s_grid, j, axis * self.h)
                s_grid_list.append(s_grid)
        return s_grid_list

    def _checking_coeff(self,
                       coeff: Union[int, float, torch.Tensor, callable],
                       grid_points: torch.Tensor) -> torch.Tensor:
        """
        Checks the type of a coefficient used in defining the equation, ensuring it's compatible with the computational graph and data structures. This preprocessing step is crucial for correct equation formation and subsequent analysis.
        
                Args:
                    coeff (Union[int, float, torch.Tensor, callable]): The coefficient to be checked. It can be a constant (int or float), a tensor, or a callable function.
                    grid_points (torch.Tensor): The grid points where the coefficient is evaluated if it's a callable or a tensor.
        
                Raises:
                    NameError: If the coefficient is not of the allowed types (int, float, torch.Tensor, or callable).
        
                Returns:
                    torch.Tensor: The processed coefficient, ensuring it's in a suitable format (usually a tensor) for further computations within the equation.
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = (coeff, grid_points)
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            pos = self.bndpos(self.grid, grid_points)
            coeff1 = coeff[pos].reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _type_op_to_grid_shift_op(self, fin_diff_op: list, grid_points) -> list:
        """
        Converts a finite difference operator for a specific grid type into a grid-shifted operator form, preparing it for numerical evaluation. This involves mapping coefficients (which can be integers, functions, or arrays) to the appropriate subgrid locations based on the point type. This conversion is crucial for applying the finite difference scheme across the grid.
        
                Args:
                    fin_diff_op (list): The finite difference operator for a specific grid type, as a result of `operator_to_type_op`.
                    grid_points: The grid points associated with the finite difference scheme.
        
                Returns:
                    list: The grid-shifted form of the differential operator, ready for use in the numerical algorithm for the given grid type.
        """

        shift_grid_op = []
        for term1 in fin_diff_op:
            grid_op = self._finite_diff_scheme_to_grid_list(term1, grid_points)
            shift_grid_op.append(grid_op)
        return shift_grid_op

    def _one_operator_prepare(self,
                             operator: dict,
                             grid_points: torch.Tensor,
                             points_type: str) -> dict:
        """
        Prepares a single operator term for equation evaluation by converting symbolic representations into numerical operations on the grid. This involves unifying equation formats, checking and converting coefficients, and transforming differential operators into grid-compatible shift operations.
        
                Args:
                    operator (dict): A dictionary representing the operator term, containing coefficients and differential operator definitions.
                    grid_points (torch.Tensor): The coordinates of the grid points where the equation is evaluated.
                    points_type (str): The type of grid points (e.g., 'uniform', 'chebyshev').
        
                Returns:
                    dict: The prepared operator term with coefficients and differential operators converted into a format suitable for grid-based computation. The operator is prepared to be evaluated on the grid by converting symbolic representations into numerical operations.
        """

        nvars = self.grid.shape[-1]
        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            dif_term = list(term.keys())[1]
            term['coeff'] = self._checking_coeff(term['coeff'], grid_points)
            term[dif_term] = self._operator_to_type_op(term[dif_term],
                                                      nvars, points_type)
            term[dif_term][0] = self._type_op_to_grid_shift_op(
                term[dif_term][0], grid_points)
        return operator

    def operator_prepare(self) -> list:
        """
        Prepares the operator(s) for equation discovery by associating them with grid points.
        
                This method ensures that the operators are correctly formatted and linked to the spatial grid,
                which is a crucial step for evaluating the equation's fitness against the data.
                If multiple equations are present, it prepares each operator individually.
        
                Args:
                    None
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a prepared operator
                          associated with the central grid points.
        """

        grid_points = self.grid_sort()['central']
        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self._one_operator_prepare(
                    self.operator[i], grid_points, 'central')
                prepared_operator.append(equation)
        else:
            equation = self._one_operator_prepare(
                self.operator, grid_points, 'central')
            prepared_operator = [equation]

        return prepared_operator

    def _apply_bnd_operators(self, bnd_operator: dict, bnd_dict: dict) -> list:
        """
        Applies the boundary operator to each specified type of boundary point.
        
                This function iterates through different types of boundary points and prepares the corresponding differential operator for each type. This is a crucial step in constructing the complete equation system, ensuring that boundary conditions are correctly incorporated into the discovered equations.
        
                Args:
                    bnd_operator (dict): A dictionary containing the boundary operator in its initial form.
                    bnd_dict (dict): A dictionary where keys represent the type of boundary points and values are the corresponding boundary points themselves.
        
                Returns:
                    list: A list of prepared differential operators, one for each type of boundary point, ready for use in the equation discovery process.
        """

        operator_list = []
        for points_type in list(bnd_dict.keys()):
            equation = self._one_operator_prepare(
                deepcopy(bnd_operator), bnd_dict[points_type], points_type)
            operator_list.append(equation)
        return operator_list

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for use in the equation discovery process. It sorts and applies boundary operators to each condition based on the grid and boundary type.
        
                Args:
                    self: The Equation_NN object containing the grid, boundary conditions, and operators.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a boundary condition with sorted grid locations and applied boundary operators. The boundary operators are applied to enforce specific constraints or conditions at the boundaries of the domain, ensuring that the discovered equations are physically meaningful and consistent with the observed data.
        """

        grid_dict = self.grid_sort()

        for bcond in self.bconds:
            bnd_dict = self.bnd_sort(grid_dict, bcond['bnd'])
            if bcond['bop'] is not None:
                if bcond['type'] == 'periodic':
                    bcond['bop'] = [self._apply_bnd_operators(
                        bcond['bop'], i) for i in bnd_dict]
                else:
                    bcond['bop'] = self._apply_bnd_operators(
                        bcond['bop'], bnd_dict)
        return self.bconds


class Equation_autograd(EquationMixin):
    """
    Prepares equation for autograd method (i.e., from conventional form to input form).
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list):
        """
        Prepares the problem for solving by storing the grid, equation, and boundary conditions.
        
                This setup is crucial for the subsequent application of the evolutionary algorithm to find the optimal equation representation.
        
                Args:
                    grid (torch.Tensor): Tensor of a n-D points where the solution is evaluated.
                    operator (Union[dict, list]): Equation to be solved, represented in a suitable format.
                    bconds (list): Boundary conditions that constrain the solution space.
        
                Returns:
                    None
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def _checking_coeff(self,
                       coeff: Union[int, float, torch.Tensor]) -> Union[int, float, torch.Tensor]:
        """
        Validates and prepares the coefficient for use in equation operations.
        
        This method ensures that the provided coefficient is of a supported type
        (int, float, torch.Tensor, or callable) and converts it into a suitable
        format for subsequent calculations within the equation. This ensures
        compatibility with the framework's computational graph.
        
        Args:
            coeff (Union[int, float, torch.Tensor]): The coefficient to be checked and prepared.
        
        Raises:
            NameError: If the provided "coeff" is not one of the supported types
                (torch.Tensor, callable, int, or float).
        
        Returns:
            Union[int, float, torch.Tensor]: The validated and potentially reshaped coefficient.
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = coeff
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            coeff1 = coeff.reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _one_operator_prepare(self, operator: dict) -> dict:
        """
        Prepares a single operator by unifying its format and checking the coefficients of its terms.
        
        This ensures that the operator is in a consistent format and that the coefficients
        are valid before further processing, which is crucial for the equation discovery process.
        
        Args:
            operator (dict): The operator in its initial input form.
        
        Returns:
            dict: The processed operator with unified format and checked coefficients.
        """

        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            term['coeff'] = self._checking_coeff(term['coeff'])
        return operator

    def operator_prepare(self) -> list:
        """
        Prepares the operators for symbolic differentiation. It handles both single equations and systems of equations by unifying and preparing each operator.
        
                Args:
                    self: The instance of the Equation_autograd class.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a prepared operator ready for symbolic differentiation. The length of the list corresponds to the number of equations in the system, or 1 if it's a single equation.
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(self._one_operator_prepare(equation))
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [self._one_operator_prepare(equation)]

        return prepared_operator

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for use in the equation discovery process.
        
        This method ensures that the boundary conditions are in the correct format
        (a list of dictionaries) for subsequent calculations within the EPDE framework.
        If no boundary conditions are provided, it returns None.
        
        Args:
            self: The Equation_autograd object.
        
        Returns:
            list: A list of dictionaries, where each dictionary represents a boundary condition.
                  Returns None if no boundary conditions are specified.
        
        Why:
            This preparation step is crucial for ensuring compatibility between the user-defined
            boundary conditions and the equation discovery algorithms, allowing the framework
            to correctly incorporate these constraints into the equation search and validation process.
        """

        if self.bconds is None:
            return None
        else:
            return self.bconds


class Equation_mat(EquationMixin):
    """
    Class realizes input data preprocessing (operator and boundary conditions
        preparing) for 'mat' method.
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[list, dict],
                 bconds: list):
        """
        Prepares the problem setup by storing the computational grid, differential operator, and boundary conditions.
        
                This initialization is crucial for subsequent calculations, ensuring that all necessary components of the problem are readily accessible.
        
                Args:
                    grid (torch.Tensor): The computational grid where the solution is approximated.
                    operator (Union[list, dict]): Definition of the differential operator.
                    bconds (list): Boundary conditions applied to constrain the solution.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def operator_prepare(self) -> list:
        """
        Transforms the input operator into a standardized format suitable for subsequent matrix construction.
        
        This method ensures that the operator, whether a single equation or a list of equations,
        is converted into a unified representation, facilitating the creation of the matrix
        required for the equation discovery process.
        
        Args:
            self: Instance of the Equation_mat class, containing the operator to be prepared.
        
        Returns:
            list: A list of unified equations, ready for matrix construction.  Each element
                  in the list represents a single equation in a standardized dictionary format.
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(equation)
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [equation]

        return prepared_operator

    def _point_position(self, bnd: torch.Tensor) -> list:
        """
        Locates the grid indices corresponding to boundary points.
        
        This method is crucial for mapping boundary conditions onto the computational grid,
        ensuring accurate problem representation. By identifying the precise grid locations
        of the boundary points, the method facilitates the application of boundary conditions
        during the equation solving process.
        
        Args:
            bnd (torch.Tensor): A tensor representing the boundary subgrid. Each element in the tensor
                corresponds to a point on the boundary.
        
        Returns:
            list: A list of tuples, where each tuple contains the indices of the grid point
                closest to the corresponding boundary point. The indices represent the
                position of the boundary point within the computational grid.
        """

        bpos = []
        for pt in bnd:
            if self.grid.shape[0] == 1:
                point_pos = (torch.tensor(self.bndpos(self.grid, pt)),)
            else:
                prod = (torch.zeros_like(self.grid[0]) + 1).bool()
                for axis in range(self.grid.shape[0]):
                    axis_intersect = torch.isclose(
                        pt[axis].float(), self.grid[axis].float())
                    prod *= axis_intersect
                point_pos = torch.where(prod == True)
            bpos.append(point_pos)
        return bpos

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for equation discovery by unifying and positioning them.
        
                This method processes the boundary conditions, converting boundary definitions
                into numerical positions within the domain and unifying the boundary operators
                to ensure compatibility with the equation discovery process. This ensures
                that boundary conditions are in a standardized format suitable for use
                in subsequent equation solving and analysis.
        
                Args:
                    self: The Equation_mat object.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a boundary
                          condition with updated 'bnd' (boundary position) and 'bop'
                          (boundary operator) fields.
        """

        for bcond in self.bconds:
            if bcond['type'] == 'periodic':
                bpos = []
                for bnd in bcond['bnd']:
                    bpos.append(self._point_position(bnd))
            else:
                bpos = self._point_position(bcond['bnd'])
            if bcond['bop'] is not None:
                bcond['bop'] = self.equation_unify(bcond['bop'])
            bcond['bnd'] = bpos
        return self.bconds


class Operator_bcond_preproc():
    """
    Interface for preparing equations due to chosen calculation method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str ='1',
                 boundary_order: str ='2'):
        """
        Initializes the Operator_bcond_preproc class, preparing the problem setup for numerical solution.
        
                This involves storing the grid, equation, and boundary conditions,
                along with discretization parameters, to facilitate the finite difference approximation.
        
                Args:
                    grid (torch.Tensor): Grid representing the domain where the equation is solved. Typically obtained from cartesian_prod or meshgrid.
                    operator (Union[dict, list]): Definition of the differential equation to be solved.
                    bconds (list): List of boundary conditions that constrain the solution.
                    h (float, optional): Discretization parameter (grid resolution) for the finite difference method. Defaults to 0.001.
                    inner_order (str, optional): Accuracy order for the finite difference scheme in the interior of the domain. Defaults to '1'.
                    boundary_order (str, optional): Accuracy order for the finite difference scheme at the boundaries. Defaults to '2'.
        
                Returns:
                    None
        """

        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """
        Specifies the numerical approach for solving the problem.
        
        This choice influences how the equation is represented and solved, 
        affecting both accuracy and computational cost. Different strategies 
        are suitable for different problem types and computational resources.
        
        Args:
            strategy (str): Calculation method. (i.e., "NN", "autograd", "mat").
        
        Returns:
            Union[Equation_NN, Equation_mat, Equation_autograd]: A given calculation method.
        """

        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h,
                               inner_order=self.inner_order,
                               boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)
