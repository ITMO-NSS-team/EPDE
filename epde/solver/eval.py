"""Module for operatoins with operator and boundaru con-ns."""

from typing import Tuple, Union, List
import torch

from epde.solver.points_type import Points_type
from epde.solver.derivative import Derivative
from epde.solver.device import device_type, check_device
from epde.solver.utils import PadTransform


def integration(func: torch.Tensor,
                grid: torch.Tensor,
                power: int = 2) \
                -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """ Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func (torch.Tensor): operator multiplied on test function
        grid (torch.Tensor): array of a n-D points.
        power (int, optional): power of func points. Defults to 2.

    Returns:
        'result' is integration result through one grid axis
        'grid' is initial grid without last column or zero (if grid.shape[N,1])
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
    """ Function for bounaries values matrix creation from dictionary.

    Args:
        bval (dict): dictionary with predicted boundaries values,
              where keys are boundaries types.
        true_bval (dict): dictionary with true boundaries values,
                   where keys are boundaries types.

    Returns:
        matrix_bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
        matrix_true_bval (torch.Tensor):matrix, where each column is true
                           boundary values of one boundary type.
        keys (list): boundary types list corresponding matrix_bval columns.
        len_list (list): list of length of each boundary type column.
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
                 weak_form: List[callable],
                 derivative_points: int):
        """
        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_operator (Union[list,dict]): prepared (after Equation class) operator.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (List[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
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
        self.derivative = Derivative(self.model,
                                self.derivative_points).set_strategy(self.mode).take_derivative

    def apply_operator(self,
                       operator: list,
                       grid_points: Union[torch.Tensor, None]) -> torch.Tensor:
        """ Deciphers equation in a single grid subset to a field.

        Args:
            operator (list): prepared (after Equation class) operator. See
            input_preprocessing.operator_prepare()
            grid_points (Union[torch.Tensor, None]): Points, where numerical
            derivative is calculated. **Uses only in 'autograd' and 'mat' modes.**

        Returns:
            total (torch.Tensor): Decoded operator on a single grid subset.
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
        """ Computes PDE residual.

        Returns:
            torch.Tensor: P/O DE residual.
        """

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(
                self.prepared_operator[0], self.sorted_grid).reshape(-1,1)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], self.sorted_grid).reshape(-1,1))
            op = torch.cat(op_list, 1)
        return op


    def _weak_pde_compute(self) -> torch.Tensor:
        """ Computes PDE residual in weak form.

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
        """ Corresponding to form (weak or strong) calculate residual of operator.

        Returns:
            torch.Tensor: operator residual.
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
        """_summary_

        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_bconds (Union[list,dict]): prepared (after Equation class) baund-y con-s.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (List[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
        """
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = Operator(self.grid, self.prepared_bconds,
                                       self.model, self.mode, weak_form,
                                       derivative_points)

    def _apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """ Method only for *NN* mode. Calculate boundary conditions with derivatives
            to use them in _apply_neumann method.

        Args:
            operator_set (list): list with prepared (after Equation_NN class) boundary operators.
            For details to Equation_NN.operator_prepare method.

        Returns:
            torch.Tensor: Decoded boundary operator on the whole grid.
        """

        field_part = []
        for operator in operator_set:
            field_part.append(self.operator.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def _apply_dirichlet(self, bnd: torch.Tensor, var: int) -> torch.Tensor:
        """ Applies Dirichlet boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            For more deatails to input_preprocessing (bnd_prepare maethos).
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition.
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
        """ Applies boundary conditions with derivative operators.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.

        Returns:
            torch.Tensor: calculated boundary condition.
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
        """ Applies periodic boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition
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
        """ Method for applying known data about solution.

        Args:
            bnd (torch.Tensor): terms (data points) of prepared boundary conditions.
            bop (list): terms of prepared data derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the data condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated data condition.
        """
        if bop is None:
            b_op_val = self._apply_dirichlet(bnd, var).reshape(-1, 1)
        else:
            b_op_val = self._apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond: dict) -> torch.Tensor:
        """ Auxiliary function. Serves only to choose *type* of the condition and evaluate one.

        Args:
            bcond (dict): terms of prepared boundary conditions
            (see input_preprocessing module -> bnd_prepare method).

        Returns:
            torch.Tensor: calculated operator on the boundary.
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
        """ Applies boundary and data conditions for each *type* in prepared_bconds.

        Returns:
            bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
            true_bval (torch.Tensor):matrix, where each column is true
                            boundary values of one boundary type.
            keys (list): boundary types list corresponding matrix_bval columns.
            bval_length (list): list of length of each boundary type column.
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
