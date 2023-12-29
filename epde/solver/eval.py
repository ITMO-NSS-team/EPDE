import torch
from typing import Tuple, Union, List

from epde.solver.points_type import Points_type
from epde.solver.derivative import Derivative
from epde.solver.device import device_type, check_device
from epde.solver.utils import PadTransform


def integration(func: torch.tensor, grid, pow: Union[int, float] = 2) \
                -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """
    Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func: operator multiplied on test function
        grid: array of a n-D points.
        pow: string (sqr ar abs) power of func points

    Returns:
        tuple(result, grid)
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
    U = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            U += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** pow + func[i - 1] ** pow) / 2
        else:
            result.append(U)
            marker = grid[i][column]
            index.append(i)
            U = 0.
    if column == -1:
        return U, 0.
    else:
        result.append(U)
        grid = grid[index, :-1]
        return result, grid


def dict_to_matrix(bval: dict, true_bval: dict):
    '''Function for bounaries values matrix creation from dictionary

    Args:
        bval: dictionary with predicted boundaries values,
              where keys are boundaries types
        true_bval: dictionary with true boundaries values,
                   where keys are boundaries types

    Returns:
        tuple(matrix_bval, matrix_true_bval, keys, len_list)
        'matrix_bval' matrix, where each column is predicted
                      boundary values of one boundary type
        'matrix_true_bval' matrix, where each column is true
                           boundary values of one boundary type
        'keys' boundary types list corresponding matrix_bval columns
        'len_list' list of length of each boundary type column
    '''

    keys = list(bval.keys())
    max_len = max([len(i) for i in bval.values()])
    pad = PadTransform(max_len, 0)
    matrix_bval = pad(bval[keys[0]]).float().reshape(-1,1)
    matrix_true_bval = pad(true_bval[keys[0]]).float().reshape(-1,1)
    len_list = [len(bval[keys[0]])]
    for key in keys[1:]:
        bval_i = pad(bval[key]).float().reshape(-1,1)
        true_bval_i = pad(true_bval[key]).float().reshape(-1,1)
        matrix_bval = torch.hstack((matrix_bval, bval_i))
        matrix_true_bval = torch.hstack((matrix_true_bval, true_bval_i))
        len_list.append(len(bval[key]))

    return matrix_bval, matrix_true_bval, keys, len_list


class Operator():
    """
    Class for differential equation calculation.
    """
    def __init__(self, grid: torch.Tensor, prepared_operator: Union[list,dict],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str,
                 weak_form: List[callable], derivative_points: int):
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.derivative_points = derivative_points
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode == 'autograd' or self.mode == 'mat':
            self.sorted_grid = self.grid

    def apply_operator(self, operator: list, grid_points: Union[torch.Tensor, None]) -> torch.Tensor:
        """
        Deciphers equation in a single grid subset to a field.

        Args:
            operator: single (len(subset)==1) operator in input form. See
            input_preprocessing.operator_prepare()
            grid_points: Points, where numerical derivative is calculated. **Uses only in 'autograd' and 'mat' modes.**
        Returns:
            Decoded operator on a single grid subset
        """
        derivative = Derivative(self.model,self.derivative_points).set_strategy(self.mode).take_derivative
        for term in operator:
            term = operator[term]
            dif = derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def pde_compute(self) -> torch.Tensor:
        """
        Computes PDE residual.

        Returns:
            PDE residual.
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


    def weak_pde_compute(self, weak_form) -> torch.Tensor:
        """
        Computes PDE residual in weak form.

        Args:
            weak_form: list of basis functions
        Returns:
            weak PDE residual.
        """
        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self.pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for k in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list).reshape(1,-1)

    def operator_compute(self):
        if self.weak_form == None or self.weak_form == []:
            return self.pde_compute()
        else:
            return self.weak_pde_compute(self.weak_form)


class Bounds():
    """
    Class for boundary and initial conditions calculation.
    """
    def __init__(self, grid: torch.Tensor, prepared_bconds: Union[list,dict],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str,
                 weak_form: List[callable], derivative_points: int):
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.apply_operator = Operator(self.grid, self.prepared_bconds,
                                       self.model, self.mode, weak_form,
                                       derivative_points).apply_operator

    def apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """
        Deciphers equation in a whole grid to a field.
        Args:
            operator_set: Multiple (len(subset)>=1) operators in input form. See
            input_preprocessing.operator_prepare().
        Returns:
            Decoded boundary operator on the whole grid.
        """
        field_part = []
        for operator in operator_set:
            field_part.append(self.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def apply_dirichlet(self, bnd: torch.Tensor, var: int) -> torch.Tensor:
        """
        Applies Dirichlet boundary conditions.

        Args:
            bnd: terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
            var: indicates for which equation it is necessary to apply the boundary condition.
        Returns:
            calculated boundary condition.
        """
        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                    b_op_val.append(self.model[var][position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def apply_neumann(self, bnd: torch.Tensor, bop: list) -> torch.Tensor:
        """
        Applies periodic boundary conditions.

        Args:
           bnd: terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
           bop: terms of operator on boundary.
        Returns:
           calculated boundary condition.
        """
        if self.mode == 'NN':
            b_op_val = self.apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.apply_operator(bop, bnd)
        elif self.mode == 'mat':
            var = bop[list(bop.keys())[0]]['var'][0]
            b_op_val = self.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                b_val.append(b_op_val[var][position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def apply_periodic(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """
        Applies periodic boundary conditions.

        Args:
           bnd: terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
           bop: terms of operator on boundary.
           var: indicates for which equation it is necessary to apply the boundary condition.

        Returns:
           calculated boundary condition
        """

        if bop is None:
            b_op_val = self.apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self.apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self.apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self.apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode == 'autograd' or self.mode == 'mat':
                b_op_val = self.apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self.apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def apply_data(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        '''method for applying data'''
        if bop is None:
            b_op_val = self.apply_dirichlet(bnd, var).reshape(-1, 1)
        else:
            b_op_val = self.apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond) -> torch.Tensor:
        """
        Auxiliary function. Serves only to evaluate operator on the boundary.

        Args:
            bcond:  terms of prepared boundary conditions (see input_preprocessing.bnd_prepare) in input form.
        Returns:
            calculated operator on the boundary.
        """
        if bcond['type'] == 'dirichlet':
            b_op_val = self.apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self.apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self.apply_periodic(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        elif bcond['type'] == 'data':
            b_op_val = self.apply_data(bcond['bnd'], bcond['bop'],
                                           bcond['var'])
        return b_op_val

    def apply_bcs(self) -> Tuple[dict, dict]:
        """
        Applies boundary conditions for each term in prepared_bconds.

        Returns:

            - model output with boundary conditions at the input.
            - true boundary values.

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
