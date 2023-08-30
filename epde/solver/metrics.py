import torch
import numpy as np
from copy import deepcopy
from typing import Tuple, Union

from epde.solver.points_type import Points_type
from epde.solver.derivative import Derivative
from epde.solver.device import device_type, check_device
import epde.solver.input_preprocessing

flatten_list = lambda t: [item for sublist in t for item in sublist]


def integration(func: torch.tensor, grid, pow: Union[int, float] = 2) -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
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


class DE_operator():
    """
    Class for differential equation calculation.
    """
    def __init__(self, grid: torch.Tensor, prepared_operator: dict,
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str):
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
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
        derivative = Derivative(self.model).set_strategy(
            self.mode).take_derivative
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
                self.prepared_operator[0], self.sorted_grid)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], self.sorted_grid))
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
            return torch.cat(sol_list)


class Bounds():
    """
    class for boundary and initial conditions calculation.
    """
    def __init__(self, grid: torch.Tensor, prepared_bconds: dict,
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str):
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.apply_operator = DE_operator(self.grid, self.prepared_bconds,
                                          self.model, self.mode).apply_operator


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
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_op_val.append(self.model[:, position])
                else:
                    b_op_val.append(self.model[position])
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
            b_op_val = self.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                    b_val.append(b_op_val[:, position])
                else:
                    b_val.append(b_op_val[position])
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
        return b_op_val

    def apply_bconds_operator(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function. Serves only to evaluate boundary values and true boundary values.
        Returns:
            * **b_val** -- calculated model boundary values.\n
            * **true_b_val** -- true grid boundary values.
        """
        true_b_val_list = []
        b_val_list = []

        for bcond in self.prepared_bconds:
            truebval = bcond['bval'].reshape(-1, 1)
            true_b_val_list.append(truebval)
            b_op_val = self.b_op_val_calc(bcond)
            b_val_list.append(b_op_val)

        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list).reshape(-1, 1)

        return b_val, true_b_val


class Solution():
    """
    class for different loss functions calculation.
    """
    def __init__(self, grid: torch.Tensor, equal_cls: Union[epde.solver.input_preprocessing.Equation_NN,
                                                            epde.solver.input_preprocessing.Equation_mat,
                                                            epde.solver.input_preprocessing.Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str):
        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
        equal_copy = deepcopy(equal_cls)
        self.prepared_operator = equal_copy.operator_prepare()
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = DE_operator(self.grid, self.prepared_operator, self.model,
                                   self.mode)
        self.bconds = Bounds(self.grid, self.prepared_bconds, self.model,
                                   self.mode)

    def default_loss(self, lambda_bound:  Union[int, float] = 10):
        """
        Computes l2 loss.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """
        op = self.operator.pde_compute()

        if self.prepared_bconds == None:
            return torch.sum(torch.mean((op) ** 2, 0))

        b_val, true_b_val = self.bconds.apply_bconds_operator()

        if self.mode == 'mat':
            loss = torch.mean((op) ** 2) + \
                   lambda_bound * torch.mean((b_val - true_b_val) ** 2)
        else:
            loss = torch.sum(torch.mean((op) ** 2, 0)) + \
                   lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))
        return loss

    def casual_loss(self, lambda_bound:  Union[int, float] = 10, tol: float = 0) -> torch.Tensor:
        """
        Computes casual loss, which is caclulated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            tol: float constant, influences on error penalty. 
        Returns:
            model loss.
        """
        
        op = self.operator.pde_compute()**2

        if self.prepared_bconds == None:
            return torch.sum(torch.mean((op) ** 2, 0))

        b_val, true_b_val = self.bconds.apply_bconds_operator()

        res = torch.sum(op, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        M = torch.triu(torch.ones((self.n_t, self.n_t)), diagonal=1).T
        with torch.no_grad():
            W = torch.exp(- tol * (M @ res))
        #plt.plot(W.detach().cpu().numpy())
        #plt.show()
        loss = torch.mean(W*res) + \
                lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))

        return loss

    def weak_loss(self, weak_form: Union[None, list], lambda_bound: Union[int, float] = 10) -> torch.Tensor:
        """
        Weak solution of O/PDE problem.
        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        op = self.operator.weak_pde_compute(weak_form)
        if self.prepared_bconds == None:
            return torch.sum(op)

        # we apply no  boundary conditions operators if they are all None

        b_val, true_b_val = self.bconds.apply_bconds_operator()

        loss = torch.sum(op) + \
               lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))

        return loss

    def loss_evaluation(self, lambda_bound: Union[int, float] = 10, weak_form: Union[None, list] = None, tol=0) -> Union[default_loss, weak_loss, casual_loss]:
        """
        Setting the required loss calculation method.
        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            weak_form: list of basis functions.
            tol: float constant, influences on error penalty. 
        Returns:
            A given calculation method.
        """
        if self.mode == 'mat' or self.mode == 'autograd':
            if self.prepared_bconds == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        if weak_form != None and weak_form != []:
            return self.weak_loss(weak_form, lambda_bound=lambda_bound)
        elif tol != 0:
            return self.casual_loss(lambda_bound=lambda_bound, tol=tol)
        else:
            return self.default_loss(lambda_bound=lambda_bound)
