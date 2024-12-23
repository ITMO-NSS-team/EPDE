
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
    """ Prepares lambdas for corresponding equation or bcond type.

    Args:
        val (_type_): operator tensor or bval tensor
        lambda_ (Union[int, list, torch.Tensor]): regularization parameters values

    Returns:
        torch.Tensor: torch.Tensor with lambda_ values,
        len(lambdas) = number of columns in val
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
        """ Adding 'var' to the 'operator' if it's absent or convert to
        list 'pow' and 'var' if it's int or float.

        Args:
            equation (dict): operator in input form.

        Returns:
            dict: equation with unified for solver parameters.
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
        """ Defines the closest boundary point to the grid.

        Args:
            grid (torch.Tensor): grid (domain discretization).
            target_point (float): boundary point.

        Returns:
            int: position of the boundary point on the grid.
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
        """ Converts points to double type.

        Args:
            bnd (Union[list, np.array]): array or list of arrays
                points that should be converted

        Returns:
            float: bnd with double type.
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
        """ Method for searching position bnd in grid.

        Args:
            grid (torch.Tensor): array of a n-D points.
            bnd (_type_): points that should be converted.

        Returns:
            list: list of positions bnd on grid.
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
        """ Returns the position of the boundary points on the grid.

        Args:
            grid (torch.Tensor): grid for coefficient in form of
            torch.Tensor mapping.
            bnd (torch.Tensor):boundary conditions.

        Returns:
            Union[list, int]: list of positions of the boundary points on the grid.
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
        """ Prepares equation, boundary conditions for *NN* mode.

        Args:
            grid (torch.Tensor): tensor of a n-D points.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions.
            h (float, optional): discretizing parameter in finite difference
            method(i.e., grid resolution for scheme). Defaults to 0.001.
            inner_order (str, optional): accuracy inner order for finite difference.
            Defaults to '1'.
            boundary_order (str, optional):accuracy boundary order for finite difference.
            Defaults to '2'.
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
        """ Function serves applying different schemes to a different point types
        for entire differentiation direction.

        Args:
            dif_direction (list): differentiation direction, (example:d2/dx2->[[0,0]])
            nvars (int): dimensionality of the problem.
            axes_scheme_type (str): 'central' or combination of 'f' and 'b'.

        Returns:
            list: list, where the conventional operator changed to
            steps and signs (see scheme_build function description).
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
        """ Method that converts integer finite difference steps in term described
        in Finite_diffs class to a grids with shifted points, i.e.
        from field (x,y) -> (x,y+h).

        Args:
            finite_diff_scheme (list): operator_to_type_op one term.
            grid_points (torch.Tensor): grid points that will be shifted
            corresponding to finite diff scheme.

        Returns:
            list: list, where the steps and signs changed to grid and signs.
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
        """ Checks the coefficient type

        Args:
            coeff (Union[int, float, torch.Tensor, callable]): coefficient
            in equation operator.
            grid_points (torch.Tensor): if coeff is callable or torch.Tensor

        Raises:
            NameError: coeff" should be: torch.Tensor or callable or int or float!

        Returns:
            torch.Tensor: coefficient
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
        """ Converts operator to a grid_shift form. Includes term coefficient
        conversion.
        Coeff may be integer, function or array, last two are mapped to a
        subgrid that corresponds point type.

        Args:
            fin_diff_op (list): operator_to_type_op result.
            grid_points (_type_): grid points that will be shifted
            corresponding to finite diff scheme.

        Returns:
            list: final form of differential operator used in the algorithm for
            single grid type.
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
        """ Method for operator preparing, there is construct all predefined
        methods.

        Args:
            operator (dict): operator in input form.
            grid_points (torch.Tensor): see type_op_to_grid_shift_op method.
            points_type (str): points type of grid_points.

        Returns:
            dict: prepared operator
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
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list: list of dictionaries, where every dictionary is the result of
            'one_operator_prepare'
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
        """ Method for applying boundary operator for all points type in bnd_dict.

        Args:
            bnd_operator (dict): boundary operator in input form.
            bnd_dict (dict): dictionary (keys is points type, values is boundary points).

        Returns:
            list: final form of differential operator used in the algorithm for
            subset grid types.
        """

        operator_list = []
        for points_type in list(bnd_dict.keys()):
            equation = self._one_operator_prepare(
                deepcopy(bnd_operator), bnd_dict[points_type], points_type)
            operator_list.append(equation)
        return operator_list

    def bnd_prepare(self) -> list:
        """ Method for boundary conditions preparing to final form.

        Returns:
            list: list of dictionaries where every dict is one boundary condition
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
        """ Prepares equation for autograd method
        (i.e., from conventional form to input form).

        Args:
            grid (torch.Tensor): tensor of a n-D points.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions in input form.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def _checking_coeff(self,
                       coeff: Union[int, float, torch.Tensor]) -> Union[int, float, torch.Tensor]:
        """ Checks the coefficient type

        Args:
            coeff (Union[int, float, torch.Tensor]): coefficient in equation operator.

        Raises:
            NameError: "coeff" should be: torch.Tensor or callable or int or float!

        Returns:
            Union[int, float, torch.Tensor]: coefficient
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
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Args:
            operator (dict): operator in input form.

        Returns:
            dict: dict, where coeff is checked.
        """

        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            term['coeff'] = self._checking_coeff(term['coeff'])
        return operator

    def operator_prepare(self) -> list:
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list: list of dictionaries, where every dictionary is the result of
            'one_operator_prepare'
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
        """ Method for boundary conditions preparing to final form

        Returns:
            list: list of dictionaries where every dict is one boundary condition
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
        """ Prepares equation for autograd method
        (i.e., from conventional form to input form).

        Args:
            grid (torch.Tensor): grid, result of meshgrid.
            operator (Union[list, dict]): operator in input form.
            bconds (list): boundary conditions in input form.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def operator_prepare(self) -> list:
        """ Method realizes operator preparing for 'mat' method
        using only 'equation_unify' method.
        Returns:
            list: final form of differential operator used in the algorithm.
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
        """ Define position of boundary points on the grid.

        Args:
            bnd (torch.Tensor): boundary subgrid.

        Returns:
            list: list of positions, where boundary points intersects on the grid.
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
        """ Method for boundary conditions preparing to final form.

        Returns:
            list: list of dictionaries where every dict is one boundary condition.
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
        """_summary_

        Args:
            grid (torch.Tensor): grid from cartesian_prod or meshgrid result.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions.
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 0.001.
            inner_order (str, optional): accuracy inner order for finite difference. Defaults to '1'.
            boundary_order (str, optional): accuracy boundary order for finite difference. Defaults to '2'.
        """

        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """ Setting the calculation method.

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
