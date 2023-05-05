import torch
import numpy as np
from copy import deepcopy
from typing import Union

from epde.solver.points_type import Points_type
from epde.solver.finite_diffs import Finite_diffs
from epde.solver.device import check_device


class Boundary():
    """
    Сlass for bringing all boundary conditions to a similar form.
    """
    def __init__(self, bconds: list):
        """
        Args:
            bconds: list with boundary conditions bconds = [bcond,bcond,..], where
                    'bcond' is list with parameters corresponding to boundary
                    condition.
        """
        self.bconds = bconds

    def dirichlet(self, bcond: list) -> list:
        """
        Boundary conditions without derivatives (bop is None), it can be
        in form: bcond = [bnd, bval, type], 'bnd' is boundary points, 'bval' is
        desired function values at 'bnd' points, 'type' should be 'dirichlet'.
        If task has several desired functions, bcond will be in form:
        bcond = [bnd, bval, var, type] where 'var' is function number.

        Args:
            bcond: list in input form: [bnd, bval, type] or [bnd, bval, var, type].

        Returns:
            boundary condition in unified form.
        """
        bcond[0] = check_device(bcond[0])
        bcond[1] = check_device(bcond[1])
        if len(bcond) == 3:
            boundary = [bcond[0], None, bcond[1], 0, bcond[2]]
        elif len(bcond) == 4:
            boundary = [bcond[0], None, bcond[1], bcond[2], bcond[3]]
        else:
            raise NameError('Incorrect Dirichlet condition')
        return boundary

    def neumann(self, bcond: list) -> list:
        """
        Boundary conditions with derivatives (bop is not None), it can be
        in form: bcond = [bnd, bop, bval, type], 'bnd' is boundary points,
        'bval' is desired function values at 'bnd' points, 'type' should be
        'dirichlet'. If task has several desired functions, bcond will be
        in form: bcond = [bnd, bop, bval, var, type] where 'var' is function
        number.

        Args:
            bcond: list in input form: [bnd, bop, bval, type] or
                                [bnd, bop, bval, var, type]
        Returns:
            boundary condition in unified form.
        """
        bcond[0] = check_device(bcond[0])
        bcond[2] = check_device(bcond[2])
        if len(bcond) == 4:
            bcond[1] = EquationMixin.equation_unify(bcond[1])
            boundary = [bcond[0], bcond[1], bcond[2], None, bcond[3]]
        elif len(bcond) == 5:
            bcond[1] = EquationMixin.equation_unify(bcond[1])
            boundary = [bcond[0], bcond[1], bcond[2], None, bcond[4]]
        else:
            raise NameError('Incorrect operator condition')
        return boundary

    def periodic(self, bcond: list) -> list:
        """
        Periodic can be: periodic dirichlet (example u(x,t)=u(-x,t))
        in form: bcond = [bnd, type], [bnd, var, type]
        or periodic operator (example du(x,t)/dx=du(-x,t)/dx)
        if from: [bnd, bop, type].
        Parameter 'bnd' is list: [b_coord1, b_coord2,..]

        Args:
            bcond: list in input form: [bnd, type] or [bnd, var, type] or
                                [bnd, bop, type]

        Returns:
            boundary condition in unfied form.
        """
        for i in range(len(bcond[0])):
            bcond[0][i] = check_device(bcond[0][i])
        if len(bcond) == 2:
            b_val = torch.zeros(bcond[0][0].shape[0])
            boundary = [bcond[0], None, b_val, 0, bcond[1]]
        elif len(bcond) == 3 and type(bcond[1]) is int:
            b_val = torch.zeros(bcond[0][0].shape[0])
            boundary = [bcond[0], None, b_val, bcond[1], bcond[2]]
        elif type(bcond[1]) is dict:
            b_val = torch.zeros(bcond[0][0].shape[0])
            bcond[1] = EquationMixin.equation_unify(bcond[1])
            boundary = [bcond[0], bcond[1], b_val, None, bcond[2]]
        else:
            raise NameError('Incorrect periodic condition')
        return boundary

    def bnd_choose(self, bcond: list) -> list:
        """
        Method that choose type of boundary condition.

        Args:
            bcond: list with boundary condition parameters.

        Returns:
            return unified condition.

        """
        if bcond[-1] == 'periodic':
            bnd = self.periodic(bcond)
        elif bcond[-1] == 'dirichlet':
            bnd = self.dirichlet(bcond)
        elif bcond[-1] == 'operator':
            bnd = self.neumann(bcond)
        else:
            raise NameError('TEDEouS can not use ' + bcond[-1] + ' condition type')
        return bnd

    def bnd_unify(self) -> list:
        """
        Method that convert result of 'bnd_choose' to dict with correspondung
        keys = ('bnd', 'bop', 'bval', 'var', 'type').

        Returns:
            unified boundary conditions in dict form.
        """
        unified_bnd = []
        for bcond in self.bconds:
            bnd = {}
            bnd['bnd'], bnd['bop'], bnd['bval'], bnd['var'], \
            bnd['type'] = self.bnd_choose(bcond)
            unified_bnd.append(bnd)
        return unified_bnd


class EquationMixin():
    """
    Auxiliary class. This one contains some methods that uses in other classes.
    """

    @staticmethod
    def equation_unify(equation: dict) -> dict:
        """
        Adding 'var' to the 'operator' if it's absent or convert to
        list 'pow' and 'var' if it's int or float.

        Args:
            operator: operator in input form.

        Returns:
            equation: equation with unified for solver parameters.
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
                elif type(operator['pow']) is list:
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
        Defines the closest boundary point to the grid.
        Args:

            target_point: boundary point.
        Returns:
            position of the boundary point on the grid.
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
        Coverts points to double type.

        Args:
            bnd: array or list of arrays
                points that should be converted
        Returns:
            bnd with double type.
        """

        if type(bnd) == list:
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.convert_to_double(cur_bnd)
            return bnd
        elif type(bnd) == np.array:
            return torch.from_numpy(bnd).double()
        return bnd.double()

    @staticmethod
    def search_pos(grid: torch.Tensor, bnd) -> list:
        """
        Method for searching position bnd in grid.

        Args:
            grid: array of a n-D points.
            bnd: points that should be converted.
        Returns:
            list of positions bnd on grid.
        """

        if type(bnd) == list:
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
        Returns the position of the boundary points on the grid.
        Args:
            grid:  grid for coefficient in form of torch.Tensor mapping.
            bnd: boundary conditions.
        Returns:
            list of positions of the boundary points on the grid.
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

    def __init__(self, grid: torch.Tensor, operator:  Union[dict, list], bconds, h: float = 0.001,
                 inner_order: str = '1', boundary_order: str = '2'):
        """
        Prepares equation, boundary conditions for NN method.
        Args:
            grid:  array of a n-D points.
            operator:  equation.
            bconds: boundary conditions.
            h: discretizing parameter in finite difference method (i.e., grid resolution for scheme).
            inner_order: accuracy inner order for finite difference. Default = 1
            boundary_order: accuracy boundary order for finite difference. Default = 2
        """
        super().__init__(grid)
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def operator_to_type_op(self, dif_direction: list, nvars: int, axes_scheme_type: str) -> list:
        """
        Function serves applying different schemes to a different point types
        for entire differentiation direction.

        Args:
            dif_direction: differentiation direction, (example:d2/dx2->[[0,0]])
            nvars: dimensionality of the problem.
            axes_scheme_type: 'central' or combination of 'f' and 'b'.

        Returns:
            list, where the conventional operator changed to steps and signs (see scheme_build function description).
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

    def finite_diff_scheme_to_grid_list(self, finite_diff_scheme: list, grid_points: torch.Tensor) -> list:
        """
        Method that converts integer finite difference steps in term described
        in Finite_diffs class to a grids with shifted points, i.e.
        from field (x,y) -> (x,y+h).

        Args:
            finite_diff_scheme: operator_to_type_op one term
            grid_points: grid points that will be shifted corresponding to finite diff
                         scheme
        Returns:
            list, where the steps and signs changed to grid and signs.
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

    def checking_coeff(self, coeff: Union[int, float, torch.Tensor], grid_points: torch.Tensor):
        """
        Checks the coefficient type

        Args:
            coeff: coefficient in equation operator.
            grid_points: if coeff is callable or torch.Tensor

        Returns:
            coefficient
        """

        if type(coeff) == int or type(coeff) == float:
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = (coeff, grid_points)
        elif type(coeff) == torch.Tensor:
            coeff = check_device(coeff)
            pos = self.bndpos(self.grid, grid_points)
            coeff1 = coeff[pos].reshape(-1, 1)
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def type_op_to_grid_shift_op(self, fin_diff_op: list, grid_points) -> list:
        """
        Converts operator to a grid_shift form. Includes term coefficient
        conversion.
        Coeff may be integer, function or array, last two are mapped to a
        subgrid that corresponds point type.

        Args:
            fin_diff_op: operator_to_type_op result.
            grid_points: grid points that will be shifted corresponding to finite diff scheme.

        Returns:
            shift_grid_op: final form of differential operator used in the algorithm for
                           single grid type.
        """
        shift_grid_op = []
        for term1 in fin_diff_op:
            grid_op = self.finite_diff_scheme_to_grid_list(term1, grid_points)
            shift_grid_op.append(grid_op)
        return shift_grid_op

    def one_operator_prepare(self, operator: dict, grid_points: torch.Tensor, points_type: str) -> dict:
        """
        Method for operator preparing, there is construct all predefined
        methods.

        Args:
            operator: operator in input form
            grid_points: see type_op_to_grid_shift_op method
            points_type: points type of grid_points

        Returns:
            prepared operator
        """

        nvars = self.grid.shape[-1]
        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            dif_term = list(term.keys())[1]
            term['coeff'] = self.checking_coeff(term['coeff'], grid_points)
            term[dif_term] = self.operator_to_type_op(term[dif_term],
                                                      nvars, points_type)
            term[dif_term][0] = self.type_op_to_grid_shift_op(
                term[dif_term][0], grid_points)
        return operator

    def operator_prepare(self) -> list:
        """
        Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list of dictionaries, where every dictionary is the result of
                'one_operator_prepare'
        """

        grid_points = self.grid_sort()['central']
        if type(self.operator) is list and type(self.operator[0]) is dict:
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.one_operator_prepare(
                    self.operator[i], grid_points, 'central')
                prepared_operator.append(equation)
        else:
            equation = self.one_operator_prepare(
                self.operator, grid_points, 'central')
            prepared_operator = [equation]

        return prepared_operator

    def apply_bnd_operators(self, bnd_operator: dict, bnd_dict: dict) -> list:
        """
        Method for applying boundary operator for all points type in bnd_dict.

        Args:
            bnd_operator: boundary operator in input form.
            bnd_dict: dictionary (keys is points type, values is boundary points).

        Returns:
            final form of differential operator used in the algorithm for
                subset grid types.

        """

        operator_list = []
        for points_type in list(bnd_dict.keys()):
            equation = self.one_operator_prepare(
                deepcopy(bnd_operator), bnd_dict[points_type], points_type)
            operator_list.append(equation)
        return operator_list

    def bnd_prepare(self) -> list:
        """
        Method for boundary conditions preparing to final form.

        Returns:
            list of dictionaries where every dict is one boundary condition
        """

        grid_dict = self.grid_sort()
        bconds1 = Boundary(self.bconds).bnd_unify()
        if bconds1 == None:
            return None
        for bcond in bconds1:
            bnd_dict = self.bnd_sort(grid_dict, bcond['bnd'])
            if bcond['bop'] != None:
                if bcond['type'] == 'periodic':
                    bcond['bop'] = [self.apply_bnd_operators(
                        bcond['bop'], i) for i in bnd_dict]
                else:
                    bcond['bop'] = self.apply_bnd_operators(
                        bcond['bop'], bnd_dict)
        return bconds1


class Equation_autograd(EquationMixin):
    """
    Prepares equation for autograd method (i.e., from conventional form to input form).
    """

    def __init__(self, grid: torch.Tensor, operator, bconds):
        """
        Prepares equation for autograd method (i.e., from conventional form to input form).

        Args:
            grid: array of a n-D points.
            operator:  equation.
            bconds: boundary conditions.
        """
        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def checking_coeff(self, coeff: Union[int, float, torch.Tensor]) -> Union[int, float, torch.Tensor]:
        """
        Checks the coefficient type

        Args:
            coeff: coefficient in equation operator.
        Returns:
            coefficient
        """

        if type(coeff) == int or type(coeff) == float:
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = coeff
        elif type(coeff) == torch.Tensor:
            coeff = check_device(coeff)
            coeff1 = coeff.reshape(-1, 1)
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def one_operator_prepare(self, operator: dict) -> dict:
        """
        Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list of dictionaries, where every dictionary is the result of
                'one_operator_prepare'
        """

        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            term['coeff'] = self.checking_coeff(term['coeff'])
        return operator

    def operator_prepare(self) -> list:
        """
        Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list of dictionaries, where every dictionary is the result of
                'one_operator_prepare'

        """

        if type(self.operator) is list and type(self.operator[0]) is dict:
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(self.one_operator_prepare(equation))
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [self.one_operator_prepare(equation)]

        return prepared_operator

    def bnd_prepare(self):
        """
        Method for boundary conditions preparing to final form
        Returns
        -------
        prepared_bnd : list
            list of dictionaries where every dict is one boundary condition
        """
        bconds = Boundary(self.bconds).bnd_unify()
        if bconds == None:
            return None

        for bcond in bconds:
            if bcond['bop'] != None:
                bcond['bop'] = self.equation_unify(bcond['bop'])

        return bconds


class Equation_mat(EquationMixin):
    """
    Class realizes input data preprocessing (operator and boundary conditions
    preparing) for 'mat' method.
    """

    def __init__(self, grid, operator, bconds):
        """
        Prepares equation for autograd method (i.e., from conventional form to input form).

        Args:
            grid: array of a n-D points.
            operator:  equation.
            bconds: boundary conditions.
        """
        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def operator_prepare(self) -> list:
        """
        Method realizes operator preparing for 'mat' method
        using only 'equation_unify' method.

        Returns:
            final form of differential operator used in the algorithm.
        """

        unified_operator = [self.equation_unify(self.operator)]
        return unified_operator

    def point_position(self, bnd) -> list:
        """
        Define position of boundary points on the grid.

        Args:
            bnd:

        Returns:
            list of positions, where boundary points intersects on the grid.
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
        Method for boundary conditions preparing to final form.

        Returns:
            list of dictionaries where every dict is one boundary condition
        """
        bconds = Boundary(self.bconds).bnd_unify()
        for bcond in bconds:
            if bcond['type'] == 'periodic':
                bpos = []
                for bnd in bcond['bnd']:
                    bpos.append(self.point_position(bnd))
            else:
                bpos = self.point_position(bcond['bnd'])
            if bcond['bop'] != None:
                bcond['bop'] = self.equation_unify(bcond['bop'])
            bcond['bnd'] = bpos
        return bconds


class Equation():
    """
    Interface for preparing equations due to chosen calculation method.
    """
    def __init__(self, grid: torch.Tensor, operator: Union[dict, list], bconds: list, h: float = 0.001,
                 inner_order: str ='1', boundary_order: str ='2'):
        """
        Args:
            grid: array of a n-D points.
            operator: equation.
            bconds: boundary conditions.
            h: discretizing parameter in finite difference method (i.e., grid resolution for scheme).
            inner_order: accuracy inner order for finite difference. Default = 1
            boundary_order:  accuracy boundary order for finite difference. Default = 2
        """
        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """
        Setting the calculation method.
        Args:
            strategy: Calculation method. (i.e., "NN", "autograd", "mat").
        Returns:
            A given calculation method.
        """

        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h,
                               inner_order=self.inner_order,
                               boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)
