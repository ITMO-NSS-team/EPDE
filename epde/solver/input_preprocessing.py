
import torch
import numpy as np


from epde.solver.points_type import Points_type
from epde.solver.finite_diffs import Finite_diffs



class EquationMixin():
    @staticmethod   
    def operator_unify(operator):
            """
            I just was annoyed adding additional square brackets to the operators.
            This one allows to make operator form simpler.

            Parameters
            ----------
            operator : list
                Operator in form ... .

            Returns
            -------
            unified_operator : list
                DESCRIPTION.

            """
            unified_operator = []
            for term in operator:
                const = term[0]
                vars_set = term[1]
                power = term[2]
                variables=[]
                if len(term)==4:
                    variables = term[3]
                elif isinstance(power,(int,float)):
                    variables=0
                elif type(power)==list:
                    for _ in range(len(power)):
                        variables.append(0)
                if variables is None:
                    if type(power) is list:
                        unified_operator.append([const, vars_set, power,0])
                    else:
                        unified_operator.append([const, [vars_set], [power],[0]])
                else:
                    if type(power) is list:
                        unified_operator.append([const, vars_set, power,variables])
                    else:
                        unified_operator.append([const, [vars_set], [power],[variables]])
                # if type(power) is list:
                #         unified_operator.append([const, vars_set, power])
                # else:
                #         unified_operator.append([const, [vars_set], [power]])
            return unified_operator

    @staticmethod
    def op_dict_to_list(opdict):
        return list([list(term.values()) for term in opdict.values()])

    @staticmethod
    def closest_point(grid,target_point):
        min_dist=np.inf
        pos=0
        min_pos=0
        for point in grid:
            dist=torch.linalg.norm(point-target_point)
            if dist<min_dist:
                min_dist=dist
                min_pos=pos
            pos+=1
        return min_pos

    @staticmethod
    def bndpos(grid, bnd):
        """
        
        Returns the position of the boundary points on the grid
        
        Parameters
        ----------
        grid : torch.Tensor
            grid for coefficient in form of torch.Tensor mapping
        bnd : torch.Tensor
            boundary
        Returns
        -------
        bndposlist : list (int)
            positions of boundaty points in grid
        """
        if grid.shape[0] == 1:
            grid=grid.reshape(-1,1)
        grid = grid.double()

        def convert_to_double(bnd):
            if type(bnd) == list:
                for i, cur_bnd in enumerate(bnd):
                    bnd[i] = convert_to_double(cur_bnd)
                return bnd
            elif type(bnd) == np.array:
                return torch.from_numpy(bnd).double()
            return bnd.double()

        bnd = convert_to_double(bnd)

        def search_pos(bnd):
            if type(bnd) == list:
                for i, cur_bnd in enumerate(bnd):
                    bnd[i] = search_pos(cur_bnd)
                return bnd
            pos_list = []
            for point in bnd:
                try:
                    pos = int(torch.where(torch.all(torch.isclose(grid, point), dim=1))[0])
                except Exception:
                    pos = EquationMixin.closest_point(grid, point)
                pos_list.append(pos)
            return pos_list

        bndposlist = search_pos(bnd)

        return bndposlist

    @staticmethod
    def bnd_unify(bconds):
        """
        Serves to add None instead of empty operator

        Parameters
        ----------
        bconds : list
            
            boundary in conventional form (see examples)

        Returns
        -------
        unified_bconds : list
            
            boundary in input-friendly form

        """
        if bconds==None:
            return None
        unified_bconds = []
        for bcond in bconds:
            if len(bcond) == 2:
                if type(bcond[1]) is str:
                    unified_bconds.append([bcond[0], None, torch.from_numpy(np.zeros(bcond[0][0].shape[0])), 0, bcond[1]])
                else:
                    unified_bconds.append([bcond[0], None, bcond[1], 0, 'boundary values'])

            elif len(bcond) == 3:
                if type(bcond[2]) is str:
                    if type(bcond[1]) is int:
                        unified_bconds.append([bcond[0], None, torch.from_numpy(np.zeros(bcond[0][0].shape[0])), bcond[1], bcond[2]])

                    elif len(EquationMixin.op_dict_to_list(bcond[1])[0]) == 3:
                        unified_bconds.append([bcond[0], bcond[1], torch.from_numpy(np.zeros(bcond[0][0].shape[0])), 0, bcond[2]])
                    
                    else:
                        var = EquationMixin.op_dict_to_list(bcond[1])[0][-1]
                        unified_bconds.append([bcond[0], bcond[1], torch.from_numpy(np.zeros(bcond[0][0].shape[0])), var, bcond[2]])
                        
                elif type(bcond[2]) is int:
                    unified_bconds.append([bcond[0], None, bcond[1], bcond[2], 'boundary values'])

                elif len(EquationMixin.op_dict_to_list(bcond[1])[0]) == 4:
                    var = EquationMixin.op_dict_to_list(bcond[1])[0][-1]
                    unified_bconds.append([bcond[0], bcond[1], bcond[2], var, 'boundary values'])

                else:
                    unified_bconds.append([bcond[0], bcond[1], bcond[2], 0, 'boundary values'])                

        return unified_bconds
  

class EquationInt(): 
    def operator_prepare(self, value): 
        raise NotImplementedError
    def bnd_prepare(self, value):
        raise NotImplementedError


class Equation_NN(EquationMixin, Points_type, Finite_diffs):
    def __init__(self, grid, operator, bconds, h=0.001, inner_order=1, boundary_order=2):
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order
        
    def operator_to_type_op(self, unified_operator, nvars, axes_scheme_type):
        """
        Function serves applying different schemes to a different point types for
        entire operator

        Parameters
        ----------
        unified_operator : list
            operator in a proper form
        nvars : int
            Dimensionality of the problem.
        axes_scheme_type : string
            'central' or combination of 'f' and 'b'.
        h : float, optional
            Derivative precision parameter (to simplify h in the expression
                                            (f(x+h)-f(x))/h). The default is 1/2.
        boundary_order : int, optional
            Order of finite difference scheme taken at the domain boundary points. 
            The default is 1.

        Returns
        -------
        fin_diff_op : list
            list, where the conventional operator changed to steps and signs
            (see scheme_build function description).

        """
        fin_diff_op = []
        for term in unified_operator:
            fin_diff_list = []
            s_order_list = []
            const = term[0]
            vars_set = term[1]
            power = term[2]
            variables = term[3]
            for k, term in enumerate(vars_set):
                # None is for case where we need the fuction without derivatives
                if term != [None]:
                    if axes_scheme_type == 'central':
                        if self.inner_order==1:
                            scheme, direction_list = self.scheme_build(term, nvars, 'central')
                            s_order = self.sign_order(len(term), 'central', h=self.h)
                        elif self.inner_order==2:
                            scheme, direction_list = self.second_order_scheme_build(term, nvars, 'central')
                            s_order = self.second_order_sign_order(len(term), 'central', h=self.h)
                    else:
                        if self.boundary_order == 1:
                            scheme, direction_list = self.scheme_build(term, nvars, axes_scheme_type)
                            s_order = self.sign_order(len(term), direction_list, h=self.h)
                        elif self.boundary_order == 2:
                            scheme, direction_list = self.second_order_scheme_build(term, nvars, axes_scheme_type)
                            s_order = self.second_order_sign_order(len(term), direction_list, h=self.h)
                else:
                    scheme = [None]
                    s_order = [1]
                fin_diff_list.append(scheme)
                s_order_list.append(s_order)
            fin_diff_op.append([const, fin_diff_list, s_order_list, power,variables])
        return fin_diff_op

    def finite_diff_scheme_to_grid_list(self, finite_diff_scheme, grid_points):
        """
        Axiluary function that converts integer grid steps in term described in
        finite-difference scheme to a grids with shifted points, i.e.
        from field (x,y) -> (x,y+h).

        Parameters
        ----------
        finite_diff_scheme : list
            operator_to_type_op one term
        grid : torch.Tensor
        h : float
            derivative precision parameter. The default is 0.001.

        Returns
        -------
        s_grid_list : list
            list, where the the steps and signs changed to grid and signs
        """
        s_grid_list = []
        for i, shifts in enumerate(finite_diff_scheme):
            s_grid = grid_points
            for j, axis in enumerate(shifts):
                if axis != 0:
                    s_grid = self.shift_points(s_grid, j, axis * self.h)
            s_grid_list.append(s_grid)
        return s_grid_list

    def type_op_to_grid_shift_op(self, fin_diff_op, grid_points):
        """
        Converts operator to a grid_shift form. Includes term coefficient
        conversion.
        Coeff may be integer, function or array, last two are mapped to a 
        subgrid that corresponds point type

        Parameters
        ----------
        fin_diff_op : list
            operator_to_type_op result.
        grid : torch.Tensor
            grid with sotred nodes (see grid_prepare)
        h : float
            derivative precision parameter. The default is 0.001.
        true_grid : TYPE, optional
            initial grid for coefficient in form of torch.Tensor mapping

        Returns
        -------
        shift_grid_op : list
            final form of differential operator used in the algorithm for single 
            grid type
        """
        shift_grid_op = []
        for term1 in fin_diff_op:
            shift_grid_list = []
            coeff1 = term1[0]
            if type(coeff1) == int or type(coeff1) == float:
                coeff = coeff1
            elif callable(coeff1):
                coeff = coeff1(grid_points)
                coeff = coeff.reshape(-1, 1)
            elif type(coeff1) == torch.Tensor:
                pos = self.bndpos(self.grid, grid_points)

                coeff = coeff1[pos].reshape(-1, 1)
            
            finite_diff_scheme = term1[1]
            s_order = term1[2]
            power = term1[3]
            variables = term1[4]
            for k, term in enumerate(finite_diff_scheme):
                if term != [None]:
                    grid_op = self.finite_diff_scheme_to_grid_list(term, grid_points)
                else:
                    grid_op = [grid_points]
                shift_grid_list.append(grid_op)
            shift_grid_op.append([coeff, shift_grid_list, s_order, power,variables])
        return shift_grid_op


    def apply_all_operators(self, unified_operator, grid_dict1):
        """
        Parameters
        ----------
        operator : list
            operator_unify result.
        grid : torch.Tensor
            grid with sotred nodes (see grid_prepare)
        h : float
            derivative precision parameter. The default is 0.001.
        subset : list, optional
            grid subsets used for the operator ,e.g. ['central','fb','ff']
        true_grid : TYPE, optional
            initial grid for coefficient in form of torch.Tensor mapping

        Returns
        -------
        operator_list :  list
            final form of differential operator used in the algorithm for subset 
            grid types

        """
        operator_list = []
        nvars =list(grid_dict1.values())[0].shape[-1]
        for operator_type in list(grid_dict1.keys()):
            b = self.operator_to_type_op(unified_operator, nvars, operator_type)
            c = self.type_op_to_grid_shift_op(b, grid_dict1[operator_type])
            operator_list.append(c)
        return operator_list


    def operator_prepare(self):
        """
        Changes the operator in conventional form to the input one
        
        Parameters
        ----------
        op : list
            operator in conventional form.
        grid : torch.Tensor
            grid with sotred nodes (see grid_prepare)
        h : float
            derivative precision parameter. The default is 0.001.
        subset : list, optional
            grid subsets used for the operator ,e.g. ['central','fb','ff']
        true_grid : torch.Tensor, optional
            initial grid for coefficient in form of torch.Tensor mapping
        Returns
        -------
        operator_list :  list
            final form of differential operator used in the algorithm for subset 
            grid types

        """
        grid_dict = self.grid_sort(self.grid)
        nvars = self.grid.shape[-1]
        if type(self.operator) is list and type(self.operator[0]) is dict:
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                op0 = self.op_dict_to_list(self.operator[i])
                op1 = self.operator_unify(op0)
                op2 = self.operator_to_type_op(op1, nvars, axes_scheme_type='central')
                prepared_operator.append(self.type_op_to_grid_shift_op(op2, grid_dict['central']))
        else:
            if type(self.operator) == dict:
                op = self.op_dict_to_list(self.operator)
            else:
                op = self.operator
            op1 = self.operator_unify(op)
            op2 = self.operator_to_type_op(op1, nvars, axes_scheme_type='central')
            prepared_operator = [self.type_op_to_grid_shift_op(op2, grid_dict['central'])]

        return prepared_operator

    def bnd_prepare(self):
        """
        Parameters
        ----------
        bconds : list
            boundary in conventional form (see examples)
        grid : torch.Tensor
            grid with sotred nodes (see grid_prepare)
        h : float
            derivative precision parameter. The default is 0.001.

        Returns
        -------
        prepared_bnd : list
            
            boundary in input form

        """
        grid_dict = self.grid_sort(self.grid)
        sorted_grid = torch.cat(list(grid_dict.values()))
        bconds1 = self.bnd_unify(self.bconds)
        if bconds1==None:
            return None
        prepared_bnd = []
        for bcond in bconds1:
            b_coord = bcond[0]
            bop = bcond[1]
            bval = bcond[2]
            bvar = bcond[3]
            btype = bcond[4]
            bpos = self.bndpos(sorted_grid, b_coord)
            if bop == [[1, [None], 1]]:
                bop = None
            if bop != None:
                if type(bop)==dict:
                    bop=self.op_dict_to_list(bop)
                bop1 = self.operator_unify(bop)
                bop2 = self.apply_all_operators(bop1, grid_dict)
            else:
                bop2 = None
            prepared_bnd.append([bpos, bop2, bval, bvar, btype])

        return prepared_bnd


class Equation_autograd(EquationMixin):
    def __init__(self, grid, operator, bconds):
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
    
    @staticmethod
    def expand_coeffs_autograd(unified_operator, grid):
        autograd_op=[]
        for term in unified_operator:
            coeff1 = term[0]
            if type(coeff1) == int or type(coeff1) == float:
                coeff = coeff1
            elif callable(coeff1):
                coeff = coeff1(grid)
                coeff = coeff.reshape(-1,1)
            elif type(coeff1) == torch.Tensor:
                coeff = coeff1.reshape(-1,1)
            
            prod = term[1]
            power = term[2]
            variables = term[3]
            autograd_op.append([coeff, prod, power, variables])
        return autograd_op

    def operator_prepare(self):
        """
        Changes the operator in conventional form to the input one
        
        Parameters
        ----------
        op : list
            operator in conventional form.
        Returns
        -------
        operator_list :  list
            final form of differential operator used in the algorithm 

        """
        if type(self.operator) is list and type(self.operator[0]) is dict:
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                op0 = self.op_dict_to_list(self.operator[i])
                unified_operator = self.operator_unify(op0)
                prepared_operator.append(self.expand_coeffs_autograd(unified_operator,self.grid))
        else:
            if type(self.operator) == dict:
                op = self.op_dict_to_list(self.operator)
            unified_operator = self.operator_unify(op)

            prepared_operator = [self.expand_coeffs_autograd(unified_operator, self.grid)]
        
        return prepared_operator

    def bnd_prepare(self):
        """
        Parameters
        ----------
        bconds : list
            boundary in conventional form (see examples)
        grid : torch.Tensor
            grid with sotred nodes (see grid_prepare)
        h : float
            derivative precision parameter. The default is 0.001.

        Returns
        -------
        prepared_bnd : list
            
            boundary in input form

        """
        bconds = self.bnd_unify(self.bconds)
        if bconds==None:
            return None
        prepared_bnd = []
        for bcond in bconds:
            b_coord = bcond[0]
            bop = bcond[1]
            bval = bcond[2]
            var = bcond[3]
            btype = bcond[4]
            bpos = self.bndpos(self.grid, b_coord)
            if bop == [[1, [None], 1]]:
                bop = None
            if bop != None:
                if type(bop)==dict:
                    bop=self.op_dict_to_list(bop)
                bop1 = self.operator_unify(bop)
            else:
                bop1 = None
            prepared_bnd.append([bpos, bop1, bval, var, btype])
        return prepared_bnd


class Equation_mat(EquationMixin):
    def __init__(self, grid, operator, bconds):
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
    
    def operator_prepare(self):
        if type(self.operator) == dict:
            operator_list = self.op_dict_to_list(self.operator)
        unified_operator = self.operator_unify(operator_list)
        return [unified_operator]

    def bnd_prepare(self):
        prepared_bconds=[]
        bconds = self.bnd_unify(self.bconds)
        for bnd in bconds:
            bpts=bnd[0]
            bop = bnd[1]
            bval=bnd[2]
            var = bnd[3]
            btype = bnd[4]
            bpos=[]
            # bpos=bndpos(grid,bpts)
            for pt in bpts:
                if self.grid.shape[0]==1:
                    point_pos=(torch.tensor(self.bndpos(self.grid,pt)),)
                else:
                    prod=(torch.zeros_like(self.grid[0])+1).bool()
                    for axis in range(self.grid.shape[0]):
                        axis_intersect=torch.isclose(pt[axis].float(),self.grid[axis].float())
                        prod*=axis_intersect
                    point_pos=torch.where(prod==True)
                bpos.append(point_pos)
            if type(bop)==dict:
                bop=self.op_dict_to_list(bop)
            if bop!= None:
                bop = self.operator_unify(bop)
            prepared_bconds.append([bpos, bop, bval, var, btype])
        return prepared_bconds


class Equation():
    def __init__(self, grid, operator, bconds, h=0.001, inner_order=1, boundary_order=2):
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order
    def set_strategy(self, strategy):
        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h, inner_order=self.inner_order, boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)
