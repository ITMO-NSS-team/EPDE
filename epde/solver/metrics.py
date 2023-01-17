import torch
import numpy as np

from epde.solver.points_type import Points_type
flatten_list = lambda t: [item for sublist in t for item in sublist]

class DerivativeInt():
    def take_derivative(self, value): 
        raise NotImplementedError

class Derivative_NN(DerivativeInt):
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model
    
    def take_derivative (self, term):
        """
        Axiluary function serves for single differential operator resulting field
        derivation

        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        term : TYPE
            differential operator in conventional form.

        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """
        # it is may be int, function of grid or torch.Tensor
        coeff = term[0]
        # this one contains shifted grids (see input_preprocessing module)
        shift_grid_list = term[1]
        # signs corresponding to a grid
        s_order_norm_list = term[2]
        # float that represents power of the differential term
        power = term[3]
        # number of variables in equation
        variables = term[4]
        # initially it is an ones field
        der_term = (torch.zeros_like(self.model(shift_grid_list[0][0])[0:,0]) + 1).reshape(-1,1)
        
        for j, scheme in enumerate(shift_grid_list):
            # every shift in grid we should add with correspoiding sign, so we start
            # from zeros
            grid_sum = torch.zeros_like(self.model(scheme[0]))[0:,0].reshape(-1,1) #почему схема от 0?
            for k, grid in enumerate(scheme):
                # and add grid sequentially
                grid_sum += (self.model(grid)[0:,variables[j]]).reshape(-1,1) * s_order_norm_list[j][k]
                # Here we want to apply differential operators for every term in the product
            der_term = der_term * grid_sum ** power[j]
        der_term = coeff * der_term

            
        return der_term

class Derivative_autograd(DerivativeInt):
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model
    
    @staticmethod
    def nn_autograd_simple(model, points, order,axis=0):
        points.requires_grad=True
        gradient_full = []
        f = model(points).sum(0)
        for i in range(len(f)):
            fi = f[i]
            for j in range(order):
                grads, = torch.autograd.grad(fi, points, create_graph=True)
                fi = grads[:,axis].sum()
            gradient_full.append(grads[:,axis].reshape(-1,1))
        gradient_full = torch.hstack(gradient_full)
        return gradient_full

    @staticmethod
    def nn_autograd_mixed(model, points,axis=[0]):
        points.requires_grad=True
        gradient_full = []
        f = model(points).sum(0)
        for i in range(len(f)):
            fi = f[i]
            for ax in axis:
                grads, = torch.autograd.grad(fi, points, create_graph=True)
                fi = grads[:,ax].sum()
            gradient_full.append(grads[:,axis[-1]].reshape(-1,1))
        gradient_full = torch.hstack(gradient_full)
        return gradient_full


    def nn_autograd(self, *args, axis=0):
        model=args[0]
        points=args[1]
        if len(args)==3:
            order=args[2]
            grads=self.nn_autograd_simple(model, points, order,axis=axis)
        else:
            grads=self.nn_autograd_mixed(model, points,axis=axis)
        return grads


    def take_derivative(self, term):
        """
        Axiluary function serves for single differential operator resulting field
        derivation

        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        term : TYPE
            differential operator in conventional form.

        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """
        # it is may be int, function of grid or torch.Tensor
        coeff = term[0]
        # this one contains shifted grids (see input_preprocessing module)
        product = term[1]
        # float that represents power of the differential term
        power = term[2]
        # list that represent using variables
        variables = term[3]
        # initially it is an ones field
        der_term = (torch.zeros_like(self.model(self.grid))[0:, 0] + 1).reshape(-1, 1)
        for j, derivative in enumerate(product):
            if derivative == [None]:
                der = self.model(self.grid)[:, variables[j]].reshape(-1, 1)
            else:
                der = self.nn_autograd(self.model, self.grid, axis=derivative)[0:, variables[j]].reshape(-1, 1)

            der_term = der_term * der ** power[j]

        der_term = coeff * der_term

        return der_term

class Derivative_mat(DerivativeInt):
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model
    
    @staticmethod
    def derivative_1d(model,grid):
        # print('1d>2d')
        u=model.reshape(-1)
        x=grid.reshape(-1)
        
        # du_forward = (u-torch.roll(u, -1)) / (x-torch.roll(x, -1))
        
        # du_backward = (torch.roll(u, 1) - u) / (torch.roll(x, 1) - x)
        du =  (torch.roll(u, 1) - torch.roll(u, -1))/(torch.roll(x, 1)-torch.roll(x, -1))
        du[0] = (u[0]-u[1])/(x[0]-x[1])
        du[-1] = (u[-1]-u[-2])/(x[-1]-x[-2])
        
        du=du.reshape(model.shape)
        
        return du

    @staticmethod
    def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):
        #print('shape=',u_tensor.shape)
        if (u_tensor.shape[0]==1):
            du = Derivative_mat.derivative_1d(u_tensor,h_tensor)
            return du

        u_tensor = torch.transpose(u_tensor, 0, axis)
        h_tensor = torch.transpose(h_tensor, 0, axis)
        
        
        if scheme_order==1:
            du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                        (-torch.roll(h_tensor, -1) + h_tensor)
        
            du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                        (torch.roll(h_tensor, 1) - h_tensor)
            du = (1 / 2) * (du_forward + du_backward)
        
        # dh=h_tensor[0,1]-h_tensor[0,0]
        
        if scheme_order==2:
            u_shift_down_1 = torch.roll(u_tensor, 1)
            u_shift_down_2 = torch.roll(u_tensor, 2)
            u_shift_up_1 = torch.roll(u_tensor, -1)
            u_shift_up_2 = torch.roll(u_tensor, -2)
            
            h_shift_down_1 = torch.roll(h_tensor, 1)
            h_shift_down_2 = torch.roll(h_tensor, 2)
            h_shift_up_1 = torch.roll(h_tensor, -1)
            h_shift_up_2 = torch.roll(h_tensor, -2)
            
            h1_up=h_shift_up_1-h_tensor
            h2_up=h_shift_up_2-h_shift_up_1
            
            h1_down=h_tensor-h_shift_down_1
            h2_down=h_shift_down_1-h_shift_down_2
        
            a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
            b_up=(h2_up+h1_up)/(h1_up*h2_up)
            c_up=-h1_up/(h2_up*(h1_up+h2_up))
            
            a_down=(2*h1_down+h2_down)/(h1_down*(h1_down+h2_down))
            b_down=-(h2_down+h1_down)/(h1_down*h2_down)
            c_down=h1_down/(h2_down*(h1_down+h2_down))
            
            du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
            du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
            du = (1 / 2) * (du_forward + du_backward)
            
            
        if boundary_order==1:
            if scheme_order==1:
                du[:, 0] = du_forward[:, 0]
                du[:, -1] = du_backward[:, -1]
            elif scheme_order==2:
                du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                            (-torch.roll(h_tensor, -1) + h_tensor)
            
                du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                            (torch.roll(h_tensor, 1) - h_tensor)
                du[:, 0] = du_forward[:, 0]
                du[:, 1] = du_forward[:, 1]
                du[:, -1] = du_backward[:, -1]
                du[:, -2] = du_backward[:, -2]
        elif boundary_order==2:
            if scheme_order==2:
                du[:, 0] = du_forward[:, 0]
                du[:, 1] = du_forward[:, 1]
                du[:, -1] = du_backward[:, -1]
                du[:, -2] = du_backward[:, -2]
            elif scheme_order==1:
                u_shift_down_1 = torch.roll(u_tensor, 1)
                u_shift_down_2 = torch.roll(u_tensor, 2)
                u_shift_up_1 = torch.roll(u_tensor, -1)
                u_shift_up_2 = torch.roll(u_tensor, -2)
                
                h_shift_down_1 = torch.roll(h_tensor, 1)
                h_shift_down_2 = torch.roll(h_tensor, 2)
                h_shift_up_1 = torch.roll(h_tensor, -1)
                h_shift_up_2 = torch.roll(h_tensor, -2)
                
                h1_up=h_shift_up_1-h_tensor
                h2_up=h_shift_up_2-h_shift_up_1
                
                h1_down=h_tensor-h_shift_down_1
                h2_down=h_shift_down_1-h_shift_down_2
            
        
                a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
                b_up=(h2_up+h1_up)/(h1_up*h2_up)
                c_up=-h1_up/(h2_up*(h1_up+h2_up))
                
                a_down=(2*h1_up+h2_up)/(h1_down*(h1_down+h2_down))
                b_down=-(h2_down+h1_down)/(h1_down*h2_down)
                c_down=h1_down/(h2_down*(h1_down+h2_down))
            
                
                du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
                du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
                du[:, 0] = du_forward[:, 0]
                du[:, -1] = du_backward[:, -1]
                
        du = torch.transpose(du, 0, axis)

        return du


    def take_derivative(self, term):
        """
        Axiluary function serves for single differential operator resulting field
        derivation

        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        term : TYPE
            differential operator in conventional form.

        Returns
        -------
        der_term : torch.Tensor
            resulting field, computed on a grid.

        """
        # it is may be int, function of grid or torch.Tensor
        coeff = term[0]
        # this one contains product of differential operator
        operator_product = term[1]
        # float that represents power of the differential term
        power = term[2]
        # initially it is an ones field
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(operator_product):
            prod=self.model
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self.grid[axis]
                    prod=self.derivative(prod, h, axis, scheme_order=1, boundary_order=1)
            der_term = der_term * prod ** power[j]
        if callable(coeff) is True:
            der_term = coeff(self.grid) * der_term
        else:
            der_term = coeff * der_term
        return der_term

class Derivative():
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model

    def set_strategy(self, strategy):
        if strategy == 'NN':
            return Derivative_NN(self.grid, self.model)

        elif strategy == 'autograd':
            return  Derivative_autograd(self.grid, self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.grid, self.model)


class Solution():
    def __init__(self, grid, equal_cls, model, mode):
        self.grid = grid
        self.prepared_operator = equal_cls.operator_prepare()
        self.prepared_bconds = equal_cls.bnd_prepare()
        self.model = model
        self.mode = mode

    def apply_operator(self, operator):
        """
        Deciphers equation in a single grid subset to a field.

        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        operator : list
            Single (len(subset)==1) operator in input form. See 
            input_preprocessing.operator_prepare()

        Returns
        -------
        total : torch.Tensor

        """
        derivative = Derivative(self.grid, self.model).set_strategy(self.mode).take_derivative

        for term in operator:
            dif = derivative(term)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def apply_bconds_set(self, operator_set):
        """
        Deciphers equation in a whole grid to a field.
        Parameters
        ----------
        model : torch.Sequential
            Neural network.
        operator : list
            Multiple (len(subset)>=1) operators in input form. See 
            input_preprocessing.operator_prepare()
        Returns
        -------
        total : torch.Tensor
        """
        field_part = []
        for operator in operator_set:
            field_part.append(self.apply_operator(operator))
        field_part = torch.cat(field_part)
        return field_part

    def b_op_val_calc(self, bcond):
        b_pos = bcond[0]
        bop = bcond[1]
        truebval = bcond[2].reshape(-1,1)
        var = bcond[3]
        btype = bcond[4]
        if bop == None or bop == [[1, [None], 1]]:
            if self.mode == 'NN':
                grid_dict = Points_type.grid_sort(self.grid)
                sorted_grid = torch.cat(list(grid_dict.values()))
                b_op_val = self.model(sorted_grid)[:,var].reshape(-1,1)
            elif self.mode == 'autograd':
                b_op_val = self.model(self.grid)[:,var].reshape(-1,1)
            elif self.mode == 'mat':
                b_op_val = self.model
        else:
            if self.mode == 'NN':
                b_op_val = self.apply_bconds_set(bop)
            elif self.mode == 'autograd' or self.mode == 'mat':
                b_op_val = self.apply_operator(bop)
        return b_op_val

    def apply_bconds_operator(self):
        true_b_val_list = []
        b_val_list = []
        
        # we apply no  boundary conditions operators if they are all None

        for bcond in self.prepared_bconds:
            b_pos = bcond[0]
            bop = bcond[1]
            truebval = bcond[2].reshape(-1,1)
            var = bcond[3]
            btype = bcond[4]
            if btype == 'boundary values':
                true_b_val_list.append(truebval)
                b_op_val = self.b_op_val_calc(bcond)
                if self.mode == 'mat':
                    for position in b_pos:
                        if self.grid.dim() == 1 or min(self.grid.shape) == 1:
                            b_val_list.append(b_op_val[:, position])
                        else:
                            b_val_list.append(b_op_val[position])
                else:
                    b_val_list.append(b_op_val[b_pos])
            if btype == 'periodic':
                true_b_val_list.append(truebval)
                b_op_val = self.b_op_val_calc(bcond)
                b_val = b_op_val[b_pos[0]]
                for i in range(1, len(b_pos)):
                    b_val -= b_op_val[b_pos[i]]
                b_val_list.append(b_val)
        true_b_val = torch.cat(true_b_val_list)
        b_val=torch.cat(b_val_list).reshape(-1,1)

        return b_val, true_b_val

    def loss_evaluation(self, lambda_bound=10):
        if self.mode == 'mat' or self.mode == 'autograd':
            if self.prepared_bconds == None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(self.prepared_operator[0])
            if self.prepared_bconds == None:
                return torch.mean((op) ** 2)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(self.prepared_operator[i]))

            op = torch.cat(op_list,1)
            if self.prepared_bconds == None:
                return torch.sum(torch.mean((op) ** 2, 0))
    

        # we apply no  boundary conditions operators if they are all None

        b_val, true_b_val = self.apply_bconds_operator()
        """
        actually, we can use L2 norm for the operator and L1 for boundary
        since L2>L1 and thus, boundary values become not so signifnicant, 
        so the NN converges faster. On the other hand - boundary conditions is the
        crucial thing for all that stuff, so we should increase significance of the
        coundary conditions
        """
        # l1_lambda = 0.001
        # l1_norm =sum(p.abs().sum() for p in model.parameters())
        # loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)+ l1_lambda * l1_norm
        if self.mode=='mat':
            loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)
        else:
            loss = torch.sum(torch.mean((op) ** 2, 0)) + lambda_bound * torch.sum(torch.mean((b_val - true_b_val) ** 2, 0))

        return loss

