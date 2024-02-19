import torch
from typing import Any, Union, List
import numpy as np
from scipy import linalg


class DerivativeInt():
    def take_derivative(self, value):
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Taking numerical derivative for 'NN' method.
    """

    def __init__(self, model: Any):
        """
        Args:
            model: neural network.
        """
        self.model = model

    def take_derivative(self, term: Union[list, int, torch.Tensor], *args) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.
        Args:
            term: differential operator in conventional form.
        Returns:
            resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        if type(term['coeff']) is tuple:
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            grid_sum = 0.
            for k, grid in enumerate(scheme):
                grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1)\
                    * term[dif_dir][1][j][k]
            print(f'grid_sum in take_derivatives is {grid_sum.shape} with the grid of {grid.shape}')
            der_term = der_term * grid_sum ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def nn_autograd(model, points, var, axis=[0]):
        """
        Computes derivative on the grid using autograd method.
        Args:
            model: neural network.
            points: points, where numerical derivative is calculated.
            axis: term of differentiation, example [0,0]->d2/dx2
                  if grid_points(x,y)
        Returns:
            the result of desired function differentiation
                in corresponding axis.

        """

        points.requires_grad = True
        fi = model(points)[:, var].sum(0)
        for ax in axis:
            grads, = torch.autograd.grad(fi, points, create_graph=True)
            fi = grads[:, ax].sum()
        gradient_full = grads[:, axis[-1]].reshape(-1, 1)
        return gradient_full

    def take_derivative(self, term: dict, grid_points:  torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.
        Args:
            term: differential operator in conventional form.
            grid_points: points, where numerical derivative is calculated.
        Returns:
            resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, derivative in enumerate(term[dif_dir]):
            if derivative == [None]:
                der = self.model(grid_points)[:, term['var'][j]].reshape(-1, 1)
            else:
                der = self.nn_autograd(
                    self.model, grid_points, term['var'][j], axis=derivative)
            der_term = der_term * der ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_mat(DerivativeInt):
    """
    Taking numerical derivative for 'mat' method.
    """
    def __init__(self, model, derivative_points):
        """
        Args:
            model: random matrix.
        """
        self.model = model
        self.backward, self.farward = Derivative_mat.labels(derivative_points)

        self.alpha_backward = Derivative_mat.linear_system(self.backward)
        self.alpha_farward = Derivative_mat.linear_system(self.farward)

        num_points = int(len(self.backward)-1)

        self.back = [int(0-i) for i in range(1,num_points+1)]

        self.farw = [int(i) for i in range(num_points)]


    @staticmethod
    def labels(num_points):
        labels_backward = [i for i in range(-num_points+1,1)]
        labels_farward = [i for i in range(num_points)]
        return labels_backward, labels_farward

    @staticmethod
    def linear_system(labels):
        points_num = len(labels) # num_points=number of equations
        labels = np.array(labels)
        A = []
        for i in range(points_num):
            A.append(labels**i)
        A = np.array(A)

        b = np.zeros_like(labels)
        b[1] = 1

        alpha = linalg.solve(A, b)

        return alpha

    def derivative_1d(self, u_tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes derivative in one dimension for matrix method.
        Args:
            model: random matrix.
            grid: array of a n-D points.
        Returns:
            computed derivative along one dimension.
        """
        shape = u_tensor.shape
        u_tensor = u_tensor.reshape(-1)

        du_back = 0
        du_farw = 0
        i=0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b)*self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f)*self.alpha_farward[i]
            i += 1
        du = (du_back+du_farw)/(2*h)
        du[self.back] = du_back[self.back]/h
        du[self.farw] = du_farw[self.farw]/h


        du = du.reshape(shape)

        return du


    def step_h(self, h_tensor: torch.Tensor) -> List[torch.Tensor]:
        h = []

        NN_grid = torch.vstack([h_tensor[i].reshape(-1) for i in \
                                range(h_tensor.shape[0])]).T.float()
        
        for i in range(NN_grid.shape[-1]):
            axis_points = torch.unique(NN_grid[:,i])
            h.append(abs(axis_points[1]-axis_points[0]))
        return h

    def derivative(self, u_tensor: torch.Tensor, h, axis: int) -> torch.Tensor:
        """
        Computing derivative for 'matrix' method.
        Args:
            u_tensor: smth.
            h_tensor: smth.
            axis: axis along which the derivative is calculated.
            scheme_order: accuracy inner order for finite difference. Default = 1
            boundary_order: accuracy boundary order for finite difference. Default = 2
        Returns:
            computed derivative.
        """
        if len(u_tensor.shape)==1 or u_tensor.shape[0]==1:
            du = self.derivative_1d(u_tensor, h)
            return du

        pos = len(u_tensor.shape) - 1

        u_tensor = torch.transpose(u_tensor, pos, axis)

        du_back = 0
        du_farw = 0
        i=0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b)*self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f)*self.alpha_farward[i]
            i += 1
        du = (du_back+du_farw)/(2*h)
        
        if pos == 1:
            du[:,self.back] = du_back[:,self.back]/h
            du[:, self.farw] = du_farw[:, self.farw]/h
        elif pos == 2:
            du[:,:, self.back] = du_back[:,:, self.back] / h
            du[:,:, self.farw] = du_farw[:,:, self.farw] / h

        du = torch.transpose(du, pos, axis)

        return du

    def take_derivative(self, term: Any, grid_points: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function serves for single differential operator resulting field
        derivation.
        Args:
            term: differential operator in conventional form.
            grid_points: grid points
        Returns:
            resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod=self.model[term['var'][j]]
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self.step_h(grid_points)[axis]
                    prod = self.derivative(prod, h, axis)
            der_term = der_term * prod ** term['pow'][j]
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    """
   Interface for taking numerical derivative due to chosen calculation method.

   """
    def __init__(self, model, derivative_points):
        """
        Args:
            model: neural network or matrix depending on the selected mode.

        """
        self.model = model
        self.derivative_points = derivative_points

    def set_strategy(self, strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Setting the calculation method.
        Args:
            strategy: Calculation method. (i.e., "NN", "autograd", "mat").
        Returns:
            equation in input form for a given calculation method.
        """
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return  Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model, self.derivative_points)
