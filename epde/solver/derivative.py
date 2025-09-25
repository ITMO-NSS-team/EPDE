"""Module of derivative calculations.
"""

from typing import Any, Union, List, Tuple, Callable
import numpy as np
from scipy import linalg
import torch


class DerivativeInt():
    """
    Interface class
    """

    def take_derivative(self, value):
        """
        Calculates the derivative of the expression with respect to a given variable.
        
        This method must be implemented by subclasses to provide specific derivative calculation logic
        based on the expression's structure. It ensures that all expression types support derivative
        calculation, which is a core requirement for equation discovery within the EPDE framework.
        
        Args:
            value (str): The variable with respect to which the derivative is calculated.
        
        Returns:
            NotImplementedError: Always raises an exception, as the method must be implemented in a subclass.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Taking numerical derivative for 'NN' method.
    """


    def __init__(self, model: Any):
        """
        Initializes the Derivative_NN with a neural network model. This model will be used to approximate derivatives, which are then used to construct and evaluate potential differential equations.
        
                Args:
                    model: A neural network used for derivative approximation.
        
                Returns:
                    None
        """
        self.model = model

    def take_derivative(self, term: Union[list, int, torch.Tensor], *args) -> torch.Tensor:
        """
        Computes the contribution of a single differential operator term to the overall equation.
        
                This function evaluates a term in the differential equation by applying the specified
                derivative scheme to the neural network's output and multiplying by the coefficient.
                It effectively calculates one component of the equation's residual, which is then
                used to optimize the neural network's parameters to satisfy the differential equation.
        
                Args:
                    term (Union[list, int, torch.Tensor]): A dictionary representing a single term
                        in the differential equation. It contains the coefficient, derivative direction,
                        variable dependencies, and power of the term.
        
                Returns:
                    torch.Tensor: The computed value of the differential operator term, evaluated
                        on the grid points. This represents the contribution of this specific term
                        to the overall differential equation at each point in the domain.
        """

        dif_dir = list(term.keys())[1]
        if isinstance(term['coeff'], tuple):
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            if isinstance(term['var'][j], (list, tuple)):
                raise NotImplementedError('Support for multivariate function tokens was introduced only for autograd.')
                if not isinstance(term['pow'][j], (Callable, torch.nn.Sequential)):
                    raise ValueError('Multivariate function can not be passed as a simple power func.')
                der_args = []
                for var_idx, cur_var in enumerate(term['var'][j]):
                    grid_sum = 0.
                    for k, grid in enumerate(scheme):
                        grid_sum += self.model(grid)[:, cur_var].reshape(-1, 1)\
                                * term[dif_dir][1][j][k]
                    der_args.append(grid_sum)

                    # if derivative[var_idx] == [None]:
                    #     der_args.append(self.model(grid_points)[:, cur_var].reshape(-1, 1))
                    # else:
                    #     der_args.append(self._nn_autograd(self.model, grid_points, cur_var, axis=derivative[var_idx]))
                if isinstance(term['pow'][j], torch.nn.Sequential):
                    der_args = torch.cat(der_args, dim=1)
                    factor_val = term['pow'][j](der_args)
                else:
                    factor_val = term['pow'][j](*der_args)
                der_term = der_term * factor_val
            else:
                grid_sum = 0.
                for k, grid in enumerate(scheme):
                    grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1)\
                            * term[dif_dir][1][j][k]
                    
                if isinstance(term['pow'][j],(int,float)):
                    der_term = der_term * grid_sum ** term['pow'][j]
                elif isinstance(term['pow'][j],Callable):
                    der_term = der_term * term['pow'][j](grid_sum)
        der_term = coeff * der_term

        return der_term


class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """


    def __init__(self, model: torch.nn.Module):
        """
        Initializes the Derivative_autograd class.
        
                This class prepares a given neural network model for subsequent symbolic differentiation, enabling the identification of potential differential equation terms.
        
                Args:
                    model (torch.nn.Module): The neural network model to be differentiated.
        
                Returns:
                    None
        """
        self.model = model

    @staticmethod
    def _nn_autograd(model: torch.nn.Module,
                     points: torch.Tensor,
                     var: int,
                     axis: List[int] = [0]):
        """
        Computes the derivative of a neural network model's output with respect to its input using PyTorch's automatic differentiation. This function is a core component for calculating the residuals required to evaluate candidate differential equations.
        
                Args:
                    model (torch.nn.Module): The neural network model.
                    points (torch.Tensor): The input points at which to compute the derivative.
                    var (int): The index of the output variable to differentiate (for systems of equations).
                    axis (List[int], optional): The axes with respect to which to differentiate. Defaults to [0].
        
                Returns:
                    torch.Tensor: The computed derivative at the given points.
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
        Computes the contribution of a single term in the differential operator.
        
        This function calculates the value of a single term within a larger differential operator,
        effectively evaluating a component of the equation on the given grid points. It handles
        various forms of coefficients and derivatives, including those defined by neural networks
        or other callable functions.
        
        Args:
            term (dict): A dictionary representing a single term in the differential operator.
                         It contains information about the coefficient, derivative direction,
                         variables, and powers involved in the term.
            grid_points (torch.Tensor): The points on which the derivative is evaluated.
                                         Shape: (N, D), where N is the number of points and D is the
                                         dimensionality of the grid.
        
        Returns:
            torch.Tensor: The value of the term evaluated at the given grid points.
                          Shape: (N, 1), where N is the number of points.
        
        Why:
            This function is a crucial part of constructing the overall differential operator.
            By computing each term individually, the method allows for flexible and modular
            representation of complex differential equations.
        """
        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1. # TODO: Переписать!
        for j, derivative in enumerate(term[dif_dir]):
            if (isinstance(term['pow'][j], torch.nn.Sequential) or
                (isinstance(term['pow'][j], Callable) and isinstance(term['var'][j], (list, tuple)))): #isinstance(term['var'][j], (list, tuple)):
                der_args = []
                iter_arg = term['var'][j] if isinstance(term['var'][j], (list, tuple)) else [term['var'][j],]
                for var_idx, cur_var in enumerate(iter_arg):
                    if derivative[var_idx] == [None] or derivative[var_idx] is None:
                        der_args.append(self.model(grid_points)[:, cur_var].reshape(-1, 1))
                    else:
                        der_args.append(self._nn_autograd(self.model, grid_points, cur_var, axis=derivative[var_idx]))
                if isinstance(term['pow'][j], torch.nn.Sequential):
                    der_args = torch.cat(der_args, dim=1)
                    factor_val = term['pow'][j](der_args)
                else:             
                    factor_val = term['pow'][j](*der_args)
                der_term = der_term * factor_val
            else:
                if derivative == [None] or derivative is None:
                    der = self.model(grid_points)[:, term['var'][j]].reshape(-1, 1)
                else:
                    der = self._nn_autograd(self.model, grid_points, term['var'][j], axis=derivative)
                if isinstance(term['pow'][j],(int,float)):
                    der_term = der_term * der ** term['pow'][j]
                elif isinstance(term['pow'][j], Callable):
                    der_term = der_term * term['pow'][j](der)
        der_term = coeff * der_term
        return der_term


class Derivative_mat(DerivativeInt):
    """
    Taking numerical derivative for 'mat' method.
    """

    def __init__(self, model: torch.Tensor, derivative_points: int):
        """
        Initializes the Derivative_mat object.
        
        This class precomputes coefficients used for approximating derivatives 
        based on a given model and number of derivative points. These coefficients 
        are then used to efficiently calculate derivatives during the equation 
        discovery process.
        
        Args:
            model (torch.Tensor): The model in *mat* mode, representing the data 
                from which derivatives will be estimated.
            derivative_points (int): The number of points used in the derivative 
                calculation.  A higher number of points can lead to more accurate 
                derivative estimates, but also increases computational cost.
        
        Returns:
            None
        """
        self.model = model
        self.backward, self.farward = Derivative_mat._labels(derivative_points)

        self.alpha_backward = Derivative_mat._linear_system(self.backward)
        self.alpha_farward = Derivative_mat._linear_system(self.farward)

        num_points = int(len(self.backward) - 1)

        self.back = [int(0 - i) for i in range(1, num_points + 1)]

        self.farw = [int(i) for i in range(num_points)]

    @staticmethod
    def _labels(derivative_points: int) -> Tuple[List, List]:
        """
        Determines the indices of neighboring points used to approximate derivatives.
        
        This function generates index offsets for both backward and forward finite difference schemes,
        defining which neighboring points are used in the derivative calculation. These indices are
        essential for constructing derivative matrices that approximate differential operators on a discrete grid.
        
        Args:
            derivative_points (int): The number of points used in the derivative calculation.
        
        Returns:
            Tuple[List[int], List[int]]: A tuple containing two lists:
                - labels_backward: Index offsets for the backward difference scheme.
                - labels_forward: Index offsets for the forward difference scheme.
        """
        labels_backward = list(i for i in range(-derivative_points + 1, 1))
        labels_farward = list(i for i in range(derivative_points))
        return labels_backward, labels_farward

    @staticmethod
    def _linear_system(labels: list) -> np.ndarray:
        """
        Solves a linear system to determine the coefficients for a numerical differentiation scheme.
        
                The coefficients are calculated by solving a linear system derived from a Vandermonde matrix.
                This approach ensures that the resulting differentiation scheme accurately approximates the derivative
                based on the provided stencil points.
        
                Args:
                    labels (list): The stencil points (x-coordinates) used in the differentiation scheme.
        
                Returns:
                    np.ndarray: The coefficients for the numerical differentiation scheme, obtained by solving the linear system.
        """
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

    def _derivative_1d(self, u_tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the numerical derivative of a tensor along one dimension using a finite difference scheme. This is a core component for estimating derivatives within the equation discovery process.
        
                Args:
                    u_tensor (torch.Tensor): The input tensor for which the derivative is computed. Represents the dependent variable in the equation.
                    h (torch.Tensor): The step size or increment used in the finite difference approximation.
        
                Returns:
                    du (torch.Tensor): The computed derivative of the input tensor along the specified dimension.
        """

        shape = u_tensor.shape
        u_tensor = u_tensor.reshape(-1)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)
        du[self.back] = du_back[self.back] / h
        du[self.farw] = du_farw[self.farw] / h


        du = du.reshape(shape)

        return du


    def _step_h(self, h_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate the grid increments along each axis.
        
        This function computes the step size (h) along each axis of the grid,
        which represents the resolution of the grid in each dimension.
        These increments are crucial for calculating derivatives on the grid.
        
        Args:
            h_tensor (torch.Tensor): Grid of points in *mat* mode.
                Shape: (num_axes, num_points_along_axis).
        
        Returns:
            List[torch.Tensor]: A list containing the increment (step size)
            along each axis of the grid. The length of the list corresponds
            to the number of axes.
        """
        h = []

        nn_grid = torch.vstack([h_tensor[i].reshape(-1) for i in \
                                range(h_tensor.shape[0])]).T.float()

        for i in range(nn_grid.shape[-1]):
            axis_points = torch.unique(nn_grid[:,i])
            h.append(abs(axis_points[1]-axis_points[0]))
        return h

    def _derivative(self,
                    u_tensor: torch.Tensor,
                    h: torch.Tensor,
                    axis: int) -> torch.Tensor:
        """
        Computes the numerical derivative of a tensor along a specified axis using a finite difference scheme. This is a core component for estimating equation terms from data.
        
                Args:
                    u_tensor (torch.Tensor): The tensor for which the derivative is computed. Represents a dependent variable in the equation.
                    h (torch.Tensor): The step size (increment) used in the finite difference approximation.
                    axis (int): The axis along which the derivative is calculated.
        
                Returns:
                    du (torch.Tensor): The computed derivative of the input tensor.
        """

        if len(u_tensor.shape)==1 or u_tensor.shape[0]==1:
            du = self._derivative_1d(u_tensor, h)
            return du

        pos = len(u_tensor.shape) - 1

        u_tensor = torch.transpose(u_tensor, pos, axis)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)

        if pos == 1:
            du[:,self.back] = du_back[:,self.back] / h
            du[:, self.farw] = du_farw[:, self.farw] / h
        elif pos == 2:
            du[:,:, self.back] = du_back[:,:, self.back] / h
            du[:,:, self.farw] = du_farw[:,:, self.farw] / h

        du = torch.transpose(du, pos, axis)

        return du

    def take_derivative(self, term: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to compute the contribution of a single term in the differential operator. This involves calculating derivatives of the dependent variables based on the specified scheme and grid points, and then combining them according to the term's structure.
        
                Args:
                    term (torch.Tensor): A dictionary representing a single term in the differential operator, containing information about the variable, derivative order, and coefficient.
                    grid_points (torch.Tensor): The coordinates at which the derivatives are evaluated.
        
                Returns:
                    der_term (torch.Tensor): The computed value of the term at each grid point. This represents the contribution of this specific term to the overall differential equation.
        
                WHY: This function is a crucial step in evaluating the differential operator on the grid. By computing each term's contribution individually, the method allows for flexible and modular equation discovery.
        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod=self.model[term['var'][j]]
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self._step_h(grid_points)[axis]
                    prod = self._derivative(prod, h, axis)
            der_term = der_term * prod ** term['pow'][j]
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    """
    Interface for calculating numerical derivatives. It supports different calculation modes to provide flexibility and accuracy in derivative estimation.
    """

    def __init__(self, 
                 model: Union[torch.nn.Module, torch.Tensor],
                 derivative_points: int):
        """
        Initializes the Derivative class.
        
        This class prepares the model for derivative calculations,
        setting the stage for subsequent equation discovery. The number of
        derivative points influences the accuracy and computational cost of
        approximating derivatives, which are crucial for identifying the
        underlying differential equations.
        
        Args:
            model (Union[torch.nn.Module, torch.Tensor]): The neural network or
                matrix representing the system's behavior. This is the model
                from which derivatives will be computed.
            derivative_points (int): The number of points to use in the
                numerical derivative calculation.  A higher number of points
                can increase accuracy but also computational complexity.
                For example, if derivative_points=2, a simple two-point
                numerical scheme like ([-1,0],[0,1]) will be used.
        
        Returns:
            None
        """

        self.model = model
        self.derivative_points = derivative_points

    def set_strategy(self,
                     strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Sets the differentiation strategy.
        
        This method configures the approach used to calculate derivatives, 
        allowing the user to select the most appropriate technique based on 
        the problem and available resources. Different strategies offer trade-offs 
        between computational cost, accuracy, and the need for analytical 
        derivatives. The choice of strategy impacts how the derivatives 
        required for equation discovery are computed.
        
        Args:
            strategy (str): The name of the differentiation strategy to use. 
                            Valid options are "NN" (neural network-based), 
                            "autograd" (automatic differentiation), and "mat" 
                            (matrix-based).
        
        Returns:
            Union[Derivative_NN, Derivative_autograd, Derivative_mat]: An instance of the class 
            corresponding to the selected differentiation strategy, initialized 
            with the provided model and derivative points (if applicable).
        """
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return  Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model, self.derivative_points)
