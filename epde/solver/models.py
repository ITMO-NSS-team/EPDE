"""Module keeps custom models arctectures"""

from typing import List, Any
import torch
from torch import nn
import numpy as np


class Fourier_embedding(nn.Module):
    """
    Class for Fourier features generation.

    Examples:
        u(t,x) if user wants to create 5 Fourier features in 'x' direction with L=5:
            L=[None, 5], M=[None, 5].
    """

    def __init__(self, L=[1], M=[1], ones=False):
        """
        Args:
            L (list, optional): (sin(w*x), cos(w*X)) frequencie parameter,
            w = 2*pi/L. Defaults to [1].
            M (list, optional): number of (sin, cos) pairs in result embedding. Defaults to [1].
            ones (bool, optional): enter or not ones vector in result embedding. Defaults to False.
        """

        super().__init__()
        self.M = M
        self.L = L
        self.idx = [i for i in range(len(self.M)) if self.M[i] is None]
        self.ones = ones
        self.in_features = len(M)
        not_none = sum(i for i in M if i is not None)
        is_none = self.M.count(None)
        if is_none == 0:
            self.out_features = not_none * 2 + self.in_features
        else:
            self.out_features = not_none * 2 + is_none
        if ones is not False:
            self.out_features += 1

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """ Forward method for Fourier features generation.

        Args:
            grid (torch.Tensor): calculation domain.

        Returns:
            torch.Tensor: embedding with Fourier features.
        """

        if self.idx == []:
            out = grid
        else:
            out = grid[:, self.idx]

        for i, _ in enumerate(self.M):
            if self.M[i] is not None:
                Mi = self.M[i]
                Li = self.L[i]
                w = 2.0 * np.pi / Li
                k = torch.arange(1, Mi + 1).reshape(-1, 1).float()
                x = grid[:, i].reshape(1, -1)
                x = (k @ x).T
                embed_cos = torch.cos(w * x)
                embed_sin = torch.sin(w * x)
                out = torch.hstack((out, embed_cos, embed_sin))

        if self.ones is not False:
            out = torch.hstack((out, torch.ones_like(out[:, 0:1])))

        return out


class FourierNN(nn.Module):
    """
    Class for realizing neural network with Fourier features
    and skip connection.
    """

    def __init__(self, layers=[100, 100, 100, 1], L=[1], M=[1],
                 activation=nn.Tanh(), ones=False):
        """

        Args:
            layers (list, optional): neurons quantity in each layer (exclusion input layer),
            the number of neurons in the hidden layers must match. Defaults to [100, 100, 100, 1].
            L (list, optional): (sin(w*x),cos(w*x)) frequency parameter, w=2*pi/L. Defaults to [1].
            M (list, optional): number of (sin, cos) pairs in result embedding. Defaults to [1].
            activation (_type_, optional): nn.Module object, activ-n function. Defaults to nn.Tanh().
            ones (bool, optional): enter or not ones vector in result embedding. Defaults to False.
        """

        super(FourierNN, self).__init__()
        self.L = L
        self.M = M
        FFL = Fourier_embedding(L=L, M=M, ones=ones)

        layers = [FFL.out_features] + layers

        self.linear_u = nn.Linear(layers[0], layers[1])
        self.linear_v = nn.Linear(layers[0], layers[1])

        self.activation = activation
        self.model = nn.ModuleList([FFL])
        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """ Forward pass for neural network.

        Args:
            grid (torch.Tensor): calculation domain.

        Returns:
            torch.Tensor: predicted values.
        """

        input_ = self.model[0](grid)
        v = self.activation(self.linear_v(input_))
        u = self.activation(self.linear_u(input_))
        for layer in self.model[1:-1]:
            output = self.activation(layer(input_))
            input_ = output * u + (1 - output) * v

        output = self.model[-1](input_)

        return output


class FeedForward(nn.Module):
    """Simple MLP neural network"""

    def __init__(self,
                 layers: List=[2, 100, 100, 100, 1],
                 activation: nn.Module = nn.Tanh(),
                 parameters: dict = None):
        """
        Args:
            layers (List, optional): neurons quantity in each layer.
            Defaults to [2, 100, 100, 100, 1].
            activation (nn.Module, optional): nn.Module object, activ-n function.
            Defaults to nn.Tanh().
            parameters (dict, optional): parameters initial values (for inverse task).
            Defaults to None.
        """
        super().__init__()
        model = []
        for i in range(len(layers)-2):
            model.append(nn.Linear(layers[i], layers[i+1]))
            model.append(activation)
        model.append(nn.Linear(layers[-2], layers[-1]))
        self.net = torch.nn.Sequential(*model)
        if parameters is not None:
            self.reg_param(parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward run

        Args:
            x (torch.Tensor): neural network inputs

        Returns:
            torch.Tensor: outputs
        """
        return self.net(x)

    def reg_param(self, parameters: dict):
        """ Parameters registration as neural network parameters.
        Should be used in inverse coefficients tasks.

        Args:
            parameters (dict): dict with initial values.
        """
        for key, value in parameters.items():
            parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                           requires_grad=True).float())
            self.net.register_parameter(key, parameters[key])


def parameter_registr(model: torch.nn.Module,
                      parameters: dict) -> None:
    """Parameters registration as neural network (mpdel) parameters.
        Should be used in inverse coefficients tasks.

    Args:
        model (torch.nn.Module): neural network.
        parameters (dict): dict with initial values.
    """
    for key, value in parameters.items():
        parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                        requires_grad=True).float())
        model.register_parameter(key, parameters[key])


def mat_model(domain: Any,
              equation: Any,
              nn_model: torch.nn.Module = None) -> torch.Tensor:
    """ Model creation for *mat* mode.

    Args:
        domain (Any): object of Domian class.
        equation (Any): Equation class object (see data module).
        nn_model (torch.nn.Module, optional): neural network which outputs will be *mat* model.
        Defaults to None.

    Returns:
        torch.nn.Module: model for *mat* mode.
    """

    grid = domain.build('mat')

    eq_num = len(equation.equation_lst)

    shape = [eq_num] + list(grid.shape)[1:]

    if nn_model is not None:
        nn_grid = torch.vstack([grid[i].reshape(-1) for i in \
                                range(grid.shape[0])]).T.float()
        model = nn_model(nn_grid).detach()
        model = model.reshape(shape)
    else:
        model = torch.ones(shape)

    return model
