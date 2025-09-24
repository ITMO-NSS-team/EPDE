"""Module keeps custom models arctectures"""

from typing import List, Any
import torch
from torch import nn
import numpy as np


class Fourier_embedding(nn.Module):
    """
    Transforms input data into a Fourier feature representation.
    
    
        Examples:
            u(t,x) if user wants to create 5 Fourier features in 'x' direction with L=5:
                L=[None, 5], M=[None, 5].
    """


    def __init__(self, L=[1], M=[1], ones=False, device='cpu'):
        """
        Initializes the Fourier embedding layer.
        
                This layer transforms input features into a higher-dimensional space using Fourier basis functions.
                The transformation enriches the representation of the input, allowing the model to capture complex relationships
                and periodic patterns in the data. The frequencies and number of components used in the Fourier embedding
                are configurable, providing flexibility in adapting to different data characteristics.
        
                Args:
                    L (list, optional): Characteristic length scales for the Fourier basis functions.
                        Determines the frequencies (w = 2*pi/L) used in the sine and cosine components. Defaults to [1].
                    M (list, optional): Number of (sin, cos) pairs to use for each length scale specified in `L`.
                        If an element is None, it will not be used. Defaults to [1].
                    ones (bool, optional): Whether to include a vector of ones in the output embedding. Defaults to False.
                    device (str, optional): Device to store tensors. Defaults to 'cpu'.
        
                Returns:
                    None
        """

        self._device = device
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
        """
        Generates Fourier feature embeddings for a given grid.
        
        This method transforms the input grid into a higher-dimensional space
        using Fourier features. This transformation allows the model to
        capture complex relationships in the data by representing it in terms
        of sinusoidal functions with varying frequencies. This is particularly
        useful when the underlying relationships in the data are periodic or
        oscillatory, which is common in many physical systems governed by
        differential equations.
        
        Args:
            grid (torch.Tensor): The input grid representing the domain of the
                problem.  Shape should be (N, D) where N is the number of points
                and D is the dimensionality of the grid.
        
        Returns:
            torch.Tensor: The embedding of the grid with Fourier features. The
                output tensor has shape (N, M), where M is the dimensionality
                of the embedded space, determined by the number of Fourier
                features added.
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
                k = torch.arange(1, Mi + 1, device=self._device).reshape(-1, 1).float()
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
        Initializes the Fourier Neural Network (FNN) with specified layer configurations and Fourier embedding parameters.
        
                This initialization prepares the network architecture, incorporating a Fourier embedding layer to map the input data into a higher-dimensional space using sinusoidal functions. This approach enhances the network's ability to capture complex patterns and relationships within the data, which is particularly useful when modeling solutions to differential equations. The Fourier embedding enables the network to learn the underlying frequencies and modes present in the data, leading to more accurate and efficient equation discovery.
        
                Args:
                    layers (list, optional): Number of neurons in each layer (excluding the input layer). The number of neurons in the hidden layers must match. Defaults to [100, 100, 100, 1].
                    L (list, optional): Frequency parameter for the Fourier embedding, where w = 2*pi/L. Defaults to [1].
                    M (list, optional): Number of (sin, cos) pairs in the Fourier embedding. Defaults to [1].
                    activation (nn.Module, optional): Activation function to be used in the hidden layers. Defaults to nn.Tanh().
                    ones (bool, optional): Whether to include a vector of ones in the Fourier embedding. Defaults to False.
        
                Returns:
                    None
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
        """
        Performs a forward pass through the Fourier Neural Network.
        
        The forward pass propagates the input grid through a series of linear and non-linear transformations,
        modulated by learned functions 'u' and 'v'. These functions control the contribution of each layer's
        output to the subsequent layer's input, effectively shaping the network's response to capture complex relationships
        within the input data. This architecture allows the network to approximate solutions to differential equations
        by learning the underlying functional relationships.
        
        Args:
            grid (torch.Tensor): The input grid representing the domain of the differential equation.
        
        Returns:
            torch.Tensor: The predicted values at each point on the grid, representing the approximate solution to the differential equation.
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
    """
    Simple MLP neural network
    """


    def __init__(self,
                 layers: List = [2, 100, 100, 100, 1],
                 activation: nn.Module = nn.Tanh(),
                 parameters: dict = None):
        """
        Initializes the FeedForward neural network.
        
                This network architecture is designed to approximate solutions
                to differential equations by learning the underlying relationships
                within the data. The structure and activation functions are
                configured to effectively capture the complex dynamics inherent
                in these equations.
        
                Args:
                    layers (List, optional): A list defining the number of neurons in each layer of the network.
                        The first element represents the input layer size, and the last element represents the output layer size.
                        Defaults to [2, 100, 100, 100, 1].
                    activation (nn.Module, optional): The activation function to be applied after each linear layer (except the last).
                        Defaults to nn.Tanh().
                    parameters (dict, optional): Initial parameter values for the network's weights and biases.
                        This can be useful for initializing the network with prior knowledge or for inverse problems.
                        Defaults to None.
        
                Returns:
                    None
        """

        super().__init__()
        self.model = []

        for i in range(len(layers) - 2):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))
            self.model.append(activation)
        self.model.append(nn.Linear(layers[-2], layers[-1]))
        self.net = torch.nn.Sequential(*self.model)
        if parameters is not None:
            self.reg_param(parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the feedforward network.
        
        This method takes input data and propagates it through the network's layers
        to generate predictions. It leverages the defined network architecture
        to transform the input into a corresponding output, effectively mapping
        the input data to a solution space.
        
        Args:
            x (torch.Tensor): The input tensor to the network.
        
        Returns:
            torch.Tensor: The output tensor produced by the network.
        """
        return self.net(x)

    def reg_param(self,
                  parameters: dict):
        """
        Registers provided parameters as trainable parameters of the neural network.
        
                This is crucial for tasks where parameters need to be optimized alongside the network's weights,
                allowing the network to learn the optimal values for these parameters during training.
        
                Args:
                    parameters (dict): A dictionary where keys are parameter names and values are initial parameter values.
        
                Returns:
                    None
        """
        for key, value in parameters.items():
            parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                              requires_grad=True).float())
            self.net.register_parameter(key, parameters[key])


def parameter_registr(model: torch.nn.Module,
                      parameters: dict) -> None:
    """
    Registers given parameters as trainable parameters of the neural network.
    
    This is essential for incorporating prior knowledge or initial guesses into the model,
    allowing the optimization process to refine these values alongside the network's weights.
    
    Args:
        model (torch.nn.Module): The neural network model to register parameters with.
        parameters (dict): A dictionary where keys are parameter names (strings) and values are initial numerical values.
    
    Returns:
        None
    """
    for key, value in parameters.items():
        parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                                          requires_grad=True).float())
        model.register_parameter(key, parameters[key])


def mat_model(domain: Any,
              equation: Any,
              nn_model: torch.nn.Module = None) -> torch.Tensor:
    """
    Generates a matrix-based representation of the model based on the domain and equation.
    
    This function constructs a model by evaluating the equation on a grid defined by the domain.
    If a neural network is provided, it uses the network's output to shape the model; otherwise,
    it initializes a matrix of ones. This matrix-based representation is useful for further
    analysis or processing within the equation discovery workflow.
    
    Args:
        domain (Any): Object representing the domain over which the equation is defined.
        equation (Any): Object containing the equation to be modeled.
        nn_model (torch.nn.Module, optional): A neural network to be used as a part of model. Defaults to None.
    
    Returns:
        torch.Tensor: A tensor representing the model in matrix form.
    """

    grid = domain.build('mat')

    eq_num = len(equation.equation_lst)

    shape = [eq_num] + list(grid.shape)[1:]

    if nn_model is not None:
        nn_grid = torch.vstack([grid[i].reshape(-1) for i in range(grid.shape[0])]).T.float()
        model = nn_model(nn_grid).detach()
        model = model.reshape(shape)
    else:
        model = torch.ones(shape)

    return model
