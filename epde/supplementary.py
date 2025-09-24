#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

from abc import ABC
from typing import Callable, Union

import numpy as np
from functools import reduce
import copy
import torch
# device = torch.device('cpu')

import matplotlib.pyplot as plt

from epde.solver.data import Domain
from epde.solver.models import Fourier_embedding, mat_model


class BasicDeriv(ABC):
    """
    Abstract base class for defining custom derivative implementations.
    
        This class serves as a template for creating new derivative methods.
        It enforces the implementation of the `take_derivative` method.
    
        Methods:
        - take_derivative: Abstract method for computing the derivative.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the BasicDeriv object.
        
        This method is designed to prevent direct instantiation of the abstract `BasicDeriv` class.
        Since `BasicDeriv` serves as a blueprint for specific differentiation strategies used in equation discovery,
        it should not be instantiated directly.
        
        Args:
            *args: Variable length argument list.  Not used directly.
            **kwargs: Arbitrary keyword arguments. Not used directly.
        
        Raises:
            NotImplementedError: Always raised, indicating that direct instantiation of `BasicDeriv` is prohibited.
        
        Returns:
            None.
        """
        raise NotImplementedError('Trying to create abstract differentiation method')
    
    def take_derivative(self, u: torch.Tensor, args: torch.Tensor, axes: list):
        """
        Takes the derivative of a tensor with respect to specified variables.
        
        This method serves as a placeholder in the abstract `BasicDeriv` class and must be implemented by subclasses to provide concrete differentiation logic.  Since `BasicDeriv` defines the interface for all differentiation methods, this abstract method ensures that any concrete implementation provides a `take_derivative` method.
        
        Args:
            u (torch.Tensor): The input tensor to differentiate.
            args (torch.Tensor): The tensor representing the variables with respect to which the derivative is taken.
            axes (list): A list of axes along which to compute the derivative.
        
        Returns:
            None: This method is abstract and should raise an error.
        
        Raises:
            NotImplementedError: Always raised, as this is an abstract method that must be implemented by a subclass.
        """
        raise NotImplementedError('Trying to differentiate with abstract differentiation method')


class AutogradDeriv(BasicDeriv):
    """
    A class for computing derivatives using PyTorch's autograd functionality.
    
        Class Methods:
        - __init__:
    """

    def __init__(self):
        """
        Initializes an AutogradDeriv object.
        
        This class facilitates automatic differentiation, enabling the computation of derivatives of mathematical expressions.
        The initialization prepares the object for tracking operations and calculating gradients.
        
        Args:
            self: The AutogradDeriv instance.
        
        Returns:
            None.
        """
        pass

    def take_derivative(self, u: Union[torch.nn.Sequential, torch.Tensor], args: torch.Tensor, 
                        axes: list = [], component: int = 0):
        """
        Computes the derivative of a function `u` with respect to input `args` along specified axes. This is a crucial step in identifying the underlying differential equations, as it allows us to estimate the rates of change of the system's variables.
        
                Args:
                    u: The function to differentiate. It can be a `torch.nn.Sequential` model or a `torch.Tensor`. Represents the system or model being analyzed.
                    args: The input tensor with respect to which the derivative is computed. Represents the independent variables of the system.
                    axes: A list of axes along which to compute the derivative. Defaults to an empty list. Specifies the dimensions along which the rate of change is calculated.
                    component: The component of the output to consider for differentiation. Defaults to 0. Allows focusing on specific parts of a multi-dimensional output.
                
                Returns:
                    torch.Tensor: The computed derivative. This derivative is then used to construct and evaluate candidate differential equations.
        """
        if not args.requires_grad:
            args.requires_grad = True
        if axes == [None,]:
            return u(args)[..., component].reshape(-1, 1)
        if isinstance(u, torch.nn.Sequential):
            comp_sum = u(args)[..., component].sum(dim = 0)
        elif isinstance(u, torch.Tensor):
            raise TypeError('Autograd shall have torch.nn.Sequential as its inputs.')
        else:
            print(f'u.shape, {u.shape}')
            comp_sum = u.sum(dim = 0)
        for axis in axes:
            output_vals = torch.autograd.grad(outputs = comp_sum, inputs = args, create_graph=True)[0]
            comp_sum = output_vals[:, axis].sum()
        output_vals = output_vals[:, axes[-1]].reshape(-1, 1)
        return output_vals

class FDDeriv(BasicDeriv):
    """
    Calculates numerical derivatives of multi-dimensional arrays using finite difference methods.
    
        This class provides a method to compute the derivative of a function represented by a multi-dimensional array.
    
        Class Methods:
        - take_derivative:
    """

    def __init__(self):
        """
        Initializes a new instance of the FDDeriv class.
        
        This method serves as the constructor for the FDDeriv class, preparing it
        for subsequent operations involving finite difference approximations.
        It currently performs no actions but initializes the object.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Why:
            The initialization prepares the FDDeriv object for calculating derivatives
            using finite difference methods, a core component in discovering
            differential equations from data by estimating derivatives from provided
            data points.
        """
        pass

    def take_derivative(self, u: np.ndarray, args: np.ndarray, 
                        axes: list = [], component: int = 0):
        """
        Calculates the numerical derivative of a function.
        
                This method computes the derivative of a function represented by a multi-dimensional array `u` with respect to specified axes.
                It uses `np.gradient` to approximate the derivative. This is a crucial step in discovering the underlying differential equations, as it allows us to estimate the rates of change of the function, which are fundamental to formulating the equations.
        
                Args:
                    u (np.ndarray): The input array representing the function values.
                    args (np.ndarray): The coordinates at which the function is evaluated.
                    axes (list, optional): The axes along which to compute the derivative. Defaults to an empty list.
                    component (int, optional): The component of the input array to differentiate. Defaults to 0.
        
                Returns:
                    np.ndarray: The derivative of the input array along the specified axes.
        """
        
        if not isinstance(args, torch.Tensor):
            args = args.detach().cpu().numpy()

        output_vals = u[..., component].reshape(args.shape)
        if axes == [None,]:
            return output_vals
        for axis in axes:
            output_vals = np.gradient(output_vals, args.reshape(-1)[1] - args.reshape(-1)[0], axis = axis, edge_order=2)  
        return output_vals

def create_solution_net(equations_num: int, domain_dim: int, use_fourier = True, #  mode: str, domain: Domain 
                        fourier_params: dict = None, device = 'cpu'):
    """
    Creates a neural network architecture suitable for solving differential equations, optionally incorporating a Fourier embedding layer.
    
        The Fourier embedding enhances the network's ability to represent complex functions, particularly those arising from differential equations.
        The network architecture is designed to map input coordinates to the solution of the differential equation.
    
        Args:
            equations_num (int): The number of equations in the system. Determines the output dimension of the network.
            domain_dim (int): The dimensionality of the input domain.
            use_fourier (bool, optional): Whether to use a Fourier embedding layer. Defaults to True.
            fourier_params (dict, optional): Parameters for the Fourier embedding layer. If None, default parameters are used.
                Should be a dict with entries like: {'L' : [4,], 'M' : [3,]}. Defaults to None.
            device (str, optional): The device to use for the network (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
    
        Returns:
            torch.nn.Sequential: A sequential neural network model.
    """
    L_default, M_default = 4, 10
    if use_fourier:
        if fourier_params is None:
            if domain_dim == 1:
                fourier_params = {'L' : [L_default],
                              'M' : [M_default]}
            else:
                fourier_params = {'L' : [L_default] + [None,] * (domain_dim - 1), 
                              'M' : [M_default] + [None,] * (domain_dim - 1)}
        fourier_params['device'] = device
        four_emb = Fourier_embedding(**fourier_params)
        if device == 'cuda':
            four_emb = four_emb.cuda()
        net_default = torch.nn.ModuleList([four_emb,])
    else:
        net_default = torch.nn.ModuleList([])
    linear_inputs = net_default[0].out_features if use_fourier else domain_dim
    
    if domain_dim == 1:            
        hidden_neurons = 128 # 64 #
    else:
        hidden_neurons = 112 # 54 #

    operators = net_default + torch.nn.ModuleList([torch.nn.Linear(linear_inputs, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, equations_num, device=device)])
    return torch.nn.Sequential(*operators)

def exp_form(a, sign_num: int = 4):
    """
    Expresses a number in exponential form for equation simplification.
    
        This method decomposes a number into its normalized form and exponent.
        The normalized form is the number divided by 10 raised to the power of the exponent,
        and is rounded to a specified number of significant digits. This is useful for
        representing coefficients and terms within differential equations in a consistent
        and comparable manner, aiding in the discovery process.
    
        Args:
            a: The number to express in exponential form.
            sign_num: The number of significant digits to round the normalized form to. Defaults to 4.
    
        Returns:
            A tuple containing:
              - The normalized form of the number, rounded to `sign_num` significant digits.
              - The exponent of the number (base 10).
    """
    if np.isclose(a, 0):
        return 0.0, 0
    exp = np.floor(np.log10(np.abs(a)))
    return np.around(a / 10**exp, sign_num), int(exp)


def rts(value, sign_num: int = 5):
    """
    Round the input value to a specified number of significant digits.
    
        This ensures that numerical values are represented with a consistent level of precision, 
        facilitating comparison and reducing the impact of insignificant variations when 
        identifying underlying equation structures.
    
        Args:
            value (float): The numerical value to be rounded.
            sign_num (int, optional): The number of significant digits to retain. Defaults to 5.
    
        Returns:
            float: The rounded numerical value.
    """
    if value == 0:
        return 0
    magn_top = np.log10(value)
    idx = -(np.sign(magn_top)*np.ceil(np.abs(magn_top)) - sign_num)
    if idx - sign_num > 1:
        idx -= 1
    return np.around(value, int(idx))


def train_ann(args: list, data: np.ndarray, epochs_max: int = 500, batch_frac = 0.5, 
              dim = None, model = None, device = 'cpu'):
    """
    Trains an artificial neural network (ANN) model to approximate a given dataset.
    
        This method refines the ANN model to accurately represent the underlying patterns
        within the data. By adjusting model architecture, training parameters, and device
        usage, it optimizes the model's ability to capture the relationships present in the data.
        This is a crucial step in creating a surrogate model that accurately reflects the
        behavior of the system being studied.
    
        Args:
            args: A list of arguments representing the grid coordinates of the data.
            data: A NumPy array containing the data to be approximated.
            epochs_max: The maximum number of training epochs.
            batch_frac: The fraction of the data to use for each batch.
            dim: The dimensionality of the data. If None, it is inferred from the data shape.
            model: A PyTorch model to be trained. If None, a default model is created.
            device: The device to use for training (e.g., 'cpu' or 'cuda').
    
        Returns:
            The best trained PyTorch model based on the minimum loss achieved during training.
    """
    if dim is None:
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim
    # assert len(args) == dim, 'Dimensionality of data does not match with passed grids.'
    data_size = data.size
    if model is None:
        model = torch.nn.Sequential(
                                    torch.nn.Linear(dim, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 64, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(64, 1024, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(1024, 1, device=device)
                                    )
    
    model.to(device)
    data_grid = np.stack([arg.reshape(-1) for arg in args])
    grid_tensor = torch.from_numpy(data_grid).float().T.to(device)
    # grid_tensor.to(device)
    data = torch.from_numpy(data.reshape(-1, 1)).float().to(device)
    # print(data.size)
    # data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = int(data_size * batch_frac)

    t = 0

    print('grid_flattened.shape', grid_tensor.shape, 'field.shape', data.shape)

    loss_mean = 1000
    min_loss = np.inf
    losses = []
    while loss_mean > 2e-3 and t < epochs_max:

        permutation = torch.randperm(grid_tensor.size()[0])

        loss_list = []

        for i in range(0, grid_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid_tensor[indices], data[indices]
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean = np.mean(loss_list)
        if loss_mean < min_loss:
            best_model = model
            min_loss = loss_mean
        losses.append(loss_mean)
        # if global_var.verbose.show_ann_loss:
        #     print('Surface training t={}, loss={}'.format(t, loss_mean))
        t += 1
    print_loss = True
    if print_loss:
        plt.plot(losses)
        plt.grid()
        plt.show()
    return best_model

def use_ann_to_predict(model, recalc_grids: list):
    """
    Uses a pre-trained ANN model to predict values on given spatial grids.
    
        This method leverages a trained artificial neural network (ANN) to estimate values
        across spatial grids. It prepares the grid data by reshaping and converting it
        into a suitable format for the ANN model, performs the prediction, and then
        restores the output to the original grid dimensions. This is a crucial step in
        approximating the solution space of the discovered differential equations.
    
        Args:
            model: The pre-trained ANN model to use for prediction.
            recalc_grids: A list of NumPy arrays representing the spatial grids for which
                predictions are to be made. These grids define the domain over which the
                solution is approximated.
    
        Returns:
            np.ndarray: A NumPy array containing the predicted values, reshaped to
            match the shape of the input grids. These values represent the ANN's
            approximation of the solution on the given spatial domain.
    """
    data_grid = np.stack([grid.reshape(-1) for grid in recalc_grids])
    recalc_grid_tensor = torch.from_numpy(data_grid).float().T
    recalc_grid_tensor = recalc_grid_tensor #.to(device)

    return model(recalc_grid_tensor).detach().numpy().reshape(recalc_grids[0].shape)

def flatten(obj):
    """
    Transforms a list of elements into a flat list of lists.
    
        This function ensures that each element within the input list is itself a list,
        converting non-list elements into single-element lists before concatenating
        all sublists into a single, flattened list. This is a preliminary step for
        further processing, ensuring data compatibility for subsequent equation discovery.
    
        Args:
            obj (list): The list to be flattened. Each element should ideally be a list
                or convertible to a list.
    
        Returns:
            list: A flattened list containing all elements from the original list,
                with each original element now residing within its own sublist.
    """
    assert type(obj) == list

    for idx, elem in enumerate(obj):
        if not isinstance(elem, (list, tuple)):
            obj[idx] = [elem,]
    return reduce(lambda x, y: x+y, obj)

def factor_params_to_str(factor, set_default_power=False, power_idx=0):
    """
    Converts factor parameters to a string representation for equation building.
    
        This method prepares the parameters of a factor within a potential differential equation
        for representation as a string. It retrieves the factor's parameters and label,
        optionally setting a specific parameter to a default value of 1. This is useful
        when exploring different equation structures where certain terms might be temporarily
        disabled or set to a neutral value during the evolutionary search process.
    
        Args:
            factor: The factor object whose parameters are to be converted.
            set_default_power: A boolean indicating whether to set a default
                value for a specific parameter. Defaults to False.
            power_idx: The index of the parameter to set to the default value
                if `set_default_power` is True. Defaults to 0.
    
        Returns:
            A tuple containing the factor's label and a tuple of its parameters.
    """
    param_label = np.copy(factor.params)
    if set_default_power:
        param_label[power_idx] = 1.
    return (factor.label, tuple(param_label))

def form_label(x, y):
    """
    Forms a descriptive label by combining a base string with a component's identifier.
    
    This function constructs a label that represents a combination of terms within a differential equation.
    It's used to create human-readable representations of equation components during the equation discovery process.
    The label is formed by concatenating a base string `x` with the `cache_label` of a component `y`,
    inserting " * " if the base string is not empty.
    
    Args:
        x (str): The base string, potentially representing a combination of terms.
        y: An object with a 'cache_label' attribute (string) representing a component of the equation.
    
    Returns:
        str: The combined label string.
    """
    print(type(x), type(y.cache_label))
    return x + ' * ' + y.cache_label if len(x) > 0 else x + y.cache_label

def detect_similar_terms(base_equation_1, base_equation_2):   # Переделать!
    same_terms_from_eq1 = []
    same_terms_from_eq2 = []
    eq2_processed = np.full(
        shape=len(base_equation_2.structure), fill_value=False)

    similar_terms_from_eq1 = []
    similar_terms_from_eq2 = []

    different_terms_from_eq1 = []
    different_terms_from_eq2 = []
    for eq1_term in base_equation_1.structure:
    """
    Detects and categorizes corresponding terms between two base equations.
    
    This method aligns terms from two base equations, classifying them as 'same',
    'similar', or 'different' based on their structural and label similarities.
    This alignment is crucial for identifying shared components and variations
    between different equation representations of the same underlying phenomenon.
    
    Args:
        base_equation_1: The first base equation to compare, represented as a structured object.
        base_equation_2: The second base equation to compare, represented as a structured object.
    
    Returns:
        tuple: A tuple containing two lists. The first list represents terms from
            `base_equation_1` and the second represents terms from `base_equation_2`.
            Each list contains three sub-lists:
            - same_terms: Terms that are identical in both structure and labels.
            - similar_terms: Terms that share the same labels but may differ in structure.
            - different_terms: Terms that are unique to the respective equation.
    """
        found_similar = False
        for idx, eq2_term in enumerate(base_equation_2.structure):
            if eq1_term == eq2_term and not eq2_processed[idx]:
                found_similar = True
                same_terms_from_eq1.append(eq1_term)
                same_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
                break
            elif ({token.label for token in eq1_term.structure} == {token.label for token in eq2_term.structure} and
                  len(eq1_term.structure) == len(eq2_term.structure) and not eq2_processed[idx]):
                found_similar = True
                similar_terms_from_eq1.append(eq1_term)
                similar_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
                break
        if not found_similar:
            different_terms_from_eq1.append(eq1_term)

    for idx, elem in enumerate(eq2_processed):
        if not elem:
            different_terms_from_eq2.append(base_equation_2.structure[idx])

    assert len(same_terms_from_eq1) + len(similar_terms_from_eq1) + \
        len(different_terms_from_eq1) == len(base_equation_1.structure)
    assert len(same_terms_from_eq2) + len(similar_terms_from_eq2) + \
        len(different_terms_from_eq2) == len(base_equation_2.structure)
    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]


def filter_powers(gene):
    """
    Filters a gene to refine the representation of equation terms.
    
        This method aggregates the 'power' parameter of tokens within a gene that
        exhibit partial equality. By summing the powers of similar tokens and
        capping the result at a maximum value, it ensures that the gene
        representation remains concise and avoids over-emphasizing redundant terms
        in the equation. This process helps to simplify the equation structure
        and improve the overall interpretability of the discovered model.
    
        Args:
            gene: A list of tokens representing a gene. Each token is expected
                to have a `partial_equlaity` method and a `params` attribute,
                where `params` is a list of parameter values and
                `params_description` is a dictionary describing the parameters.
                Each parameter description should have 'name' and 'bounds' keys.
    
        Returns:
            A list of tokens representing the filtered gene, where each token's
            'power' parameter has been updated based on the total power of
            partially equal tokens in the original gene.
    """
    gene_filtered = []

    for token_idx in range(len(gene)):
        total_power = sum([factor.param(name = 'power') for factor in gene 
                           if gene[token_idx].partial_equlaity(factor)])#gene.count(gene[token_idx])
        powered_token = copy.deepcopy(gene[token_idx])
        
        power_idx = np.inf
        for param_idx, param_info in powered_token.params_description.items():
            if param_info['name'] == 'power':
                max_power = param_info['bounds'][1]
                power_idx = param_idx
                break
        powered_token.params[power_idx] = total_power if total_power < max_power else max_power
        if powered_token not in gene_filtered:
            gene_filtered.append(powered_token)
    return gene_filtered


def define_derivatives(var_name='u', dimensionality=1, max_order=2):
    """
    Generates derivative keys and corresponding derivative orders up to a specified order for each dimension.
    
    This function is crucial for constructing the search space of potential differential equations. 
    By systematically generating derivative keys, the algorithm can explore various combinations of derivatives 
    to identify the equation that best describes the underlying dynamics of the system.
    
    Args:
        var_name (`str`): Name of the dependent variable. Defaults to 'u'.
        dimensionality (`int`): Dimensionality of the data. Defaults to 1.
        max_order (`int` | `list`): Maximum order of derivative. If an integer, the same maximum order is applied to all dimensions. 
            If a list, each element specifies the maximum order for the corresponding dimension. Defaults to 2.
    
    Returns:
        `tuple`: A tuple containing two lists:
            - `deriv_names` (`list` of `str`): Keys representing the derivative terms (e.g., 'du/dx0', 'd^2u/dx1^2').
            - `var_deriv_orders` (`list` of `list` of `int`): Keys for accessing the derivatives in numerical solvers. Each sublist indicates the variable index and the order of differentiation with respect to that variable (e.g., `[[0], [0, 0]]` for du/dx0 and d^2u/dx0^2).
    """
    deriv_names = []
    var_deriv_orders = []
    if isinstance(max_order, int):
        max_order = [max_order for dim in range(dimensionality)]
    for var_idx in range(dimensionality):
        for order in range(max_order[var_idx]):
            var_deriv_orders.append([var_idx,] * (order+1))
            if order == 0:
                deriv_names.append('d' + var_name + '/dx' + str(var_idx))
            else:
                deriv_names.append(
                    'd^'+str(order+1) + var_name + '/dx'+str(var_idx)+'^'+str(order+1))
    print('Deriv orders after definition', var_deriv_orders)
    return deriv_names, var_deriv_orders


def population_sort(input_population):
    """
    Sorts the population to prioritize well-performing individuals.
    
        This function arranges the population based on the fitness of each
        individual, ensuring that those with higher fitness values are placed
        earlier in the sorted list. This is a crucial step in the evolutionary
        process, as it allows the algorithm to focus on the most promising
        candidates for equation discovery.
    
        Args:
            input_population: A list of individuals representing the population.
    
        Returns:
            A new list containing the individuals from the input population,
            sorted in descending order of fitness value.
    """
    individ_fitvals = [
        individual.fitness_value if individual.fitness_calculated else 0 for individual in input_population]
    pop_sorted = [x for x, _ in sorted(
        zip(input_population, individ_fitvals), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))


def normalize_ts(Input):
    """
    Normalizes a time series matrix by subtracting the mean and dividing by the standard deviation.
    
        This normalization ensures that each time series has a zero mean and unit variance,
        which is crucial for algorithms that are sensitive to the scale of the input data.
        If a time series has a standard deviation of zero, it is set to a constant value of 1 to avoid division by zero and to represent a stable, unchanging signal.
    
        Args:
            Input (np.ndarray): The input time series data. It can be a 1D or 2D numpy array.
    
        Returns:
            np.ndarray: The normalized time series data. If the input is a 1D array, it returns the same array.
                If the input is a 2D array, it returns a normalized 2D array.
    
        Raises:
            ValueError: If the input data has 0 dimensions.
    """
    matrix = np.copy(Input)
    if np.ndim(matrix) == 0:
        raise ValueError(
            
            'Incorrect input to the normalizaton: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return matrix
    else:
        for i in np.arange(matrix.shape[0]):
            std = np.std(matrix[i])
            if std != 0:
                matrix[i] = (matrix[i] - np.mean(matrix[i])) / std
            else:
                matrix[i] = 1
        return matrix

def minmax_normalize(matrix):
    """
    Apply min-max normalization to the input matrix to ensure consistent scaling of data ranges, which is crucial for the equation discovery process. Normalization helps to prevent features with larger values from dominating the search for governing equations.
    
        Args:
            matrix (numpy.ndarray): The input matrix to be normalized.
    
        Returns:
            numpy.ndarray: The normalized matrix. For 1D arrays, the original array is returned. For 2D+ arrays, each row is normalized to the [0, 1] range.
    """
    matrix = np.copy(matrix)

    if np.ndim(matrix) == 0:
        raise ValueError('Incorrect input to the normalization: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return matrix
    else:
        domain_min = np.min(matrix)
        domain_max = np.max(matrix)
        domain_mean = np.mean(matrix)
        if domain_max != domain_min:
            matrix = (matrix - domain_mean - domain_min) / (domain_max - domain_min)
        # for i in np.arange(matrix.shape[0]):
        #     row_min = np.min(matrix[i])
        #     row_max = np.max(matrix[i])
        #
        #     # Only normalize if the row has variation
        #     if domain_max != domain_min:
        #         matrix[i] = (matrix[i] - domain_mean - domain_min) / (domain_max - domain_min)
        #     else:
        #         # If all values are the same, set to 0.5 or keep original (0.5 is midpoint)
        #         matrix[i] = 0.5

        return matrix
