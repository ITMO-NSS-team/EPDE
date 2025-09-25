#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:41:07 2021

@author: mike_ubuntu
"""

import numpy as np
import torch
# device = torch.device('cpu')

from abc import ABC, abstractmethod
from typing import Callable, Union, List

import epde.globals as global_var
from epde.supplementary import factor_params_to_str

class EvaluatorTemplate(ABC):
    """
    An abstract base class for evaluators.
    
        This class defines the basic structure for evaluators,
        providing a call method that must be implemented by subclasses.
    
        Class Methods:
        - __call__: Call method for the abstract class.
    """

    def __init__(self):
        """
        Initializes the evaluator.
        
                This method prepares the evaluator for assessing the fitness of candidate equation structures.
                It sets up the necessary internal state, such as data handling configurations or result storage,
                required for subsequent evaluation steps.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        pass

    @abstractmethod
    def __call__(self, factor, structural: bool = False, grids: list = None, 
                 torch_mode: bool = False, **kwargs):
        """
        Abstract call method for evaluating equation fitness.
        
        This method must be implemented by subclasses to define how the fitness
        of a candidate equation is evaluated based on the provided data and
        settings. It raises a NotImplementedError if called directly, ensuring
        that concrete evaluator classes provide their own implementation.
        
        Args:
          factor: The factor to be used in the evaluation.
          structural: A boolean indicating whether the method should consider
            structural properties during evaluation. Defaults to False.
          grids: A list of grids to be used for evaluation. Defaults to None.
          torch_mode: A boolean indicating whether to use torch mode for
            computation. Defaults to False.
          **kwargs: Additional keyword arguments to be passed to the
            evaluation function.
        
        Returns:
          None.  Subclasses should return a fitness score or a tuple of scores.
        
        Raises:
          NotImplementedError: Always raised, indicating that the method
            must be implemented by a subclass.
        
        WHY: This abstract method enforces that each specific evaluation strategy
        (implemented in subclasses) defines its own way of calculating the
        fitness of a candidate equation, which is a crucial step in the
        equation discovery process.
        """
        raise NotImplementedError(
            'Trying to call the method of an abstract class')


class CustomEvaluator(EvaluatorTemplate):
    """
    A custom evaluator class for NumPy and Torch evaluation functions.
    
        This class allows users to define and apply custom evaluation functions,
        supporting both NumPy and Torch implementations. It can handle single
        evaluation functions or dictionaries of functions, and it provides a
        flexible interface for evaluating functions with different parameters.
    
        Class Methods:
        - __init__
        - __call__
    
        Attributes:
          _evaluation_functions_np: NumPy evaluation function(s).
          _evaluation_functions_torch: Torch evaluation function(s).
          _single_function_token: A boolean token indicating whether a single evaluation function or a dictionary of functions is used.
          eval_fun_params_labels: Labels for the evaluation function parameters.
    """

    def __init__(self, evaluation_functions_np: Union[Callable, dict] = None, 
                 evaluation_functions_torch: Union[Callable, dict] = None,
                 eval_fun_params_labels: Union[list, tuple, set] = ['power']):
        """
        Initializes the CustomEvaluator with NumPy and Torch evaluation functions.
        
                This setup is crucial for assessing the fitness of candidate equation structures generated during the evolutionary search. The evaluator determines how well a given equation matches the observed data, guiding the optimization process towards more accurate and representative models.
        
                Args:
                  evaluation_functions_np: NumPy evaluation function(s). Can be a single callable or a dictionary of callables.
                  evaluation_functions_torch: Torch evaluation function(s). Can be a single callable or a dictionary of callables.
                  eval_fun_params_labels: Labels for the evaluation function parameters.
        
                Raises:
                  ValueError: If both `evaluation_functions_np` and `evaluation_functions_torch` are None, as at least one evaluation function is required to assess equation fitness.
        
                Returns:
                  None
        
                Class Fields Initialized:
                  _evaluation_functions_np (Union[Callable, dict]): NumPy evaluation function(s).
                  _evaluation_functions_torch (Union[Callable, dict]): Torch evaluation function(s).
                  _single_function_token (bool): A boolean token indicating whether a single evaluation function or a dictionary of functions is used.
                  eval_fun_params_labels (Union[list, tuple, set]): Labels for the evaluation function parameters.
        """
        self._evaluation_functions_np = evaluation_functions_np
        self._evaluation_functions_torch = evaluation_functions_torch

        if (evaluation_functions_np is None) and (evaluation_functions_torch is None):
            raise ValueError('No evaluation function set in the initialization of CustomEvaluator.')

        if isinstance(evaluation_functions_np, dict):
            self._single_function_token = False
        else:
            self._single_function_token = True

        self.eval_fun_params_labels = eval_fun_params_labels

    def __call__(self, factor, structural: bool = False, func_args: List[Union[torch.Tensor, np.ndarray]] = None, 
                 torch_mode: bool = False, **kwargs): # s
        if torch_mode: # TODO: rewrite
            torch_mode_explicit = True
        if not self._single_function_token and factor.label not in self._evaluation_functions_np.keys():
        """
        Evaluates the function associated with a given factor, using either NumPy or Torch implementations.
        
                This method selects the appropriate evaluation function based on the provided arguments and the evaluator's configuration.
                It then applies this function to the provided arguments, using either pre-computed grids or a new set of arguments.
                This evaluation is a core step in exploring the search space of potential equation structures.
        
                Args:
                    factor: The factor object containing the function label, parameters, and grids.
                    structural: A boolean flag indicating whether the evaluation is structural. Defaults to False.
                    func_args: A list of arguments to be passed to the evaluation function.
                        If None, the factor's grids are used. Defaults to None.
                    torch_mode: A boolean flag indicating whether to use the Torch evaluation
                        functions. Defaults to False.
                    **kwargs: Additional keyword arguments.
        
                Returns:
                    np.ndarray: The result of evaluating the function on the given arguments.
        
                WHY: This method is used to calculate the value of a symbolic expression (factor) for a given set of input values.
                The result of this evaluation is then used to assess the fitness of the expression in representing the underlying dynamics of the system.
        """
            raise KeyError(
                'The label of the token function does not match keys of the evaluator functions')
        if func_args is not None:
            if isinstance(func_args[0], np.ndarray) or self._evaluation_functions_torch is None:
                funcs = self._evaluation_functions_np if self._single_function_token else self._evaluation_functions_np[factor.label]
            elif isinstance(func_args[0], torch.Tensor) or self._evaluation_functions_np is None or torch_mode_explicit:
                funcs = self._evaluation_functions_torch if self._single_function_token else self._evaluation_functions_torch[factor.label]
        elif torch_mode:
            funcs = self._evaluation_functions_torch if self._single_function_token else self._evaluation_functions_torch[factor.label]
        else:
            funcs = self._evaluation_functions_np if self._single_function_token else self._evaluation_functions_np[factor.label]

        eval_fun_kwargs = dict()
        for key in self.eval_fun_params_labels:
            for param_idx, param_descr in factor.params_description.items():
                if param_descr['name'] == key:
                    eval_fun_kwargs[key] = factor.params[param_idx]

        grid_function = np.vectorize(lambda args: funcs(*args, **eval_fun_kwargs))

        if func_args is None:
            new_grid = False
            func_args = factor.grids
        else:
            new_grid = True
        try:
            if new_grid:
                raise AttributeError
            self.indexes_vect
        except AttributeError:
            self.indexes_vect = np.empty_like(func_args[0], dtype=object)
            for tensor_idx, _ in np.ndenumerate(func_args[0]):
                self.indexes_vect[tensor_idx] = tuple([subarg[tensor_idx]
                                                       for subarg in func_args])
        value = grid_function(self.indexes_vect)
        return value


def simple_function_evaluator(factor, structural: bool = False, grids=None, 
                              torch_mode: bool = False, **kwargs):
    """
    Evaluates the value of a factor, potentially retrieving it from a cache. This is useful for reusing previously computed values of equation terms, such as derivatives or coordinates, during the equation discovery process. The method prioritizes using pre-computed values to enhance computational efficiency.
    
        Args:
            factor (epde.factor.Factor): The factor object representing the term to evaluate.
            structural (bool, optional): Indicates whether the value is used for structural discovery (True) or coefficient calculation (False). Defaults to False.
            grids (optional): Grid values for prediction with an ANN, if applicable. Defaults to None.
            torch_mode (bool, optional): Flag indicating whether to use torch tensors. Defaults to False.
            **kwargs: Additional keyword arguments.
    
        Returns:
            numpy.ndarray: The evaluated value of the factor, which can be used as a target or feature in regression.
    """

    for param_idx, param_descr in factor.params_description.items():
        if param_descr['name'] == 'power':
            power_param_idx = param_idx
        
    if grids is not None:
        value = factor.predict_with_ann(grids)
        value = value**(factor.params[power_param_idx])

        return value

    else:
        if factor.params[power_param_idx] == 1:
            value = global_var.tensor_cache.get(factor.cache_label, structural = structural, torch_mode = torch_mode)
            return value
        else:
            value = global_var.tensor_cache.get(factor_params_to_str(factor, set_default_power = True,
                                                                     power_idx = power_param_idx),
                                                structural = structural, torch_mode = torch_mode)
            value = value**(factor.params[power_param_idx])
            return value


sign_eval_fun_np = lambda *args, **kwargs: np.sign(args[0]) # If dim argument is needed here: int(kwargs['dim'])
sign_eval_fun_torch = lambda *args, **kwargs: torch.sign(args[0])

trig_eval_fun_np = {'cos': lambda *grids, **kwargs: np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
                    'sin': lambda *grids, **kwargs: np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}

trig_eval_fun_torch = {'cos': lambda *grids, **kwargs: torch.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
                       'sin': lambda *grids, **kwargs: torch.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}

inverse_eval_fun_np = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power'])
inverse_eval_fun_torch = lambda *grids, **kwargs: torch.pow(grids[int(kwargs['dim'])], - kwargs['power'])

grid_eval_fun_np = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power'])
grid_eval_fun_torch = lambda *grids, **kwargs: torch.pow(grids[int(kwargs['dim'])], kwargs['power'])

def phased_sine_np(*grids, **kwargs):
    """
    Generates a phased sine wave using NumPy.
    
    This method constructs a basis function consisting of a phased sine wave,
    parameterized by frequency, phase, and power. These basis functions are
    essential components for representing more complex functions within the
    equation discovery process. By combining multiple phased sine waves with
    different parameters, the framework can approximate a wide range of
    potential solutions to the differential equation being sought.
    
    Args:
      *grids: Variable number of grid arrays, one for each dimension. These grids
        define the coordinate system on which the sine wave is evaluated.
      **kwargs: Keyword arguments containing 'freq' (frequencies for each dimension),
        'phase' (phases for each dimension), and 'power' (the exponent to raise
        the sine wave to). These parameters control the shape and characteristics
        of the generated sine wave.
    
    Returns:
      np.ndarray: A NumPy array representing the phased sine wave. This array
        can then be used as a building block in the equation discovery process.
    """
    coordwise_elems = [kwargs['freq'][dim] * 2*np.pi*(grids[dim] + kwargs['phase'][dim]) 
                       for dim in range(len(grids))]
    return np.power(np.sin(np.sum(coordwise_elems, axis = 0)), kwargs['power'])

def phased_sine_torch(*grids, **kwargs):
    """
    Generates a coordinate-wise phased sine wave using PyTorch.
    
        This method constructs a sine wave by combining coordinate-wise elements,
        modulated by specified frequencies and phases, and then applies a power exponent.
        This is useful for creating basis functions with controlled spatial variations,
        allowing the discovery process to represent complex functions as combinations of simpler,
        spatially-varying sinusoidal components.
    
        Args:
            *grids: Variable number of grid tensors, one for each dimension. These grids
                define the spatial coordinates over which the sine wave is generated.
            **kwargs: Keyword arguments containing:
                - 'freq' (torch.Tensor): Frequencies for each dimension.
                - 'phase' (torch.Tensor): Phases for each dimension.
                - 'power' (float): The exponent to raise the sine to.
    
        Returns:
            torch.Tensor: The resulting phased sine wave tensor. This tensor represents
                the function value at each point defined by the input grids.
    """
    coordwise_elems = [kwargs['freq'][dim] * 2*torch.pi*(grids[dim] + kwargs['phase'][dim]) 
                       for dim in range(len(grids))]
    return torch.pow(torch.sin(torch.sum(coordwise_elems, axis = 0)), kwargs['power'])    

def phased_sine_1d_np(*grids, **kwargs):
    """
    Generates a 1D phased sine wave using NumPy to represent a basis function.
        
        This function constructs a 1D sine wave, a fundamental component in representing more complex functions within the equation search space.
        The sine wave is defined by its frequency, phase, and power, allowing for flexible adaptation to different equation structures.
        
        Args:
            *grids: A tuple containing the grid along which the sine wave is calculated.
            **kwargs: A dictionary containing the keyword arguments:
                freq (float): The frequency of the sine wave.
                phase (float): The phase of the sine wave.
                power (float): The power to which the sine wave is raised.
        
        Returns:
            np.ndarray: A NumPy array representing the 1D phased sine wave.
    """
    coordwise_elems = kwargs['freq'] * 2*np.pi*(grids[0] + kwargs['phase']/kwargs['freq']) 
    return np.power(np.sin(coordwise_elems), kwargs['power'])

def phased_sine_1d_torch(*grids, **kwargs):
    """
    Computes a 1D phased sine wave using PyTorch to represent a component within a larger equation.
    
        This method generates a 1D sine wave with adjustable frequency, phase, and power,
        serving as a fundamental building block for constructing more complex equation terms.
        By combining such elements, the framework can explore a wide range of potential equation structures.
    
        Args:
            *grids: A tuple containing a single PyTorch tensor representing the 1D grid coordinates.
            **kwargs: A dictionary containing the keyword arguments:
                freq: The frequency of the sine wave.
                phase: The phase of the sine wave.
                power: The power to which the sine wave is raised.
    
        Returns:
            torch.Tensor: A PyTorch tensor containing the 1D phased sine wave.
    """
    coordwise_elems = kwargs['freq'] * 2*torch.pi*(grids[0] + kwargs['phase']/kwargs['freq']) 
    return torch.pow(torch.sin(coordwise_elems), kwargs['power'])

def const_eval_fun_np(*grids, **kwargs):
    """
    Evaluates a constant function, creating a grid filled with a specified value.
    
        This function is used to initialize or set a baseline value across the entire
        computational domain when constructing more complex equation structures.
        It leverages NumPy for efficient array creation.
    
        Args:
            *grids: One or more NumPy arrays representing the grid(s). The first grid
                is used to determine the shape and data type of the output array.
            **kwargs: Keyword arguments. Must contain the 'value' key, which
                specifies the constant value to fill the array with.
    
        Returns:
            np.ndarray: A NumPy array with the same shape and data type as the
                first input grid, filled with the specified constant value.
    """
    return np.full_like(a=grids[0], fill_value=kwargs['value'])

def const_eval_fun_torch(*grids, **kwargs):
    """
    Evaluates a constant function using PyTorch to provide a foundational building block for more complex equation discovery.
    
        This function generates a tensor of constant values, matching the shape and data type of the input grid. This is useful for creating constant terms within candidate differential equations during the equation search process.
    
        Args:
          *grids: Variable number of input grids (tensors). The shape and
            data type of the first grid are used to create the output tensor.
          **kwargs: Keyword arguments. Must contain the 'value' key, which
            specifies the constant value to fill the output tensor with.
    
        Returns:
          torch.Tensor: A tensor with the same shape and data type as the
            first input grid, filled with the specified constant value.
    """
    return torch.full_like(a=grids[0], fill_value=kwargs['value'])    

def const_grad_fun_np(*grids, **kwargs):
    """
    Returns a zero-filled NumPy array with the same shape and data type as the first input grid.
    
        This function is designed to provide a constant gradient of zero, which can be useful as a baseline or a starting point
        when searching for more complex gradient functions within the equation discovery process. By providing a zero gradient,
        it allows the evolutionary algorithm to explore other potential terms and structures in the differential equation.
    
        Args:
          *grids: One or more NumPy arrays representing the input grids. Only the first grid's shape and data type are used to create the zero-filled array.
          **kwargs: Arbitrary keyword arguments. These are not used in the function.
    
        Returns:
          np.ndarray: A NumPy array filled with zeros, having the same shape and data type as the first input grid.
    """
    return np.zeros_like(a=grids[0])

def const_grad_fun_torch(*grids, **kwargs):
    """
    Creates a zero-filled tensor matching the shape of the input grid.
    
    This function is used to initialize gradient tensors during the equation discovery process.
    By providing a zero-filled tensor with the correct dimensions, it ensures that the optimization
    process starts with a clean slate for gradient calculations, avoiding any potential bias
    from pre-existing values.
    
    Args:
      *grids: One or more input tensors. The shape and data type of the first
        tensor in `grids` are used to create the output tensor of zeros.
      **kwargs: Additional keyword arguments. These are not used in the function.
    
    Returns:
      torch.Tensor: A tensor of zeros with the same shape and data type as the
        first input tensor in `grids`.
    """
    return torch.zeros_like(a=grids[0])

def get_velocity_common(*grids, **kwargs):
    """
    Calculates velocity components alpha and beta based on input grids and polynomial coefficients.
    
        These components are crucial for modeling the dynamics of the system by representing velocity profiles
        derived from the identified differential equation. The method employs polynomial and exponential functions
        to map the input grids to velocity components, using coefficients optimized during the equation discovery process.
    
        Args:
            *grids: Variable number of grid arrays. The first grid is used in polynomial calculations,
                and the second grid is used in exponential and polynomial calculations.
            **kwargs: Keyword arguments representing polynomial coefficients (p1 to p15) obtained during the equation discovery process.
                These coefficients fine-tune the velocity component calculations.
    
        Returns:
            tuple: A tuple containing two arrays, alpha and beta, representing the calculated velocity components.
            These components are essential outputs for further analysis or simulation within the identified model.
    """
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return alpha, beta

def velocity_heating_eval_fun(*grids, **kwargs):
    """
    Calculates the velocity field based on spatial grids, which is used to model convection in heat transfer problems. The velocity field is a crucial component in simulating how heat is transported by the movement of a fluid or gas.
    
        Args:
            *grids: Spatial grids representing the domain of the heat equation.
            **kwargs: Additional keyword arguments passed to `get_velocity_common`.
    
        Returns:
            The calculated velocity field as a product of two intermediate fields, `alpha` and `beta`. This field is then used to simulate the convective heat transfer.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha * beta

def vhef_grad_1(*grids, **kwargs):
    """
    Computes a gradient-like expression used to evaluate candidate differential equations.
    
        This function calculates a weighted product of input grids, using velocity parameters
        (alpha and beta) to refine the expression. These parameters, derived from the input grids,
        act as coefficients that scale the contribution of each grid to the overall gradient.
        This gradient approximation is used within the evolutionary algorithm to assess how well
        a candidate equation fits the observed data.
    
        Args:
            *grids: Variable number of grid-like arrays representing different terms in the equation.
            **kwargs: Keyword arguments to be passed to `get_velocity_common` for velocity parameter calculation.
    
        Returns:
            The computed gradient-like value, representing a component of the overall equation's gradient.
            This value is calculated as the product of the square of the first grid, the second grid, and the
            alpha and beta velocity parameters. This result contributes to the fitness evaluation of the
            candidate equation.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1] * alpha * beta

def vhef_grad_2(*grids, **kwargs):
    """
    Computes a component of the equation search gradient using velocity fields.
    
    This function calculates a term contributing to the overall gradient used in the equation discovery process. It leverages velocity fields derived from the input grids to weight the product of the first two grids. This weighting helps to emphasize regions where the velocity fields indicate significant changes or flows, thereby guiding the search towards equation terms that capture these dynamics.
    
    Args:
        *grids: Variable number of grid arrays representing different state variables or spatial coordinates. These grids are used to compute velocity components and contribute to the final result.
        **kwargs: Keyword arguments passed to the `get_velocity_common` function, influencing the velocity field calculation.
    
    Returns:
        The weighted product of the first two grids, `alpha`, and `beta`, representing a component of the overall gradient.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]  * grids[1] * alpha * beta

def vhef_grad_3(*grids, **kwargs):
    """
    Calculates a component of the gradient used in the equation discovery process.
    
        This function computes a gradient term by scaling the second input grid
        with velocity components `alpha` and `beta`. These velocity components
        are derived from the input grids, representing potential relationships
        between variables, and additional keyword arguments. The resulting gradient
        contributes to the overall fitness evaluation of candidate equations.
    
        Args:
            *grids: Variable number of grid arrays representing different terms in the equation.
            **kwargs: Keyword arguments passed to the `get_velocity_common` function,
                potentially influencing the calculation of velocity components.
    
        Returns:
            numpy.ndarray: The calculated gradient, representing a component of the
                overall equation gradient, obtained by element-wise multiplication
                of the second grid, `alpha`, and `beta`. This gradient is used to
                evaluate the fitness of candidate equations within the EPDE framework.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1] * alpha * beta

def vhef_grad_4(*grids, **kwargs):
    """
    Computes a gradient-related term for equation discovery.
    
        This method calculates a component used in the equation discovery process,
        specifically a term resembling a gradient. It utilizes input grids,
        representing potential variables or features, and combines them using
        coefficients derived from `get_velocity_common`. This term contributes
        to the overall fitness evaluation of candidate equations.
    
        Args:
            *grids: Variable number of grid-like arguments. These represent the
                spatial or temporal data upon which the equation is being discovered.
            **kwargs: Keyword arguments passed to `get_velocity_common`. These
                arguments control aspects of the coefficient calculation, influencing
                the weighting of different terms in the equation.
    
        Returns:
            The computed gradient-related value. This value represents a component
            of a potential term in the discovered equation, calculated as the square
            of the first grid multiplied by the alpha and beta values obtained from
            `get_velocity_common`. This value contributes to the overall equation's
            representation and fitness.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * alpha * beta

def vhef_grad_5(*grids, **kwargs):
    """
    Calculates a gradient used in the equation discovery process.
    
        This method refines the search for governing equations by calculating a gradient
        based on input grids and shared velocity components. This gradient guides the
        evolutionary algorithm towards equation structures that better fit the data.
        
        Args:
            *grids: Variable number of grid arrays representing different data fields.
            **kwargs: Keyword arguments passed to `get_velocity_common` to control
                the computation of shared velocity components.
        
        Returns:
            The calculated gradient, which is the product of the first grid, alpha,
            and beta. This gradient is used to evaluate and refine candidate equations.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * alpha * beta

def vhef_grad_6(*grids, **kwargs):
    """
    Computes a scalar value reflecting the interplay between different velocity field characteristics.
    
        This function leverages velocity grid data to compute two intermediate values, `alpha` and `beta`, using `get_velocity_common`. The product of these values provides a combined measure reflecting the relationships within the velocity fields. This can be used to quantify certain aspects of the underlying dynamics.
    
        Args:
            *grids: Variable number of grid arguments representing velocity fields, passed to `get_velocity_common`.
            **kwargs: Keyword arguments passed to `get_velocity_common` to configure the computation of `alpha` and `beta`.
    
        Returns:
            float: The product of `alpha` and `beta`, representing a combined measure derived from the velocity fields.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha * beta

def vhef_grad_7(*grids, **kwargs):
    """
    Calculates a component of the equation discovery process by combining grid data with velocity components.
    
        This method computes a weighted value based on the input grids and velocity components,
        contributing to the overall equation discovery process. It uses `get_velocity_common`
        to obtain intermediate velocity components (alpha and beta) and then combines these
        with the input grids to produce a component that reflects the relationships
        between spatial variations and underlying dynamics.
    
        Args:
            *grids: Variable number of grid-like arrays representing spatial data.
            **kwargs: Keyword arguments passed to `get_velocity_common` to configure
                velocity component calculation.
    
        Returns:
            The computed gradient-like value, representing a component of the discovered equation,
            calculated as the product of the squares of the first two grids and the alpha
            component returned by `get_velocity_common`. This value reflects the influence
            of spatial variations on the identified equation.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1]**2 * alpha

def vhef_grad_8(*grids, **kwargs):
    """
    Calculates a component used in the equation discovery process.
    
        This method computes a value derived from the input grids and common velocity components (alpha, beta).
        It multiplies the first grid by the square of the second grid and the alpha component
        obtained from the velocity calculation. This result contributes to the overall equation construction
        by representing a potential term within the differential equation.
    
        Args:
          *grids: Variable number of grid-like arrays representing different terms or variables in the equation.
          **kwargs: Keyword arguments passed to `get_velocity_common` to influence velocity calculation.
    
        Returns:
          The calculated gradient-like quantity, representing a component of the discovered equation.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * grids[1]**2 * alpha

def vhef_grad_9(*grids, **kwargs):
    """
    Computes a term that contributes to the overall equation discovery process by evaluating the importance of the gradient.
    
        This method calculates a value that represents a component of the gradient within a potential differential equation. It uses provided grids to estimate gradient-related quantities (alpha and beta) and then combines these with the square of one of the grids. This term helps the evolutionary algorithm to prioritize equation structures that incorporate gradient information effectively.
    
        Args:
            *grids: Variable number of grid arguments representing different data dimensions or variables. These grids are used to calculate gradient-related quantities.
            **kwargs: Keyword arguments passed to the `get_velocity_common` function, allowing for customization of the gradient estimation process.
    
        Returns:
            float: A value representing the contribution of the gradient (specifically, the square of the second grid multiplied by alpha) to the overall equation. This value is used to guide the search for the best-fitting differential equation.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1]**2 * alpha

def vhef_grad_10(*grids, **kwargs):
    """
    Computes a gradient-like quantity based on input grids.
    
        This method calculates a value that approximates the gradient,
        derived from the input grids and velocity parameters.
        It squares the first grid, multiplies it by the second grid,
        and scales the result by a velocity parameter (alpha).
        This computation helps in identifying potential relationships
        between spatial variations and underlying dynamics.
    
        Args:
            *grids: Variable number of grid arrays representing spatial data.
            **kwargs: Keyword arguments passed to `get_velocity_common` to determine velocity parameters.
    
        Returns:
            The computed gradient-like value, scaled by the velocity parameter.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1] * alpha

def vhef_grad_11(*grids, **kwargs):
    """
    Calculates a gradient based on input grids and common velocity components.
    
        This method computes a gradient by multiplying the first two input grids
        with a combined velocity component derived from all input grids. This
        gradient is a component of the discovered differential equation.
    
        Args:
            *grids: Variable number of grid arrays representing different terms in the equation.
            **kwargs: Keyword arguments passed to `get_velocity_common` to influence velocity calculation.
    
        Returns:
            np.ndarray: The calculated gradient, representing a component of the discovered equation,
                which is the product of the first two grids and the alpha component of the common velocity.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * grids[1] * alpha

def vhef_grad_12(*grids, **kwargs):
    """
    Computes a gradient-like quantity using a combination of input grids and shared velocity components.
    
        This function estimates a gradient-related value by multiplying the second input grid with a shared velocity component (alpha) derived from all input grids. This approach helps to approximate spatial derivatives, which are essential for constructing differential equation models.
    
        Args:
            *grids: Variable number of grid-like arguments, representing spatial data. These grids are used to estimate the shared velocity components.
            **kwargs: Arbitrary keyword arguments. These are passed to the `get_velocity_common` function, allowing for customization of the velocity component calculation.
    
        Returns:
            The product of the second grid and the alpha velocity component, representing an approximation of a spatial derivative. This value contributes to the overall equation discovery process by providing gradient-related information.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1] * alpha

def vhef_grad_13(*grids, **kwargs):
    """
    Calculates a scaled representation of a velocity component based on input grids.
    
        This method computes a value related to a velocity component, derived from the input grids using `get_velocity_common`.
        The result is then scaled by the square of the first grid. This scaling emphasizes regions where the first grid has a larger magnitude,
        effectively highlighting areas of interest within the data.
    
        Args:
            *grids: Variable number of grid-like arrays. These grids are used to
                calculate velocity-related quantities.
            **kwargs: Keyword arguments passed to the `get_velocity_common` function, influencing the velocity component calculation.
    
        Returns:
            float: The calculated and scaled velocity component. This value represents a weighted measure of velocity,
                   emphasizing regions where the first grid's magnitude is significant.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * alpha

def vhef_grad_14(*grids, **kwargs):
    """
    Calculates a gradient used in the equation discovery process.
    
        This method computes a gradient by multiplying the first input grid with a
        velocity component derived from all input grids. This gradient is a component
        used within the evolutionary algorithm to evaluate potential equation structures.
        It leverages shared velocity information across multiple grids to refine the search
        for the underlying differential equation.
    
        Args:
            *grids: Variable number of grid arrays representing different data fields.
            **kwargs: Keyword arguments passed to `get_velocity_common`, influencing
                the calculation of common velocity components.
    
        Returns:
            np.ndarray: The gradient, computed as the product of the first grid and
                the alpha velocity component. This gradient contributes to the fitness
                evaluation of candidate equations.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * alpha

def vhef_grad_15(*grids, **kwargs):
    """
    Estimates the gradient of a field based on provided grid data.
    
    This function calculates the gradient by analyzing velocity fields derived from the input grids.
    It uses `get_velocity_common` to extract alpha and beta velocity components, which are then used to approximate the gradient.
    The alpha component is returned as an estimate of the gradient. This is useful for understanding the spatial changes
    within the field represented by the grids, which is a key step in discovering underlying differential equations.
    
    Args:
        *grids: Variable number of grid arguments passed to `get_velocity_common`. These grids provide the spatial data for gradient estimation.
        **kwargs: Keyword arguments passed to `get_velocity_common`. These arguments configure the velocity field calculation.
    
    Returns:
        The alpha velocity component, representing the estimated gradient of the field.
    """
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha


vhef_grad = [vhef_grad_1, vhef_grad_2, vhef_grad_3,
             vhef_grad_4, vhef_grad_5, vhef_grad_6,
             vhef_grad_7, vhef_grad_8, vhef_grad_9,
             vhef_grad_10, vhef_grad_11, vhef_grad_12,
             vhef_grad_13, vhef_grad_14, vhef_grad_15]

sign_evaluator = CustomEvaluator(evaluation_functions_np=sign_eval_fun_np, 
                                evaluation_functions_torch=sign_eval_fun_torch, 
                                eval_fun_params_labels = ['power', 'dim'])

phased_sine_evaluator = CustomEvaluator(evaluation_functions_np = phased_sine_1d_np, 
                                        evaluation_functions_torch = phased_sine_1d_torch,
                                        eval_fun_params_labels = ['power', 'freq', 'phase']) # , use_factors_grids = True
trigonometric_evaluator = CustomEvaluator(evaluation_functions_np = trig_eval_fun_np,
                                          evaluation_functions_torch = trig_eval_fun_torch,
                                          eval_fun_params_labels=['freq', 'dim', 'power']) # , use_factors_grids = True
grid_evaluator = CustomEvaluator(evaluation_functions_np = grid_eval_fun_np,
                                 evaluation_functions_torch = grid_eval_fun_torch,
                                 eval_fun_params_labels=['dim', 'power']) # , use_factors_grids=True

inverse_function_evaluator = CustomEvaluator(evaluation_functions_np = inverse_eval_fun_np,
                                             evaluation_functions_torch = inverse_eval_fun_torch,
                                             eval_fun_params_labels=['dim', 'power']) # , use_factors_grids=True

const_evaluator = CustomEvaluator(evaluation_functions_np = const_eval_fun_np,
                                  evaluation_functions_torch = const_eval_fun_torch, 
                                  eval_fun_params_labels = ['power', 'value'])
const_grad_evaluator = CustomEvaluator(evaluation_functions_np = const_grad_fun_np,
                                       evaluation_functions_torch =  const_grad_fun_np,
                                       eval_fun_params_labels = ['power', 'value'])

velocity_evaluator = CustomEvaluator(velocity_heating_eval_fun, ['p' + str(idx+1) for idx in range(15)])
velocity_grad_evaluators = [CustomEvaluator(component, ['p' + str(idx+1) for idx in range(15)])
                            for component in vhef_grad]