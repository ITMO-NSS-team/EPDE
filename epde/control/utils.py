from typing import List, Tuple, Union
from collections import OrderedDict

import numpy as np
import torch

from epde.supplementary import BasicDeriv, AutogradDeriv

def prepare_control_inputs(model: Union[torch.nn.Sequential, List[np.ndarray]], grid: torch.Tensor, 
                           args: List[Tuple[Union[int, List]]], diff_method: BasicDeriv = None) -> torch.Tensor:
    """
    Recompute the arguments for the control ANN by evaluating derivatives of the solution with respect to the input grid.
    
        This function calculates the necessary derivatives of the solution `model` on the given `grid`
        according to the specifications in `args`. These derivatives then form the input to the control ANN.
        This process allows the control ANN to learn the relationship between the solution and its derivatives,
        enabling the discovery of the underlying differential equation.
    
        Args:
            model (`torch.nn.Sequential` or `List[np.ndarray]`): The solution of the controlled equation.
                Can be a neural network or a list of numpy arrays.
    
            grid (`torch.Tensor`): A tensor representing the grid points where the solution is evaluated.
                Shape: (m, n), where m is the number of points and n is the number of input dimensions.
    
            args (`List[Tuple[Union[int, List]]]`): A list of tuples specifying the derivative operators.
                Each tuple contains the component index (int) and the axes (list of ints) with respect to which
                the derivative is taken.
    
            diff_method (`BasicDeriv`, optional): The method used for calculating derivatives.
                Defaults to `AutogradDeriv`.
    
        Returns:
            `torch.Tensor`: A tensor containing the arguments for the control ANN.
                Each column represents a different derivative, and each row corresponds to a grid point.
                Shape: (m, k), where m is the number of grid points and k is the number of derivatives.
    """
    if diff_method is None:
        diff_method = AutogradDeriv

    differentiator = diff_method()
    ctrl_inputs = [differentiator.take_derivative(u = model, args = grid,
                                                  axes = arg[1], component = arg[0]).reshape(-1, 1) for arg in args]
    if not isinstance(model, torch.nn.Sequential):
        ctrl_inputs = [torch.from_numpy(inp).reshape((-1, 1)) for inp in ctrl_inputs]
    ctrl_inputs = torch.cat(ctrl_inputs, dim = 1).float()
    # print(f'ctrl_inputs shape is {ctrl_inputs.shape}')
    return ctrl_inputs

    # if isinstance(model, torch.nn.Sequential):
    #     differentiator = AutogradDeriv()
    #     ctrl_inputs = torch.cat([differentiator.take_derivative(u = model, args = grid, axes = arg[1],
    #                                                             component = arg[0]).reshape(-1, 1) for arg in args], dim = 1).float()
    # else:
    #     assert isinstance(grid, torch.Tensor) and grid.ndim == 2 and grid.shape[-1] == 1
    #     grid = grid.detach().cpu().numpy()
    #     differentiator = FDDeriv()
    #     ctrl_inputs = torch.cat([torch.from_numpy(differentiator.take_derivative(model, grid, axes = arg[1],
    #                                                                              component = arg[0])).reshape(-1, 1) 
    #                              for arg in args], dim = 1).float()
    return ctrl_inputs

@torch.no_grad()
def eps_increment_diff(input_params: OrderedDict, loc: List[Union[str, Tuple[int]]], 
                       forward: bool = True, eps = 1e-4): # input_keys: list,  prev_loc: List = None, 
    if forward:
    """
    Incrementally modifies a parameter in the input dictionary by a small value (epsilon).
    
        This method perturbs a specific parameter within a nested dictionary structure by either adding or subtracting a small value (epsilon). The modification is done in-place. This perturbation is crucial for exploring the parameter space and evaluating the sensitivity of the discovered equations to slight variations in parameter values. By making these small adjustments, the algorithm can refine its search for the optimal equation structure and parameterization that best fits the observed data.
    
        Args:
            input_params: The input dictionary containing the parameters to be modified.
            loc: A list specifying the location of the parameter to be modified within the input dictionary. The first element is the key for the outer dictionary, and the remaining elements form a tuple used as a key for the inner dictionary.
            forward: A boolean indicating whether to increment (True) or decrement (False) the parameter. Defaults to True.
            eps: The small value (epsilon) to add or subtract. Defaults to 1e-4.
    
        Returns:
            OrderedDict: The modified input dictionary with the parameter at the specified location incremented or decremented.
    """
        input_params[loc[0]][tuple(loc[1:])] += eps  
    else:     
        input_params[loc[0]][tuple(loc[1:])] -= 2*eps
    return input_params
