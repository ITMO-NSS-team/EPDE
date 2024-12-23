from typing import List, Tuple, Union
from collections import OrderedDict

import numpy as np
import torch

from epde.supplementary import BasicDeriv, AutogradDeriv

def prepare_control_inputs(model: Union[torch.nn.Sequential, List[np.ndarray]], grid: torch.Tensor, 
                           args: List[Tuple[Union[int, List]]], diff_method: BasicDeriv = None) -> torch.Tensor:
    '''
    Recompute the control ANN arguments tensor from the solutions of 
    controlled equations $L \mathbf{u}(t, \mathbf{x}, \mathbf{c}) = 0$, 
    calculating necessary derivatives, as `args` 

    Args:
        model (`torch.nn.Sequential`): solution of the controlled equation $\mathbf{u}(\mathbf{u})$.
        
        grid (`torch.Tensor`): tensor of the grids m x n, where m - number of points in the domain, n - number of NN inputs.

        args (`List[Tuple[Union[int, List]]]`) - list of arguments of derivative operators.

    Returns:
        `torch.Tensor`: tensor of arguments for the control ANN.
    '''
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
        input_params[loc[0]][tuple(loc[1:])] += eps  
    else:     
        input_params[loc[0]][tuple(loc[1:])] -= 2*eps
    return input_params
