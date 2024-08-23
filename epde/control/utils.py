from typing import List, Tuple, Union
from collections import OrderedDict

import torch

from epde.supplementary import AutogradDeriv

def prepare_control_inputs(model: torch.nn.Sequential, grid: torch.Tensor, 
                           args: List[Tuple[Union[int, List]]]) -> torch.Tensor:
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
    differntiatior = AutogradDeriv()
    ctrl_inputs = torch.cat([differntiatior.take_derivative(u = model, args = grid, axes = arg[1],
                                                            component = arg[0]).reshape(-1, 1) for arg in args], dim = 1)
    return ctrl_inputs

@torch.no_grad()
def eps_increment_diff(input_params: OrderedDict, loc: List[Union[str, Tuple[int]]], 
                       forward: bool = True, eps = 1e-4): # input_keys: list,  prev_loc: List = None, 
    if forward:
        input_params[loc[0]][tuple(loc[1:])] += eps  
    else:     
        input_params[loc[0]][tuple(loc[1:])] -= 2*eps
    return input_params