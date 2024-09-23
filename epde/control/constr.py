from typing import List, Union, Tuple, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch

from epde.supplementary import BasicDeriv

class ConstrLocation():
    def __init__(self, domain_shape: Tuple[int], axis: int = None, loc: int = None, 
                 indices: List[np.ndarray] = None, device: str = 'cpu'):
        '''
        Objects to contain the indices of the control training contraint location.

        Args:
            domain_shape (`Tuple[int]`): shape of the domain, for which the control problem is solved.

            axis (`int`): axis, along that the boundary conditions are selected. Shall be introduced 
                only for constraints on the boundary. 
                Optional, the default value (`None`) matches the entire domain.
            
            
            loc (`int`): position along axis, where "bounindices = self.get_boundary_indices(self.domain_indixes, axis, loc)
                self.flat_idxdary" is located. Shall be introduced only for constraints on the boundary.
                Optional, the default value (`None`) matches the entire domain. For example, -1 will correspond 
                to the end of the domain along axis. 

            device (`str`): string, matching the device, used for computation. Uses default torch designations.
                Optional, defaults to `cpu` for CPU computations.
        
        '''
        self._device = device
        self._initial_shape = domain_shape
        
        self.domain_indixes = np.indices(domain_shape)
        if indices is not None:
            self.loc_indices = indices
        elif axis is not None and loc is not None:
            self.loc_indices = self.get_boundary_indices(self.domain_indixes, axis, loc)
        else:
            self.loc_indices = self.domain_indixes
        self.flat_idxs = torch.from_numpy(np.ravel_multi_index(self.loc_indices,
                                                               dims = self._initial_shape)).long().to(self._device)


    @staticmethod
    def get_boundary_indices(domain_indices: np.ndarray, axis: int, 
                             loc: Union[int, Tuple[int]]) -> np.array:
        '''
        Method of obtaining domain indices for specified position, i.e. all 0-th elements along an axis, or the last
        elements along a specific axis.
        
        Args:
            domain_indices (`np.ndarray`): an array representing the indices of a grid. The subarrays contain index 
                values 0, 1, … varying only along the corresponding axis. For furher details inspect `np.indices(...)`
                function.
            
            axis (`int`): index of the axis, along which the elements are taken,

            loc (`int` or `tuple` of `int`): positions along the specified axis, which are taken. Can be tuple to 
            accomodate for multiple elements along axis.

        Returns:
            `np.ndarray` of indicies, where the conditions are estimated.
        '''
        return np.stack([np.take(domain_indices[idx], indices = loc, axis = axis).reshape(-1)
                         for idx in np.arange(domain_indices.shape[0])])
    
    def apply(self, tensor: torch.Tensor, flattened: bool = True, along_axis: int = None):
        '''
        Get `tensor` values at the locations, specified by the object indexing. The resulting tensor will be flattened.

        Args:
            tensor (`torch.Tensor`): the filtered tensor.

            flattened (`bool`): marker, of will the tensor be flattened. Optional, default `True`, and
            `False` is not yet implemented.

            along_axis (`int`): axis, for which the filtering is taken.
        '''
        if flattened:
            shape = [1,] * tensor.ndim
            shape[along_axis] = -1
            return torch.take_along_dim(input = tensor, indices = self.flat_idxs.view(*shape), dim = along_axis)
        else:
            raise NotImplementedError('Currently, apply can be applied only to flattened tensors.')
            idxs = self.loc_indices # loop will be held over the first dimension
            return tensor.take()


class ControlConstraint(ABC):
    '''
    Abstract class for constraints declaration in the control optimization problems.
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, indices: ConstrLocation,
                 device: str = 'cpu', deriv_axes: List = [None,], nn_output: int = 0, **kwargs):
        self._val = val
        self._indices = indices
        self._axes = deriv_axes
        self._nn_output = nn_output
        self._deriv_method = deriv_method
        self._device = device

    @abstractmethod
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor], 
                 arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq(ControlConstraint):
    '''
    Class for equality constrints of type $c(u^(n)) = f(u) - val = 0$ .
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, device: str = 'cpu', deriv_axes: List = [None,], 
                 nn_output: int = 0, tolerance: float = 1e-7, estim_func: Callable = None):
        super().__init__(val, deriv_method, indices, device, deriv_axes, nn_output) # grid, 
        self._eps = tolerance
        self._estim_func = estim_func
 
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor], 
                 arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        '''
        Calculate  
        '''
        to_compare = self._deriv_method.take_derivative(u = fun_nn, 
                                                        args = self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes = self._axes)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val).to(self._device)
        else:
            if to_compare.shape != self._val.shape:
                try:
                    to_compare = to_compare.view(self._val.size())
                except:
                    raise TypeError(f'Incorrect shapes of constraint value tensor: expected {self._val.shape}, got {to_compare.shape}.')
            val_transformed = self._val
        if self._estim_func is not None:
            constr_enf = self._estim_func(to_compare, val_transformed)
        else:
            constr_enf = val_transformed - to_compare
            
        return (torch.isclose(constr_enf, torch.zeros_like(constr_enf).to(self._device), rtol = self._eps), 
                constr_enf) # val_transformed - to_compare
        
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Return value of the loss function term, created by the condition. 

        Args:
            fun_nn (`torch.nn.Sequential`): artificial neural network, approximating the function used in the condition.

            arg_tensor (`torch.Tensor`): tensor, used as the argument of the network, passed as `fun_nn`. 

        Returns:
            `torch.Tensor` with norm of the contraint discrepancy to be used in the combined loss.
        '''
        _, discrepancy = self(fun_nn, arg_tensor)
        return torch.norm(discrepancy)

class ControlConstrNEq(ControlConstraint):
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val `self._sign` 0$
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, device: str = 'cpu', sign: str = '>', deriv_axes: List = [None,], 
                 nn_output: int = 0, tolerance: float = 1e-7, estim_func: Callable = None):
        super().__init__(val, deriv_method, indices, device, deriv_axes, nn_output) # grid, 
        self._sign = sign
        self._estim_func = estim_func

    def __call__(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, args=self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes=self._axes, component = self._nn_output)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            if not to_compare.shape == self._val.shape:
                to_compare = torch.reshape(to_compare, shape=self._val.shape)
            val_transformed = self._val
        
        if self._estim_func is not None:
            constr_enf = self._estim_func(val_transformed, to_compare)
        else:
            constr_enf = val_transformed - to_compare

        if self._sign == '>':
            return torch.greater(constr_enf, torch.zeros_like(constr_enf).to(self._device)), torch.nn.functional.relu(constr_enf)
        elif self._sign == '<':
            return (torch.less(constr_enf, torch.zeros_like(constr_enf).to(self._device)), 
                    torch.nn.functional.relu(constr_enf))
        #torch.less(val_transformed, to_compare), torch.nn.functional.relu(to_compare - val_transformed)            

        
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Return value of the loss function term, created by the condition. 

        Args:
            fun_nn (`torch.nn.Sequential`): artificial neural network, approximating the function used in the condition.

            arg_tensor (`torch.Tensor`): tensor, used as the argument of the network, passed as `fun_nn`. 

        Returns:
            `torch.Tensor` with norm of the contraint discrepancy to be used in the combined loss.
        '''        
        _, discrepancy = self(fun_nn, arg_tensor)
        return torch.norm(discrepancy)

class ConditionalLoss():
    '''
    Class for the loss, used in the control function opimizaton procedure. Conrains terms of the loss
    function in `self._cond` attribute. 
    '''
    def __init__(self, conditions: List[Tuple[Union[float, ControlConstraint, int]]]):
        '''
        Initialize the conditional loss with the terms, partaking in evaluating inflicted control and 
        quality of the equation solution with the current control. 

        Args:
            conditions (`list` of triplet `tuple` as (`float`, `ControlConstraint`, `int`))
        '''
        self._cond = conditions

    def __call__(self, models: List[torch.nn.Sequential], args: list): # Introduce prepare control input: get torch tensors from solver & autodiff them
        '''
        Return the summed values of the loss function component 
        '''
        temp = []
        for cond in self._cond:
            temp.append(cond[0] * cond[1].loss(models[cond[2]], args[cond[2]]))
        # print('temp loss', temp)
        return torch.stack(temp, dim=0).sum(dim=0).sum(dim=0)