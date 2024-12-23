from abc import ABC
from collections import OrderedDict
from typing import List, Dict

from warnings import warn

import numpy as np
import torch

class FirstOrderOptimizerNp(ABC):
    behavior = 'None'      
    def __init__(self, parameters: np.ndarray, optimized: np.ndarray):
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def step(self, gradient: np.ndarray):
        raise NotImplementedError('Calling step of an abstract optimizer')

class AdamOptimizerNp(FirstOrderOptimizerNp):
    behavior = 'Gradient'      
    def __init__(self, optimized: np.ndarray, parameters: np.ndarray = np.array([0.001, 0.9, 0.999, 1e-8])):
        '''
        parameters[0] - alpha, parameters[1] - beta_1, parameters[2] - beta_2
        parameters[3] - eps  
        '''
        self.reset(optimized, parameters)

    def reset(self, optimized: np.ndarray, parameters: np.ndarray):
        self._moment = np.zeros_like(optimized)
        self._second_moment = np.zeros_like(optimized)
        self._second_moment_max = np.zeros_like(optimized)
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: np.ndarray, optimized: np.ndarray):
        self.time += 1
        self._moment = self.parameters[1] * self._moment + (1-self.parameters[1]) * gradient
        self._second_moment = self.parameters[2] * self._second_moment +\
                              (1-self.parameters[2]) * np.power(gradient)
        moment_cor = self._moment/(1 - np.power(self.parameters[1], self.time))
        second_moment_cor = self._second_moment/(1 - np.power(self.parameters[2], self.time))
        return optimized - self.parameters[0]*moment_cor/(np.sqrt(second_moment_cor)+self.parameters[3])
    
class FirstOrderOptimizer(ABC):
    behavior = 'Gradient'    
    def __init__(self, optimized: List[torch.Tensor], parameters: list):
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        raise NotImplementedError('Calling reset method of an abstract optimizer')

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor],
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError('Calling step of an abstract optimizer')
    
class AdamOptimizer(FirstOrderOptimizer):
    behavior = 'Gradient'
    def __init__(self, optimized: List[torch.Tensor], parameters: list = [0.001, 0.9, 0.999, 1e-8]):
        '''
        parameters[0] - alpha, parameters[1] - beta_1, parameters[2] - beta_2
        parameters[3] - eps
        '''
        self.reset(optimized, parameters)

    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        self._moment = [torch.zeros_like(param_subtensor) for param_subtensor in optimized.values()] 
        self._second_moment = [torch.zeros_like(param_subtensor) for param_subtensor in optimized.values()]
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor],
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        self.time += 1

        self._moment = [self.parameters[1] * self._moment[tensor_idx] + (1-self.parameters[1]) * grad_subtensor
                        for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        
        self._second_moment = [self.parameters[2]*self._second_moment[tensor_idx] + 
                               (1-self.parameters[2])*torch.pow(grad_subtensor, 2)
                               for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        
        moment_cor = [moment_tensor/(1 - self.parameters[1] ** self.time) for moment_tensor in self._moment] 
        second_moment_cor = [sm_tensor/(1 - self.parameters[2] ** self.time) for sm_tensor in self._second_moment] 
        return OrderedDict([(subtensor_key, optimized[subtensor_key] - self.parameters[0] * moment_cor[tensor_idx]/\
                             (torch.sqrt(second_moment_cor[tensor_idx]) + self.parameters[3]))
                            for tensor_idx, subtensor_key in enumerate(optimized.keys())])
    
class CoordDescentOptimizer(FirstOrderOptimizer):
    behavior = 'Coordinate'    
    def __init__(self, optimized: List[torch.Tensor], parameters: list = [0.001,]):
        '''
        parameters[0] - alpha, parameters[1] - beta_1, parameters[2] - beta_2
        parameters[3] - eps  
        '''
        self.reset(optimized, parameters)

    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor], 
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        self.time += 1
        assert 'loc' in kwargs.keys(), 'Missing location of parameter value shift in coordinate descent.'
        loc = kwargs['loc']
        if torch.isclose(gradient[loc[0]][tuple(loc[1:])], 
                         torch.tensor((0,)).to(device=gradient[loc[0]][tuple(loc[1:])].device).float()):
            warn(f'Gradient at {loc} is close to zero: {gradient[loc[0]][tuple(loc[1:])]}.')
        optimized[loc[0]][tuple(loc[1:])] = optimized[loc[0]][tuple(loc[1:])] -\
                                            self.parameters[0]*gradient[loc[0]][tuple(loc[1:])]
        return optimized
    
# TODO: implement coordinate descent
    #np.power(self.parameters[1], self.time)) # np.power(self.parameters[2], self.time)

# class LBFGS(FirstOrderOptimizer):
#     def __init__(self, optimized: List[torch.Tensor], parameters: list = []):
#         pass

#     def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         pass

#     def update_hessian(self, gradient: Dict[str, torch.Tensor], x_vals: Dict[str, torch.Tensor]):
#         # Use self._prev_grad
#         for i in range(self._mem_size - 1):
#             alpha = 

#     def get_alpha(self):
#         return alpha