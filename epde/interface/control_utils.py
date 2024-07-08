import numpy as np
import torch

from functools import partial
from typing import Tuple, List, Dict, Union
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict

import epde.globals as global_var
from epde.interface.interface import EpdeMultisample, EpdeSearch, ExperimentCombiner
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.interface.solver_integration import SolverAdapter, BOPElement 
from epde.supplementary import BasicDeriv, AutogradDeriv

from epde.solver.data import Conditions
from epde.solver.optimizers.closure import Closure


# TODO: Later to be refactored into multiple files

# class ControlOptClosure():
#     '''
#     Inspired by https://github.com/ITMO-NSS-team/torch_DE_solver/blob/main/tedeous/optimizers/closure.py
#     '''
#     def __init__(self, model):
#         self.set_model()
#         self.parameters = model.parameters
#         self._optimizer = model.optimizer

#     def get_closure(self):
#         self._optimizer.zero_grad()

class ControlConstraint(ABC):
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 deriv_axes: List = [None,], nn_output: int = 0, **kwargs):
        self._val = val
        # self._grid = grid
        self._axes = deriv_axes
        self._nn_output = nn_output
        self._deriv_method = deriv_method

    @abstractmethod
    def __call__(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq(ControlConstraint):
    '''
    Class for constrints of type $c(u^(n)) = f(u) - val = 0$
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 deriv_axes: List = [None,], nn_output: int = 0, tolerance: float = 1e-7, ):
        super().__init__(val, deriv_method, deriv_axes, nn_output) # grid, 
        self._eps = tolerance
 
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor]) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, grid=self._grid, axes=self._axes)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            assert to_compare.shape == self._val.shape, 'Incorrect shapes of constraint value tensor'
            val_transformed = self._val
        return torch.isclose(val_transformed, to_compare, rtol = self._eps), val_transformed - to_compare
        
    def loss(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor]) -> torch.Tensor:
        print(f'fun_nn is {fun_nn}')
        _, discrepancy = self(fun_nn)
        return torch.norm(discrepancy)

class ControlConstrNEq(ControlConstraint):
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val < 0$
    '''
    def __call__(self, fun_nn: torch.nn.Sequential) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, grid=self._grid,
                                                        axes=self._axes, component = self._nn_output)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            assert to_compare.shape == self._val.shape, 'Incorrect shapes of constraint value tensor'
            val_transformed = self._val
        return torch.greater(val_transformed, to_compare), torch.nn.functional.relu(val_transformed - to_compare)
        
    def loss(self, fun_nn: torch.nn.Sequential) -> torch.Tensor:
        _, discrepancy = self(fun_nn)
        return torch.norm(discrepancy)

class ConditionalLoss():
    def __init__(self, conditions: List[Tuple[Union[float, ControlConstraint, int]]]):
        self._cond = conditions

    def __call__(self, models: List[torch.nn.Sequential], ):
        temp = []
        for cond in self._cond:
            print(cond[1]._val, cond[1]._grid, cond[1]._axes)
            temp.append(cond[0] * cond[1].loss(models[cond[2]]))
        print('temp loss', temp)
        return torch.stack(temp, dim=0).sum(dim=0).sum(dim=0)

class ControlExp():
    def __init__(self, loss : ConditionalLoss):
        self._state_net = None
        self._best_control_net = None
        self.loss = loss

    def create_best_equations(self, optimal_equations: Union[list, ParetoLevels]):
        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)

    @staticmethod
    def create_ode_bop(key, var, term, grid_loc, value):
        bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        return bop

    def set_control_optim_params(self, lr: float = 1e-2, max_iter: int = 10, max_eval = None, tolerance_grad: float = 1e-7,
                                 tolerance_change: float = 1e-9, history_size: int = 100, line_search_fn = 'strong_wolfe'):
        self._control_opt_params = {'lr': lr, 'max_iter': max_iter, 'max_eval': max_eval, 'tolerance_grad': tolerance_grad,
                                    'tolerance_change': tolerance_change, 'history_size': history_size, 
                                    'line_search_fn': line_search_fn}

    def set_solver_params(self, mode: str = 'autograd', compiling_params: dict = {}, optimizer_params: dict = {},
                          cache_params: dict = {}, early_stopping_params: dict = {}, plotting_params: dict = {}, 
                          training_params: dict = {'epochs':1e2,}, use_cache: bool = False, use_fourier: bool = False, 
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False, device = torch.device('cpu')):
        self._solver_params = {'mode': mode, 
                               'compiling_params': compiling_params, 
                               'optimizer_params': optimizer_params,
                               'cache_params': cache_params,
                               'early_stopping_params': early_stopping_params,
                               'plotting_params': plotting_params,
                               'training_params': training_params,
                               'use_cache': use_cache,
                               'use_fourier': use_fourier,
                               'fourier_params': fourier_params,
                               'use_adaptive_lambdas': use_adaptive_lambdas,
                               'device': device
                               }

    @staticmethod
    def finite_diff_calculation(system, adapter, loc, control_loss, state_net: torch.nn.Sequential, # prev_loc,  
                                bc_operators, grids: list, solver_params: dict, eps: float):
        # Calculating loss in p[i]+eps:        
        global_var.control_nn.load_state_dict(eps_increment_diff(input_params=global_var.control_nn.state_dict(),
                                                                 loc = loc, forward=True, eps=eps))
        adapter.set_net(state_net)
        _, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                             boundary_conditions = bc_operators,
                                             mode = solver_params['mode'],
                                             use_cache = solver_params['use_cache'],
                                             use_fourier = solver_params['use_fourier'],
                                             fourier_params = solver_params['fourier_params'],
                                             use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        
        loss_forward = control_loss([model, global_var.control_nn])
        # Calculating loss in p[i]-eps:

        global_var.control_nn.load_state_dict(eps_increment_diff(input_params=global_var.control_nn.state_dict(),
                                                                 loc = loc, forward=False, eps=eps))
        adapter.set_net(state_net)
        _, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                             boundary_conditions = bc_operators,
                                             mode = solver_params['mode'],
                                             use_cache = solver_params['use_cache'],
                                             use_fourier = solver_params['use_fourier'],
                                             fourier_params = solver_params['fourier_params'],
                                             use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        
        loss_back = control_loss([model, global_var.control_nn])
        
        # Restore values of the control NN parameters
        global_var.control_nn.load_state_dict(eps_increment_diff(input_params=global_var.control_nn.state_dict(),
                                                                 loc = loc, forward=True, eps=eps))

        return (loss_forward - loss_back)/(2*eps)

    def train_pinn(self, bc_operators: List[BOPElement], grids: List[Union[np.ndarray, torch.Tensor]], 
                   control_args = [], n_control: int = 1, epochs: int = 1e4, 
                   state_net: torch.nn.Sequential = None, control_net: torch.nn.Sequential = None):
        # Properly formulate training approach
        t = 0
        min_loss = np.inf
        stop_training = False

        if isinstance(state_net, torch.nn.Sequential): self._state_net = state_net
        global_var.reset_control_nn(n_var = len(control_args), n_control = n_control, ann = control_net)
        
        # TODO: To optimize the net in gloabl variables is a terrific approach, rethink it

        if isinstance(grids[0], np.ndarray):
            grids_merged = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T
        elif isinstance(grids[0], torch.Tensor):
            grids_merged = torch.cat([subgrid.reshape(-1, 1) for subgrid in grids], dim = 1).float()
        grids_merged.to(device=self._solver_params['device'])

        # Implement closure for loss function?
        # Parameters update - by epsilon in loop one-by-one, without deepcopy.
        grad_tensors = deepcopy(global_var.control_nn.state_dict())
        param_keys = list(global_var.control_nn.state_dict().keys())

        optimizer = AdamOptimizer(optimized = global_var.control_nn.state_dict())
        adapter = self.get_solver_adapter(None)
        while t < epochs and not stop_training:
            state_net = deepcopy(self._state_net)
            eps = 1e-4

            for param_key, param_tensor in grad_tensors.items():
                if len(param_tensor.size()) == 1:
                    #loss in the forward parameter point calculation
                    for param_idx, _ in enumerate(param_tensor):
                        loc = (param_key, param_idx)
                        grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system, 
                                                                                     adapter = adapter,
                                                                                     loc = loc, control_loss = self.loss, 
                                                                                     state_net = self._state_net,
                                                                                     bc_operators = bc_operators,
                                                                                     grids = [grids, grids_merged], 
                                                                                     solver_params = self._solver_params,
                                                                                     eps = eps) # param_keys = param_keys
                elif len(param_tensor.size()) == 2:
                    for param_outer_idx, _ in enumerate(param_tensor):
                        for param_inner_idx, _ in enumerate(param_tensor[0]):
                            loc = (param_key, param_outer_idx, param_inner_idx)
                            grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system, 
                                                                                         adapter = adapter,
                                                                                         loc = loc, control_loss = self.loss, 
                                                                                         state_net = self._state_net,
                                                                                         bc_operators = bc_operators,
                                                                                         grids = [grids, grids_merged], 
                                                                                         solver_params = self._solver_params,
                                                                                         eps = eps) # param_keys = param_keys                       
                else:
                    raise Exception(f'Incorrect shape of weights/bias. Got {param_tensor.size()} tensor.')
            global_var.control_nn.load_state_dict(optimizer.step(gradient = grad_tensors, 
                                                                 optimized = global_var.control_nn.state_dict()))
            adapter.set_net(self._state_net)
            _, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                 boundary_conditions = bc_operators,
                                                 mode = self._solver_params['mode'],
                                                 use_cache = self._solver_params['use_cache'],
                                                 use_fourier = self._solver_params['use_fourier'],
                                                 fourier_params = self._solver_params['fourier_params'],
                                                 use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

            var_prediction = model(grids_merged)
            self._state_net = model

            loss = self.loss([self._state_net, global_var.control_nn])
            print(loss)
            if loss < min_loss:
                min_loss = loss
                self._best_control_params = global_var.control_nn.state_dict()
        ctrl_pred = global_var.control_nn(var_prediction)

        return self._state_net, global_var.control_nn, ctrl_pred

    def get_solver_adapter(self, net: torch.nn.Sequential):
        adapter = SolverAdapter(net = net, use_cache = False)
        # Edit solver forms of functions of dependent variable to Callable objects.
        # Setting various adapater parameters
        adapter.set_compiling_params(**self._solver_params['compiling_params'])
        adapter.set_optimizer_params(**self._solver_params['optimizer_params'])
        adapter.set_cache_params(**self._solver_params['cache_params'])
        adapter.set_early_stopping_params(**self._solver_params['early_stopping_params'])
        adapter.set_plotting_params(**self._solver_params['plotting_params'])
        adapter.set_training_params(**self._solver_params['training_params'])
        adapter.change_parameter('mode', self._solver_params['mode'], param_dict_key = 'compiling_params')
        
        return adapter

@torch.no_grad()
def eps_increment_diff(input_params: OrderedDict, loc: List[Union[str, Tuple[int]]], 
                       forward: bool = True, eps = 1e-4): # input_keys: list,  prev_loc: List = None, 
    if forward:
        input_params[loc[0]][loc[1:]] += eps
    else:
        input_params[loc[0]][loc[1:]] -= 2*eps
    return input_params

class FirstOrderOptimizerNp(ABC):
    def __init__(self, parameters: np.ndarray, optimized: np.ndarray):
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def step(self, gradient: np.ndarray):
        raise NotImplementedError('Calling step of an abstract optimizer')

class AdamOptimizerNp(FirstOrderOptimizerNp):
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
    def __init__(self, optimized: List[torch.Tensor], parameters: list):
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def step(self, gradient: List[torch.Tensor], optimized: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError('Calling step of an abstract optimizer')
    
class AdamOptimizer(FirstOrderOptimizer):
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

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.time += 1
        self._moment = [self.parameters[1] * self._moment[tensor_idx] + (1-self.parameters[1]) * grad_subtensor
                        for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        self._second_moment = [self.parameters[2]*self._second_moment[tensor_idx] + (1-self.parameters[2])*torch.power(grad_subtensor, 2)
                               for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        moment_cor = self._moment/(1 - self.parameters[1] ** self.time) #np.power(self.parameters[1], self.time))
        second_moment_cor = self._second_moment/(1 - self.parameters[2] ** self.time) # np.power(self.parameters[2], self.time)
        return [optimized[subtensor_key] - self.parameters[0]*moment_cor[tensor_idx]/(torch.sqrt(second_moment_cor[tensor_idx]) +\
                                                                           self.parameters[3])
                for tensor_idx, subtensor_key in enumerate(optimized.keys())]