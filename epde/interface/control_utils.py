import numpy as np
import torch

import matplotlib.pyplot as plt
import datetime
import os 

import gc
from functools import partial
from typing import Tuple, List, Dict, Union, Callable
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

def prepare_control_inputs(model: torch.nn.Sequential, grid: torch.Tensor, args: List[Tuple[Union[int, List]]]):
    differntiatior = AutogradDeriv()
    res = torch.cat([differntiatior.take_derivative(u = model, args = grid, axes = arg[1],
                                                    component = arg[0]).reshape(-1, 1) for arg in args], dim = 1)
    # print('res.shape', res.shape)
    return res

class ConstrLocation():
    def __init__(self, domain_shape: Tuple[int], axis: int = None, loc: int = None, 
                 indices: List[np.ndarray] = None, device: str = 'cpu'):
        '''
        Contruct indices of the control training contraint location.
        Args:
            domain_shape (`Tuple[int]`): shape of the domain, for which the control problem is solved.
            axis (`int`): axis, along that the boundary conditions are selected. Shall be introduced only for constraints on the boundary.
            loc (`int`): position along axis, where the "bounindices = self.get_boundary_indices(self.domain_indixes, axis, loc)
        else:
            self.loc_indices = self.domain_indixes
        self.flat_idxdary" is located. Shall be introduced only for constraints on the boundary.
        
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
    def get_boundary_indices(domain_indices: np.ndarray, axis: int, loc: Union[int, Tuple[int]]): # , shape: Tuple[int]
        return np.stack([np.take(domain_indices[idx], indices = loc, axis = axis).reshape(-1)
                         for idx in np.arange(domain_indices.shape[0])])
    
    def apply(self, tensor: torch.Tensor, flattened: bool = True, along_axis: int = None): # Union[int, Tuple[int]]
        if flattened:
            shape = [1,] * tensor.ndim
            shape[along_axis] = -1
            return torch.take_along_dim(input = tensor, indices = self.flat_idxs.view(*shape), dim = along_axis)
        else:
            raise NotImplementedError('Currently, apply can be applied only to flattened tensors.')
            idxs = self.loc_indices # loop will be held over the first dimension
            return tensor.take()
        # idxs = torch.from_numpy(idxs).long().unsqueeze(1) # am I needed?


class ControlConstraint(ABC):
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, deriv_axes: List = [None,], nn_output: int = 0, **kwargs):
        self._val = val
        self._indices = indices
        self._axes = deriv_axes
        self._nn_output = nn_output
        self._deriv_method = deriv_method

    @abstractmethod
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor], 
                 arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq(ControlConstraint):
    '''
    Class for constrints of type $c(u^(n)) = f(u) - val = 0$
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, deriv_axes: List = [None,], 
                 nn_output: int = 0, tolerance: float = 1e-7):
        super().__init__(val, deriv_method, indices, deriv_axes, nn_output) # grid, 
        self._eps = tolerance
 
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor], 
                 arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, args=self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes=self._axes)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val).to(self._device)
        else:
            if to_compare.shape != self._val.shape:
                raise TypeError(f'Incorrect shapes of constraint value tensor: expected {self._val.shape}, got {to_compare.shape}.')
            val_transformed = self._val
        return torch.isclose(val_transformed, to_compare, rtol = self._eps), val_transformed - to_compare
        
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        _, discrepancy = self(fun_nn, arg_tensor)
        return torch.norm(discrepancy)

class ControlConstrNEq(ControlConstraint):
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val `self._sign` 0$
    '''
    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, sign: str = '>', deriv_axes: List = [None,], 
                 nn_output: int = 0, tolerance: float = 1e-7):
        super().__init__(val, deriv_method, indices, deriv_axes, nn_output) # grid, 
        self._sign = sign

    def __call__(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, args=self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes=self._axes, component = self._nn_output)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            assert to_compare.shape == self._val.shape, 'Incorrect shapes of constraint value tensor'
            val_transformed = self._val
        if self._sign == '>':
            return torch.greater(val_transformed, to_compare), torch.nn.functional.relu(val_transformed - to_compare)
        elif self._sign == '<':
            return torch.less(val_transformed, to_compare), torch.nn.functional.relu(to_compare - val_transformed)            

        
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        _, discrepancy = self(fun_nn, arg_tensor)
        return torch.norm(discrepancy)

class ConditionalLoss():
    def __init__(self, conditions: List[Tuple[Union[float, ControlConstraint, int]]]):
        self._cond = conditions

    def __call__(self, models: List[torch.nn.Sequential], args: list): # Introduce prepare control input: get torch tensors from solver & autodiff them
        temp = []
        for cond in self._cond:
            temp.append(cond[0] * cond[1].loss(models[cond[2]], args[cond[2]]))
        # print('temp loss', temp)
        return torch.stack(temp, dim=0).sum(dim=0).sum(dim=0)

class ControlExp():
    def __init__(self, loss : ConditionalLoss, device: str = 'cpu'):
        self._device = device
        self._state_net = None
        self._best_control_net = None
        self.loss = loss

    def create_best_equations(self, optimal_equations: Union[list, ParetoLevels]):
        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)

    @staticmethod
    def create_ode_bop(key, var, term, grid_loc, value, device: str = 'cpu'):
        bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor).to(device))
        bop.values = torch.from_numpy(np.array([[value,]])).float().to(device)
        return bop

    def set_control_optim_params(self, lr: float = 1e-3, max_iter: int = 10, max_eval = None, tolerance_grad: float = 1e-7,
                                 tolerance_change: float = 1e-9, history_size: int = 100, line_search_fn = 'strong_wolfe'):
        self._control_opt_params = {'lr': lr, 'max_iter': max_iter, 'max_eval': max_eval, 'tolerance_grad': tolerance_grad,
                                    'tolerance_change': tolerance_change, 'history_size': history_size, 
                                    'line_search_fn': line_search_fn}

    def set_solver_params(self, mode: str = 'autograd', compiling_params: dict = {}, optimizer_params: dict = {},
                          cache_params: dict = {}, early_stopping_params: dict = {}, plotting_params: dict = {},
                          training_params: dict = {'epochs': 150,}, use_cache: bool = False, use_fourier: bool = False, #  5*1e0
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False, device: str = 'cpu'):
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
                               'device': torch.device(self._device)
                               }

    @staticmethod
    def finite_diff_calculation(system, adapter, loc, control_loss, state_net: torch.nn.Sequential, # prev_loc,
                                bc_operators, grids: list, solver_params: dict, eps: float):
        # Calculating loss in p[i]+eps: 
        state_dict_prev = global_var.control_nn.net.state_dict()
        state_dict = eps_increment_diff(input_params=state_dict_prev,
                                        loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(state_dict)
        state_dict_prev = state_dict = None

        adapter.set_net(state_net)
        _, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                             boundary_conditions = bc_operators,
                                             mode = solver_params['mode'],
                                             use_cache = solver_params['use_cache'],
                                             use_fourier = solver_params['use_fourier'],
                                             fourier_params = solver_params['fourier_params'],
                                             use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        
        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args)
        loss_forward = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])
        # Calculating loss in p[i]-eps:

        state_dict_prev = global_var.control_nn.net.state_dict()
        state_dict = eps_increment_diff(input_params=state_dict_prev,
                                        loc = loc, forward=False, eps=eps)
        global_var.control_nn.net.load_state_dict(state_dict)

        adapter.set_net(state_net)
        solver_loss, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                                       boundary_conditions = bc_operators,
                                                       mode = solver_params['mode'],
                                                       use_cache = solver_params['use_cache'],
                                                       use_fourier = solver_params['use_fourier'],
                                                       fourier_params = solver_params['fourier_params'],
                                                       use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        # print(f'solver_loss: {solver_loss}')

        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args)
        loss_back = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])
        
        # Restore values of the control NN parameters
        state_dict_prev = global_var.control_nn.net.state_dict()
        state_dict = eps_increment_diff(input_params=state_dict_prev,
                                        loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(state_dict)
        state_dict = state_dict_prev = None

        res = (loss_forward - loss_back)/(2*eps)
        return res
        

    def train_pinn(self, bc_operators: List[Union[dict, float]], grids: List[Union[np.ndarray, torch.Tensor]], 
                   n_control: int = 1, epochs: int = 1e2, state_net: torch.nn.Sequential = None, opt_params: List[float] = [0.01, 0.9, 0.999, 1e-8],
                   control_net: torch.nn.Sequential = None, fig_folder: str = None, LV_exp: bool = True):
        def modify_bc(operator: dict, scale: Union[float, torch.Tensor]) -> dict:
            noised_operator = deepcopy(operator)
            # noised_operator['bnd_val'] = torch.normal(operator['bnd_val'], scale).to(self._device)
            return noised_operator

        # Properly formulate training approach
        t = 0
        # min_loss = np.inf
        loss_hist = []
        stop_training = False

        time = datetime.datetime.now()

        if isinstance(state_net, torch.nn.Sequential): self._state_net = state_net
        global_var.reset_control_nn(n_control = n_control, ann = control_net, ctrl_args = global_var.control_nn.net_args)
        
        # TODO Refactor hook: To optimize the net in global variables is a terrific approach, rethink it

        if isinstance(grids[0], np.ndarray):
            grids_merged = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T.to(self._device)
        elif isinstance(grids[0], torch.Tensor):
            grids_merged = torch.cat([subgrid.reshape(-1, 1) for subgrid in grids], dim = 1).float()#.to(self._device)
        grids_merged.to(device=self._device)

        grad_tensors = deepcopy(global_var.control_nn.net.state_dict())

        optimizer = AdamOptimizer(optimized = global_var.control_nn.net.state_dict(), parameters = opt_params)
        adapter = self.get_solver_adapter(None)
        while t < epochs and not stop_training:
            sampled_bc = [modify_bc(operator, noise_std) for operator, noise_std in bc_operators]

            state_net  = deepcopy(self._state_net)
            eps = 1e-3
            print(f'Control function optimization epoch {t}.')
            for param_key, param_tensor in grad_tensors.items():
                print(f'Optimizing {param_key}: shape is {param_tensor.shape}')
                if len(param_tensor.size()) == 1:
                    #loss in the forward parameter point calculation
                    for param_idx, _ in enumerate(param_tensor):
                        loc = (param_key, param_idx)
                        grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()
                        grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system,
                                                                                     adapter = adapter,
                                                                                     loc = loc, control_loss = self.loss, 
                                                                                     state_net = self._state_net,
                                                                                     bc_operators = sampled_bc,
                                                                                     grids = [grids, grids_merged], 
                                                                                     solver_params = self._solver_params,
                                                                                     eps = eps)
                elif len(param_tensor.size()) == 2:
                    for param_outer_idx, _ in enumerate(param_tensor):
                        for param_inner_idx, _ in enumerate(param_tensor[0]):
                            loc = (param_key, param_outer_idx, param_inner_idx)
                            grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()
                            grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system,
                                                                                         adapter = adapter,
                                                                                         loc = loc, control_loss = self.loss, 
                                                                                         state_net = self._state_net,
                                                                                         bc_operators = sampled_bc,
                                                                                         grids = [grids, grids_merged], 
                                                                                         solver_params = self._solver_params,
                                                                                         eps = eps)                   
                else:
                    raise Exception(f'Incorrect shape of weights/bias. Got {param_tensor.size()} tensor.')
            state_dict_prev = global_var.control_nn.net.state_dict()
            state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev)
            global_var.control_nn.net.load_state_dict(state_dict)
            del state_dict, state_dict_prev

            adapter.set_net(self._state_net)
            _, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                 boundary_conditions = sampled_bc,
                                                 mode = self._solver_params['mode'],
                                                 use_cache = self._solver_params['use_cache'],
                                                 use_fourier = self._solver_params['use_fourier'],
                                                 fourier_params = self._solver_params['fourier_params'],
                                                 use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

            var_prediction = model(grids_merged)
            self._state_net = model

            control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args)        
            loss = self.loss([self._state_net, global_var.control_nn.net], [grids_merged, control_inputs])
            print('current loss is ', loss)
            # if loss < min_loss:
            # min_loss = loss
            loss_hist.append(loss)
            self._best_control_params = global_var.control_nn.net.state_dict()
            
            if fig_folder is not None and LV_exp:
                plt.figure(figsize=(11, 6))
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 0], color = 'k')
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 1], color = 'r')
                plt.plot(grids_merged.cpu().detach().numpy(), global_var.control_nn.net(control_inputs).cpu().detach().numpy(),
                         color = 'tab:orange')
                plt.grid()
                frame_name = f'Exp_{time.month}_{time.day}_at_{time.hour}_{time.minute}_{t}.png'
                plt.savefig(os.path.join(fig_folder, frame_name))
            gc.collect()
            t += 1

        ctrl_pred = global_var.control_nn.net(var_prediction)

        return self._state_net, global_var.control_nn.net, ctrl_pred, loss_hist

                            
    def get_solver_adapter(self, net: torch.nn.Sequential):
        adapter = SolverAdapter(net = net, use_cache = False, device = self._device)
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
        input_params[loc[0]][tuple(loc[1:])] += eps
    else:
        input_params[loc[0]][tuple(loc[1:])] -= 2*eps
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
    
    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        raise NotImplementedError('Calling reset method of an abstract optimizer')

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        self._second_moment = [self.parameters[2]*self._second_moment[tensor_idx] + (1-self.parameters[2])*torch.pow(grad_subtensor, 2)
                               for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        moment_cor = [moment_tensor/(1 - self.parameters[1] ** self.time) for moment_tensor in self._moment] 
        second_moment_cor = [sm_tensor/(1 - self.parameters[2] ** self.time) for sm_tensor in self._second_moment] 
        return OrderedDict([(subtensor_key, optimized[subtensor_key] - self.parameters[0] * moment_cor[tensor_idx]/\
                             (torch.sqrt(second_moment_cor[tensor_idx]) + self.parameters[3]))
                            for tensor_idx, subtensor_key in enumerate(optimized.keys())])
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