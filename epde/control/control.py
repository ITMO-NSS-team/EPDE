import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt
import datetime
import os 
from warnings import warn

import gc
from typing import List, Union
from copy import deepcopy


import epde.globals as global_var
from epde.interface.interface import ExperimentCombiner
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.integrate import SolverAdapter, OdeintAdapter, BOPElement 

from epde.control.constr import ConditionalLoss
from epde.control.utils import prepare_control_inputs, eps_increment_diff
from epde.control.optim import AdamOptimizer, CoordDescentOptimizer

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
        bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var, device = device)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor).to(device))
        bop.values = torch.from_numpy(np.array([[value,]])).float().to(device)
        return bop

    def set_solver_params(self, use_pinn: bool = True, mode: str = 'autograd', compiling_params: dict = {}, 
                          optimizer_params: dict = {}, cache_params: dict = {}, early_stopping_params: dict = {}, 
                          plotting_params: dict = {}, training_params: dict = {'epochs': 150,}, 
                          use_cache: bool = False, use_fourier: bool = False, #  5*1e0
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False): #  device: str = 'cpu'
        self._use_pinn = use_pinn
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

    def get_solver_adapter(self, net: torch.nn.Sequential = None):
        if self._use_pinn:
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
        else:
            try:
                self._solver_params['method']
            except KeyError:
                self._solver_params['method'] = 'Radau'
            adapter = OdeintAdapter(method = self._solver_params['method'])
        return adapter

    @staticmethod
    def finite_diff_calculation(system, adapter, loc, control_loss, state_net: torch.nn.Sequential,
                                bc_operators, grids: list, solver_params: dict, eps: float):
        '''
        Calculate finite-differecnce approximation of gradient in respect to the specified parameter.

        '''
        # Calculating loss in p[i]+eps:
        ctrl_dict_prev = global_var.control_nn.net.state_dict()       
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_dict_prev,
                                          loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        if isinstance(adapter, SolverAdapter):
            adapter.set_net(deepcopy(state_net))
        solver_loss_forward, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                             boundary_conditions = bc_operators,
                                             mode = solver_params['mode'],
                                             use_cache = solver_params['use_cache'],
                                             use_fourier = solver_params['use_fourier'],
                                             fourier_params = solver_params['fourier_params'],
                                             use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        
        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args)
        loss_forward = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])

        # Calculating loss in p[i]-eps:
        ctrl_dict_prev = global_var.control_nn.net.state_dict() # deepcopy()
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_dict_prev,
                                          loc = loc, forward=False, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        if isinstance(adapter, SolverAdapter):
            adapter.set_net(deepcopy(state_net))
        solver_loss_backward, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                                       boundary_conditions = bc_operators,
                                                       mode = solver_params['mode'],
                                                       use_cache = solver_params['use_cache'],
                                                       use_fourier = solver_params['use_fourier'],
                                                       fourier_params = solver_params['fourier_params'],
                                                       use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])

        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args)
        loss_back = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])
        
        # Restore values of the control NN parameters
        ctrl_nn_dict = global_var.control_nn.net.state_dict()
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_nn_dict,
                                          loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        loss_max = 1e-3
        if solver_loss_backward > loss_max or solver_loss_forward > loss_max:
            warn(f'High solver loss occured: backward {solver_loss_backward} and forward {solver_loss_forward}.')
        loss_alpha = 1e1

        with torch.no_grad():
            delta = loss_forward - loss_back
            if torch.abs(delta) < solver_loss_forward+solver_loss_backward:
                res = 0*delta
            else:
                res = delta/(2*eps*(1+loss_alpha*(solver_loss_forward+solver_loss_backward)))
        # print(f'loss_forward {loss_forward, solver_loss_forward}, loss_backward {loss_back, solver_loss_backward}, res {res}')
        # print(f'loss_alpha*(solver_loss_forward+solver_loss_backward) {loss_alpha*(solver_loss_forward+solver_loss_backward)}')
        return res
        

    def feedback(self, bc_operators: List[Union[dict, float]], grids: List[Union[np.ndarray, torch.Tensor]], 
                 n_control: int = 1, epochs: int = 1e2, state_net: torch.nn.Sequential = None, 
                 opt_params: List[float] = [0.01, 0.9, 0.999, 1e-8],
                 control_net: torch.nn.Sequential = None, fig_folder: str = None, 
                 LV_exp: bool = True, eps: float = 1e-2, solver_params: dict = {}):
        def modify_bc(operator: dict, scale: Union[float, torch.Tensor]) -> dict:
            noised_operator = deepcopy(operator)
            noised_operator['bnd_val'] = torch.normal(operator['bnd_val'], scale).to(self._device)
            return noised_operator

        # Properly formulate training approach
        t = 0

        loss_hist = []
        stop_training = False

        time = datetime.datetime.now()

        if isinstance(state_net, torch.nn.Sequential): self._state_net = state_net
        global_var.reset_control_nn(n_control = n_control, ann = control_net, 
                                    ctrl_args = global_var.control_nn.net_args, device = self._device)

        # TODO Refactor hook: To optimize the net in global variables is a terrific approach, rethink it

        if isinstance(grids[0], np.ndarray):
            grids_merged = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T.to(self._device)
        elif isinstance(grids[0], torch.Tensor):
            grids_merged = torch.cat([subgrid.reshape(-1, 1) for subgrid in grids], dim = 1).float()
        grids_merged.to(device=self._device)

        grad_tensors = deepcopy(global_var.control_nn.net.state_dict())

        min_loss = np.inf
        self._best_control_params = global_var.control_nn.net.state_dict()

        # optimizer = AdamOptimizer(optimized = global_var.control_nn.net.state_dict(), parameters = opt_params)
        optimizer = CoordDescentOptimizer(optimized = global_var.control_nn.net.state_dict(), parameters = opt_params)

        self.set_solver_params(**solver_params['full'])
        adapter = self.get_solver_adapter(None)
        if isinstance(adapter, SolverAdapter): 
            adapter.set_net(deepcopy(self._state_net))

        sampled_bc = [modify_bc(operator, noise_std) for operator, noise_std in bc_operators]
        loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                     boundary_conditions = sampled_bc,
                                                     mode = self._solver_params['mode'],
                                                     use_cache = self._solver_params['use_cache'],
                                                     use_fourier = self._solver_params['use_fourier'],
                                                     fourier_params = self._solver_params['fourier_params'],
                                                     use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

        control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args)
        loss = self.loss([self._state_net, global_var.control_nn.net], [grids_merged, control_inputs])
        print('current loss is ', loss, 'model undertrained with loss of ', loss_pinn)

        while t < epochs and not stop_training:
            self.set_solver_params(**solver_params['abridged'])            
            adapter = self.get_solver_adapter(None)
            sampled_bc = [modify_bc(operator, noise_std) for operator, noise_std in bc_operators]

            # self.set_solver_params(**solver_params['full'])
            adapter = self.get_solver_adapter(None)
            if isinstance(adapter, SolverAdapter):            
                adapter.set_net(self._state_net)
            loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                         boundary_conditions = sampled_bc,
                                                         mode = self._solver_params['mode'],
                                                         use_cache = self._solver_params['use_cache'],
                                                         use_fourier = self._solver_params['use_fourier'],
                                                         fourier_params = self._solver_params['fourier_params'],
                                                         use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

            control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args)
            loss = self.loss([self._state_net, global_var.control_nn.net], [grids_merged, control_inputs])            
            self._state_net = model
            self.set_solver_params(**solver_params['abridged'])

            global_var.control_nn.net.load_state_dict(self._best_control_params)
            state_net  = deepcopy(self._state_net)
            print(f'Control function optimization epoch {t}.')
            for param_key, param_tensor in grad_tensors.items():
                print(f'Optimizing {param_key}: shape is {param_tensor.shape}')
                if len(param_tensor.size()) == 1:
                    for param_idx, _ in enumerate(param_tensor):
                        loc = (param_key, param_idx)
                        grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()                  
                        grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system,
                                                                                     adapter = adapter,
                                                                                     loc = loc, control_loss = self.loss,
                                                                                     state_net = state_net,
                                                                                     bc_operators = sampled_bc,
                                                                                     grids = [grids, grids_merged],
                                                                                     solver_params = self._solver_params,
                                                                                     eps = eps)
                        if optimizer.behavior == 'Coordinate':
                            state_dict_prev = global_var.control_nn.net.state_dict()
                            state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev, loc = loc)
                            global_var.control_nn.net.load_state_dict(state_dict)                                            
                elif len(param_tensor.size()) == 2:
                    for param_outer_idx, _ in enumerate(param_tensor):
                        for param_inner_idx, _ in enumerate(param_tensor[0]):
                            loc = (param_key, param_outer_idx, param_inner_idx)
                            grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()
                            grad_tensors[loc[0]][tuple(loc[1:])] = self.finite_diff_calculation(system = self.system,
                                                                                                adapter = adapter,
                                                                                                loc = loc,
                                                                                                control_loss = self.loss,
                                                                                                state_net = state_net,
                                                                                                bc_operators = sampled_bc,
                                                                                                grids = [grids, grids_merged],
                                                                                                solver_params = self._solver_params,
                                                                                                eps = eps)
                            if optimizer.behavior == 'Coordinate':
                                state_dict_prev = global_var.control_nn.net.state_dict()
                                state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev, loc = loc)
                                global_var.control_nn.net.load_state_dict(state_dict)                               
                else:
                    raise Exception(f'Incorrect shape of weights/bias. Got {param_tensor.size()} tensor.')
            if optimizer.behavior == 'Gradient':
                state_dict_prev = global_var.control_nn.net.state_dict()
                state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev)
                global_var.control_nn.net.load_state_dict(state_dict)
            del state_dict, state_dict_prev

            self.set_solver_params(**solver_params['full'])
            adapter = self.get_solver_adapter(None)
            if isinstance(adapter, SolverAdapter):
                adapter.set_net(self._state_net)
            loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
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
            print('current loss is ', loss, 'model undertrained with loss of ', loss_pinn)
            # if loss < min_loss:
            #     min_loss = loss
            self._best_control_params = global_var.control_nn.net.state_dict()
            loss_hist.append(loss)
            
            if fig_folder is not None and LV_exp:
                plt.figure(figsize=(11, 6))
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 0], color = 'k')
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 1], color = 'r')
                plt.plot(grids_merged.cpu().detach().numpy(), global_var.control_nn.net(control_inputs).cpu().detach().numpy(),
                         color = 'tab:orange')
                plt.grid()
                frame_name = f'Exp_{time.month}_{time.day}_at_{time.hour}_{time.minute}_{t}.png'
                plt.savefig(os.path.join(fig_folder, frame_name))
            
            if fig_folder is not None:
                exp_res = {'state'   : control_inputs.cpu().detach().numpy(),
                           'control' : global_var.control_nn.net(control_inputs).cpu().detach().numpy()}
                frame_name = f'Exp_{time.month}_{time.day}_at_{time.hour}_{time.minute}_{t}.pickle'
                with open(os.path.join(fig_folder, frame_name), 'wb') as ctrl_output_file:  
                    pickle.dump(exp_res, file = ctrl_output_file)

            gc.collect()
            t += 1

        control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args)
        ctrl_pred = global_var.control_nn.net(control_inputs)

        return self._state_net, global_var.control_nn.net, ctrl_pred, loss_hist

    def time_based(self, bc_operators: List[Union[dict, float]], grids: List[Union[np.ndarray, torch.Tensor]], 
                   n_control: int = 1, epochs: int = 1e2, state_net: torch.nn.Sequential = None, 
                   opt_params: List[float] = [0.01, 0.9, 0.999, 1e-8],
                   control_net: torch.nn.Sequential = None, fig_folder: str = None, 
                   LV_exp: bool = True, eps: float = 1e-2, solver_params: dict = {}):        
        self.set_solver_params(**solver_params['full'])
        adapter = self.get_solver_adapter(None)
        solver_form = self.system

        raise NotImplementedError()