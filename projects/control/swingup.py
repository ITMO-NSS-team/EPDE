#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime
from collections import OrderedDict
from typing import List
import faulthandler

faulthandler.enable()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
import torch
import pickle

from tqdm import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import epde
import epde.globals as global_var

from epde.control import ControlExp, ConstrLocation, ConditionalLoss, ControlConstrEq, ControlConstrNEq
from projects.control.swingup_aux import DMCEnvWrapper, RandomPolicy, CosinePolicy, CosineSignPolicy, \
                                         TwoCosinePolicy, rollout_env, VarTrigTokens # ,, DerivSignFunction
from epde.interface.prepared_tokens import DerivSignFunction

def get_additional_token_families(ctrl, device = 'cpu'):
    angle_trig_tokens = VarTrigTokens('phi', max_power=2, freq_center=1.)
    sgn_tokens = DerivSignFunction(token_type = 'speed_sign', var_name = 'y', token_labels=['sign(dy/dx1)',],
                                   deriv_solver_orders = [[0,],])
    control_var_tokens = epde.interface.prepared_tokens.ControlVarTokens(sample = ctrl, arg_var = [(0, [None,]), (1, [None,]), 
                                                                                                   (0, [0,]), (1, [0,])], 
                                                                         device = device)
    return [angle_trig_tokens, sgn_tokens, control_var_tokens] #  

def epde_discovery(t, x, angle, u, derivs, diff_method = 'FD', data_nn: torch.nn.Sequential = None, device: str = 'cpu'):
    dimensionality = x.ndim - 1
    use_solver = True
    epde_search_obj = epde.EpdeSearch(use_solver = use_solver, dimensionality = dimensionality, boundary = 30,
                                      coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : True}, device=device)    
    
    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max' : 50000})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 4}) 
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})
    else:
        raise ValueError('Incorrect preprocessing tool selected.')

    eps = 5e-7
    popsize = 10
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=30)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    epde_search_obj.fit(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,),
                        equation_terms_max_number=10, data_fun_pow = 2, derivs = [derivs['y'], derivs['phi']],
                        additional_tokens=get_additional_token_families(ctrl=u, device=device),
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-7, 1e-4), data_nn=data_nn) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj

def epde_multisample_discovery(t: List[np.ndarray], x: List[np.ndarray], angle: List[np.ndarray], 
                               derivs: List[np.ndarray], u: List[np.ndarray], diff_method: str = 'FD'):
    dimensionality = 0
    
    print(len(t), len(x), len(angle))
    samples = [[t[i], [x[i], angle[i]]] for i in range(len(t))]    
    epde_search_obj = epde.EpdeMultisample(data_samples=samples, use_solver = False, dimensionality = dimensionality,
                                           boundary = 30, verbose_params = {'show_iter_idx' : True})

    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max' : 50000})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 4}) 
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={}) 
    else:
        raise ValueError('Incorrect preprocessing tool selected.')
    
    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=150)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}
    
    epde_search_obj.fit(samples = samples, variable_names = ['y', 'phi'], max_deriv_order = (2,),
                        equation_terms_max_number = 15, data_fun_pow = 2, deriv_fun_pow=2, derivs = derivs,
                        additional_tokens = get_additional_token_families(ctrl=u), # , control_var_tokens, 
                        equation_factors_max_number = factors_max_number,
                        eq_sparsity_interval = (1e-7, 1e-5)) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj

def translate_equation(t, x, angle, u, derivs: dict, diff_method = 'FD', data_nn = None, device: str = 'cpu'):
    print('Shapes:', x.shape, angle.shape)
    dimensionality = x.ndim - 1
    
    lp_y_terms = [['ctrl{power: 1}',], 
                  ['d^2phi/dx0^2{power: 1}', 'cos(phi){power: 1, freq: 1.}'],
                  ['dphi/dx0{power: 2}', 'sin(phi){power: 1, freq: 1.}'],
                  ['sign(dy/dx1){power: 1}']]
    rp_y_term  = ['d^2y/dx0^2{power: 1}',]
    
    lp_phi_terms = [['d^2phi/dx0^2{power: 1}', 'cos(phi){power: 2, freq: 1.}'],
                    ['sin(phi){power: 1, freq: 1.}'],
                    ['ctrl{power: 1}', 'cos(phi){power: 1, freq: 1.}'], 
                    ['dphi/dx0{power: 2}', 'sin(phi){power: 1, freq: 1.}', 'cos(phi){power: 1, freq: 1.}'],
                    ['sign(dy/dx1){power: 1}', 'cos(phi){power: 1, freq: 1.}'], 
                    ['dphi/dx0{power: 1}']]
    rp_phi_term  = ['d^2phi/dx0^2{power: 1}',]
        


    epde_search_obj = epde.EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 30,
                                      coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : False}, device=device)

    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max' : 50000})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1,
                                                              'polynomial_window' : 3, 'poly_order' : 4})
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={}) 
    else:
        raise ValueError('Incorrect preprocessing tool selected.')

    epde_search_obj.create_pool(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,), derivs = [derivs['y'], derivs['phi']],
                                additional_tokens = get_additional_token_families(ctrl=u, device = device), data_nn = data_nn)

    test = epde.interface.equation_translator.CoeffLessEquation(lp_terms = {'phi': lp_phi_terms, 'y': lp_y_terms},
                                                                rp_term = {'phi': rp_phi_term, 'y': rp_y_term}, 
                                                                pool = epde_search_obj.pool, all_vars = ['y', 'phi'])
    
    def visualize_var(system, variable: str = 'u'):
        val, target, features = system.vals[variable].evaluate(normalize = False, return_val = True)
        print(np.mean(val), np.mean(np.abs(val)), features.shape)
        plt.plot(target, color = 'r', label = 'Equation target')
        plt.plot(val, '-', color = 'k', label = 'Equation discrepancy')
        plt.plot((system.vals[variable].weights_final.reshape((1, -1))[:, :-1] @ features.T + 
                  system.vals[variable].weights_final[-1]).reshape(-1), color = 'b', label = 'Equation features')
        plt.title(f'Discrepancy of the variable {variable}')
        plt.grid()
        plt.legend()
        plt.show()

    visualize_var(test.system, 'y')
    visualize_var(test.system, 'phi')

    return test

def optimize_ctrl(eq: epde.structure.main_structures.SoEq, t: torch.tensor,
                  y_init: float, dy_init: float, phi_init: float, dphi_init: float,
                  ctrl_max: float, stab_der_ord: int, state_nn_pretrained: torch.nn.Sequential,
                  ctrl_nn_pretrained: torch.nn.Sequential, fig_folder: str, device = 'cpu'):
    
    from epde.supplementary import AutogradDeriv

    autograd = AutogradDeriv()

    loc_domain = ConstrLocation(domain_shape = (t.size()[0],), device=device) # Declaring const in the entire domain
    loc_end = ConstrLocation(domain_shape = (t.size()[0],), axis = 0, loc = -1, device=device) # Check format
    print(f'loc_end.flat_idxs : {loc_end.flat_idxs}, device {device}')

    def cosine_cond(x, ref):
        # print(f'cos of {x} is {torch.cos(x)}, while ref was {ref}')
        return torch.abs(torch.cos(x) - ref)
    
    phi_tar_constr = ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 1., device=device), # Better processing for periodic
                                     indices = loc_end, deriv_axes=[None,], deriv_method = autograd, nn_output=1, 
                                     estim_func=cosine_cond, device=device)
    dphi_tar_constr = ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 0, device=device),
                                      indices = loc_end, deriv_axes=[0,], deriv_method = autograd, nn_output=1, device=device)
    contr_constr = ControlConstrEq(val = torch.full_like(input = t, fill_value = 0., device=device),
                                   indices = loc_domain, deriv_axes=[None,], deriv_method = autograd, nn_output=0, device=device)

    contr_right_bnd = ControlConstrNEq(val = torch.full_like(input = t, fill_value = ctrl_max, device=device), sign='<',
                                       indices = loc_domain, deriv_method = autograd, nn_output=0, device=device)
    contr_left_bnd = ControlConstrNEq(val = torch.full_like(input = t, fill_value = -ctrl_max, device=device), sign='>',
                                      indices = loc_domain, deriv_method = autograd, nn_output=0, device=device)
    
    loss = ConditionalLoss([(1e6, phi_tar_constr, 0),
                            (10., dphi_tar_constr, 0), 
                            (0.001, contr_constr, 1),
                            (100., contr_right_bnd, 1),
                            (100., contr_left_bnd, 1)])
    
    optimizer = ControlExp(loss=loss, device=device)
    
    def get_ode_bop(key, var, term, grid_loc, value):
        bop = epde.interface.solver_integration.BOPElement(axis = 0, key = key, term = term,
                                                           power = 1, var = var, device = device)
        if isinstance(grid_loc, float):
            bop_grd_np = np.array([[grid_loc,]])
            bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor)).to(device)
        elif isinstance(grid_loc, torch.Tensor):
            bop.set_grid(grid_loc.reshape((1, 1)).type(torch.FloatTensor))
        else:
            raise TypeError('Incorret value type, expected float or torch.Tensor.')
        bop.values = torch.from_numpy(np.array([[value,]])).float().to(device)
        return bop

    bop_y = get_ode_bop('y', 0, [None], t[0, 0], y_init)
    bop_dy = get_ode_bop('y', 0, [0,], t[0, 0], dy_init)

    bop_phi = get_ode_bop('phi', 0, [None], t[0, 0], phi_init)
    bop_dphi = get_ode_bop('phi', 0, [0,], t[0, 0], dphi_init)

    optimizer.system = eq.system

    # optimizer.set_control_optim_params()

    solver_params = {'full':     {'training_params': {'epochs': 1000,}, 'optimizer_params': {'params': {'lr': 1e-5}}}, 
                     'abridged': {'training_params': {'epochs': 300,}, 'optimizer_params': {'params': {'lr': 5e-5}}}}
    
    state_nn, ctrl_net, ctrl_pred, hist = optimizer.train_pinn(bc_operators = [(bop_y(), 0.1),
                                                                               (bop_dy(), 0.1),
                                                                               (bop_phi(), 0.1),
                                                                               (bop_dphi(), 0.1)],
                                                               grids = [t,], n_control = 1., 
                                                               state_net = state_nn_pretrained, 
                                                               opt_params = [0.0001, 0.9, 0.999, 1e-8],
                                                               control_net = ctrl_nn_pretrained, epochs = 55,
                                                               fig_folder = fig_folder, eps=1e-1,
                                                               solver_params = solver_params)

    return state_nn, ctrl_net, ctrl_pred, hist

def prepare_trajectories(n_sample: int = 250, single_sample: bool = True):
    env_config = {'domain_name': "cartpole",
                  'task_name': "swingup",
                  'frame_skip': 1,
                  'from_pixels': False}

    cart_env = DMCEnvWrapper(env_config)
    random_policy = RandomPolicy(cart_env.action_space)
    cosine_policy = CosinePolicy(period=180, amplitude=0.004)
    cosine_signum_policy = CosineSignPolicy(period=180, amplitude=0.002)
    two_cosine_policy = TwoCosinePolicy(180, 90, 0.002)

    step = 0.01
    t = np.linspace(0, step*n_sample, num = n_sample, endpoint=False)

    return t, rollout_env(cart_env, two_cosine_policy, n_steps = 500, 
                          n_steps_reset=1000)

def general(experiment, res_folder, single_sample: bool = True, discover: bool = True, device: str = 'cpu'):
    env_config = {'domain_name': "cartpole",
                  'task_name': "swingup",
                  'frame_skip': 1,
                  'from_pixels': False}

    cart_env = DMCEnvWrapper(env_config)
    random_policy = RandomPolicy(cart_env.action_space)
    cosine_policy = CosinePolicy(period=180, amplitude=0.004)
    cosine_signum_policy = CosineSignPolicy(period=180, amplitude=0.002)
    two_cosine_policy = TwoCosinePolicy(360, 180, 0.001)

    if single_sample:
        traj_obs, traj_acts, traj_rews = rollout_env(cart_env, two_cosine_policy, n_steps = 500, 
                                                    n_steps_reset=1000)
        
        
        def get_angle_rot(cosine, sine):
            if sine >= 0 and cosine >= 0:
                return np.arccos(cosine)
            elif sine >= 0 and cosine < 0:
                return np.arccos(cosine)
            elif sine < 0 and cosine >= 0:
                return 2*np.pi + np.arcsin(sine)
            else:
                return 2*np.pi - np.arccos(cosine)

        xs, x_ds = [], []
        angles, angles_d = [], []
        us = []
        
        for idx, _ in enumerate(traj_acts[:1]):
            step = 0.01
            t = np.linspace(0, step*traj_acts[0].size, num = traj_acts[0].size, endpoint=False)[1:-1]
            
            angles_calc = np.vectorize(get_angle_rot)
            angle, angle_d = angles_calc(traj_obs[idx][:, 1], traj_obs[idx][:, 2]), traj_obs[idx][:, 4]
            angle_dd = (angle_d[2:] - angle_d[:-2])/(2*step)
            angle = angle[1:-1] ; angle_d = angle_d[1:-1]
            
            x, x_d = traj_obs[idx][:, 0], traj_obs[idx][:, 3]
            x_dd = (x_d[2:] - x_d[:-2])/(2*step)
            u = traj_acts[idx].reshape(x.shape)        
            
            x = x[1:-1] ; x_d = x_d[1:-1]; u = u[1:-1]
            derivs = {'y': np.vstack((x_d, x_dd)).T, 'phi': np.vstack((angle_d, angle_dd)).T}


            plt.plot(u)
            plt.grid()
            plt.title('Actions, taken as control.')
            plt.show()

            plt.plot(angle, color = 'k', label = 'Pole angle, rad.')
            plt.plot(derivs['phi'][:, 0], color = 'r', label = 'Angle deriv, rad./s')
            plt.plot(derivs['phi'][:, 1], color = 'b', label = 'Angle deriv, rad./s')
            plt.legend()
            plt.grid()
            plt.title('Inputs, angle')
            plt.show()

            plt.plot(x, color = 'k', label = 'Cart position.')
            plt.plot(derivs['y'][:, 0], color = 'r', label = 'Cart pos deriv, rad./s')
            plt.plot(derivs['y'][:, 1], color = 'b', label = 'Cart pos deriv, rad./s')
            plt.legend()
            plt.grid()
            plt.title('Inputs, cart position')
            plt.show()

            plt.plot(np.cos(angle), color = 'k', label = 'Sine')
            plt.plot(np.sin(angle), color = 'r', label = 'Cosine')
            plt.plot(1 - (np.sin(angle)**2 + np.cos(angle)**2), '-', color = 'b', label = 'Trig identity check')

            plt.legend()
            plt.grid()
            plt.title('Angle cosine')
            plt.show()


            xs.append(x)
            x_ds.append(x_d)
            angles.append(angle)
            angles_d.append(angle_d)
            us.append(u)

            if device == 'cpu':
                fname = os.path.join(res_folder, f"data_ann_{experiment}_cpu.pickle")
            else:
                fname = os.path.join(res_folder, f"data_ann_{experiment}_cuda.pickle")            
            try:
                with open(fname, 'rb') as data_input_file:  
                    data_nn = pickle.load(data_input_file)
                data_nn = data_nn.to(device)
                save_nn = False
            except FileNotFoundError:
                print(f'No model located, with name {fname}')
                data_nn = None
                save_nn = True

            if discover:
                res = epde_discovery(t, x, angle, u, derivs, 'FD', data_nn = data_nn, device = device)
            else:
                res = translate_equation(t, x, angle, u, derivs, 'FD', data_nn = data_nn, device=device)
            if save_nn:
                if device == 'cpu':
                    fname =  os.path.join(res_folder, f"data_ann_{experiment}_cpu.pickle")
                else:
                    fname = os.path.join(res_folder, f"data_ann_{experiment}_cuda.pickle")
                with open(fname, 'wb') as output_file:
                    pickle.dump(epde.globals.solution_guess_nn, output_file)

            return t, u, np.stack([x, x_d, angle, angle_d]).T, res
    else:
        traj_obs_sc, traj_acts_sc, _ = rollout_env(cart_env, cosine_signum_policy, n_steps = 500, 
                                                   n_steps_reset=1000)

        traj_obs_tc, traj_acts_tc, _ = rollout_env(cart_env, two_cosine_policy, n_steps = 250, 
                                                   n_steps_reset=1000)
        
        traj_obs_c, traj_acts_c, _ = rollout_env(cart_env, cosine_policy, n_steps = 250, 
                                                 n_steps_reset=1000)
        
        
        def get_angle_rot(cosine, sine):
            if sine >= 0 and cosine >= 0:
                return np.arccos(cosine)
            elif sine >= 0 and cosine < 0:
                return np.arccos(cosine)
            elif sine < 0 and cosine >= 0:
                return 2*np.pi + np.arcsin(sine)
            else:
                return 2*np.pi - np.arccos(cosine)

        xs, x_ds = [], []
        angles, angles_d = [], []
        
        
        def prepare_data(traj_acts, traj_obs):
            step = 0.01
            t = np.linspace(0, step*traj_acts[0].size, num = traj_acts[0].size, endpoint=False)[1:-1]
            
            angles_calc = np.vectorize(get_angle_rot)
            angle, angle_d = angles_calc(traj_obs[0][:, 1], traj_obs[0][:, 2]), traj_obs[0][:, 4]
            angle_dd = (angle_d[2:] - angle_d[:-2])/(2*step)
            angle = angle[1:-1] ; angle_d = angle_d[1:-1]
            
            x, x_d = traj_obs[0][:, 0], traj_obs[0][:, 3]
            x_dd = (x_d[2:] - x_d[:-2])/(2*step)
            u = traj_acts[0].reshape(x.shape)        
            
            x = x[1:-1] ; x_d = x_d[1:-1]; u = u[1:-1]
            derivs = {'y': np.vstack((x_d, x_dd)).T, 'phi': np.vstack((angle_d, angle_dd)).T}
            return t, x, angle, derivs, u

        tc_comb = prepare_data(traj_acts_tc, traj_obs_tc)
        sc_comb = prepare_data(traj_acts_sc, traj_obs_sc)
        c_comb = prepare_data(traj_acts_c, traj_obs_c)

        samples_t = [[tc_comb[0],], [sc_comb[0],], [c_comb[0],]]
        samples_pos = [tc_comb[1], sc_comb[1], c_comb[1]]
        samples_angle = [tc_comb[2],  sc_comb[2],  c_comb[2]]

        samples_derivs = [[tc_comb[3]['y'], sc_comb[3]['y'], c_comb[3]['y']],
                          [tc_comb[3]['phi'], sc_comb[3]['phi'], c_comb[3]['phi']]]

        samples_u = [tc_comb[4], sc_comb[4], c_comb[4]]

        return tc_comb[0], tc_comb[4], np.stack([tc_comb[1], tc_comb[3]['y'], tc_comb[2], tc_comb[3]['phi']]).T, \
               epde_multisample_discovery(samples_t, samples_pos,
                                          samples_angle, samples_derivs,
                                          samples_u, 'FD')

if __name__ == '__main__':
    import pickle

    experiment = 'swingup'
    explicit_cpu = False
    device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'
    print(f'Working on {device}')

    res_folder = '/home/mikemaslyaev/Documents/EPDE/projects/control'

    fig_folder = os.path.join(res_folder, 'figs')

    only_prepare = False
    discover = False
    if only_prepare:
        t, traj_data = prepare_trajectories()
        print(f'Acts len is {len(traj_data[0])}, while shape {traj_data[0][0].shape}')
        assert(t.size == traj_data[0][0].shape[0]), 'Incorrect t detected'
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/t.npy', arr = t)
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/state.npy', arr = traj_data[0][0])
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/acts.npy', arr = traj_data[1][0])
    else:

        t, ctrl, solution, res = general(experiment, res_folder=res_folder, discover=discover, device = device)

        example_sol = epde.globals.solution_guess_nn(torch.from_numpy(t).reshape((-1, 1)).float().to(device))
        epde.globals.solution_guess_nn.to(device)
        print(f'example_sol: {type(example_sol)}, {example_sol.shape}, {example_sol.get_device()}')

        def create_shallow_nn(arg_num: int = 1, output_num: int = 1, device = 'cpu') -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
            hidden_neurons = 30
            layers = [torch.nn.Linear(arg_num, hidden_neurons, device=device),
                      torch.nn.Tanh(), # ReLU(),
                      torch.nn.Linear(hidden_neurons, output_num, device=device)]
            return torch.nn.Sequential(*layers)
        
        def create_deep_nn(arg_num: int = 1, output_num: int = 1, device = 'cpu') -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
            hidden_neurons = 18
            layers = [torch.nn.Linear(arg_num, hidden_neurons, device=device),
                      torch.nn.Tanh(),
                      torch.nn.Linear(hidden_neurons, hidden_neurons, device=device),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_neurons, output_num, device=device)]
            return torch.nn.Sequential(*layers)        

        time_exp_start = datetime.datetime.now()
        
        from epde.supplementary import define_derivatives
        from epde.preprocessing.preprocessor_setups import PreprocessorSetup
        from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

        
        def prepare_derivs(var_name: str, var_array: np.ndarray, grid: np.ndarray, max_order: tuple = (2,)):
            default_preprocessor_type = 'FD'
            preprocessor_kwargs = {}

            setup = PreprocessorSetup()
            builder = ConcretePrepBuilder()
            setup.builder = builder
            
            if default_preprocessor_type == 'ANN':
                setup.build_ANN_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'poly':
                setup.build_poly_diff_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'spectral':
                setup.build_spectral_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'FD':
                setup.build_FD_preprocessing(**preprocessor_kwargs)

            preprocessor_pipeline = setup.builder.prep_pipeline

            if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
                preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None
                
            max_order = (1,)
            deriv_names, _ = define_derivatives(var_name, dimensionality=var_array.ndim,
                                                max_order=max_order)

            _, derivatives_n = preprocessor_pipeline.run(var_array, grid=[grid,],
                                                        max_order=max_order)
            return deriv_names, derivatives_n
        
        # der_names_u, derivatives_u = prepare_derivs('u', var_array = solution[:, 0], grid = t)
        # der_names_v, derivatives_v = prepare_derivs('v', var_array = solution[:, 2], grid = t)
        
        args = torch.from_numpy(solution).float().to(device)
        print(f'args.shape is {args.shape}')


        nn = 'shallow'
        load_ctrl = False

        if device == 'cpu':
            ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cpu.pickle")
        else:
            ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cuda.pickle")
        if load_ctrl:
            with open(ctrl_fname, 'rb') as ctrl_input_file:  
                ctrl_ann = pickle.load(ctrl_input_file)
        else:
            nn_method = create_shallow_nn if nn == 'shallow' else create_deep_nn
            ctrl_args = [solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]]
            ctrl_ann = epde.supplementary.train_ann(args=ctrl_args,
                                                    data = ctrl, 
                                                    epochs_max = 5e7, 
                                                    dim = len(ctrl_args), 
                                                    model = nn_method(len(ctrl_args), 1, device=device),
                                                    device = device)

            with open(ctrl_fname, 'wb') as ctrl_output_file:  
                pickle.dump(ctrl_ann, file = ctrl_output_file)
                
        res = optimize_ctrl(res, torch.from_numpy(t).reshape((-1, 1)).float().to(device),
                            y_init=solution[0, 0], dy_init=solution[0, 1], phi_init=solution[0, 2], dphi_init=solution[0, 3],
                            ctrl_max = 1, stab_der_ord = 2, 
                            state_nn_pretrained=epde.globals.solution_guess_nn, ctrl_nn_pretrained=ctrl_ann, 
                            fig_folder=fig_folder, device=device)

        savename = f'res_{time_exp_start.month}_{time_exp_start.day}_at_{time_exp_start.hour}_{time_exp_start.minute}_{experiment}.pickle'

        with open(os.path.join(res_folder, savename), 'wb') as output_file:  
            pickle.dump(res, output_file)                