#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:43:03 2024

@author: maslyaev
"""
import os
import sys
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
import epde.interface.control_utils as control_utils
import epde.globals as global_var
from projects.control.swingup_aux import DMCEnvWrapper, RandomPolicy, CosinePolicy, CosineSignPolicy, \
                                         TwoCosinePolicy, rollout_env, VarTrigTokens # ,, DerivSignFunction
from epde.interface.prepared_tokens import DerivSignFunction

def get_additional_token_families(ctrl):
    angle_trig_tokens = VarTrigTokens('phi', max_power=2, freq_center=1.)
    sgn_tokens = DerivSignFunction(token_type = 'speed_sign', var_name = 'y', token_labels=['sign(dy/dx1)',],
                                   deriv_solver_orders = [[0,],])
    control_var_tokens = epde.interface.prepared_tokens.ControlVarTokens(sample = ctrl, arg_var = [(0, [None,]), (1, [None,]), 
                                                                                                   (0, [0,]), (1, [0,])])
    return [angle_trig_tokens, sgn_tokens, control_var_tokens] 

def epde_discovery(t, x, angle, u, derivs, diff_method = 'FD', data_nn: torch.nn.Sequential = None, device: str = 'cpu'):
    dimensionality = x.ndim - 1
    
    epde_search_obj = epde.EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 30,
                                      coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : True})    
    
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
    
    # angle_cos = np.cos(angle)
    # angle_sin = np.sin(angle)
    
    # angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
    #                                            {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
    #                                            OrderedDict([('power', (1, 3))]), {'power': 0}, meaningful=True)
    # control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u}, OrderedDict([('power', (1, 1))]),
    #                                             {'power': 0}, meaningful=True)
    # sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
    #                                     {'sgn(dy)' : np.sign(derivs['y'][:, 0]),
    #                                      'sgn(ddy)' : np.sign(derivs['y'][:, 1]),}, 
    #                                     OrderedDict([('power', (1, 1))]), {'power': 0}, meaningful=True)    

    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=5)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    # custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)
    epde_search_obj.create_pool(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,),
                                data_fun_pow = 2, derivs = [derivs['y'], derivs['phi']],
                                additional_tokens=get_additional_token_families(ctrl=u), data_nn=data_nn)
    
    if data_nn is None:
        data_nn = global_var.solution_guess_nn
        if device == 'cpu':
            fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cpu.pickle"
        else:
            fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cuda.pickle"
        with open(fname, 'wb') as output_file:
            pickle.dump(data_nn, output_file)

    epde_search_obj.fit(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,),
                        equation_terms_max_number=10, data_fun_pow = 2, derivs = [derivs['y'], derivs['phi']],
                        additional_tokens=get_additional_token_families(ctrl=u),
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-7, 1e-5), data_nn=data_nn) # TODO: narrow sparsity interval, reduce the population size
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
    
    # angle_cos = np.cos(np.concatenate([angle_arr for angle_arr in angle]))
    # angle_sin = np.sin(np.concatenate([angle_arr for angle_arr in angle]))
    
    # angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
    #                                            {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
    #                                            OrderedDict([('power', (1, 3))]), {'power': 0}, meaningful=True, 
    #                                            unique_token_type=False, unique_specific_token=False, non_default_power=True)
    # u_concat = np.concatenate(u)
    # print(f'u_concat.shape is {u_concat.shape}')
    # control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u_concat}, OrderedDict([('power', (1, 1))]),
    #                                             {'power': 0}, meaningful=True)
    # der_y = np.concatenate(derivs[0])
    # print(f'der_y shape is {der_y.shape}')
    # sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
    #                                     {'sgn(dy)' : np.sign(der_y[:, 0]),
    #                                      'sgn(ddy)' : np.sign(der_y[:, 1]),}, 
    #                                     OrderedDict([('power', (1, 1))]), {'power': 0}, meaningful=True)    

    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=150)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    # custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)

    epde_search_obj.fit(samples = samples, variable_names = ['y', 'phi'], max_deriv_order = (2,),
                        equation_terms_max_number = 15, data_fun_pow = 2, deriv_fun_pow=2, derivs = derivs,
                        additional_tokens = get_additional_token_families(ctrl=u), # , control_var_tokens, 
                        equation_factors_max_number = factors_max_number,
                        eq_sparsity_interval = (1e-7, 1e-5)) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj

def translate_equation(t, x, angle, u, derivs: dict, diff_method = 'FD'):
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
        


    epde_search_obj = epde.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 30,
                                      coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : False})

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
    
    # angle_cos = np.cos(angle)
    # angle_sin = np.sin(angle)
    
    # angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
    #                                            {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
    #                                            OrderedDict([('power', (1, 3))]), {'power': 0})
    # control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u}, OrderedDict([('power', (1, 1))]),
    #                                             {'power': 0})
    # sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
    #                                     {'sgn(dy)' : np.sign(derivs['y'][:, 0]),
    #                                      'sgn(ddy)' : np.sign(derivs['y'][:, 1]),}, 
    #                                     OrderedDict([('power', (1, 1))]), {'power': 0})
    


    epde_search_obj.create_pool(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,), derivs = [derivs['y'], derivs['phi']],
                                additional_tokens = get_additional_token_families(ctrl=u))

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
                  y_left: float, y_right: float, stab_der_ord: int,
                  state_nn_pretrained: torch.nn.Sequential, ctrl_nn_pretrained: torch.nn.Sequential, 
                  fig_folder: str, device = 'cpu'):
    
    from epde.supplementary import AutogradDeriv
    autograd = AutogradDeriv()

    loc = control_utils.ConstrLocation(domain_shape = (t.size()[0],), device=device) # Declaring const in the entire domain
    cosine_cond = lambda x, ref: torch.cos(x) - ref
    phi_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 1., device=device), # Better processing for periodic
                                                   indices = loc, deriv_axes=[None,], deriv_method = autograd, nn_output=0, 
                                                   estim_func=cosine_cond)
    dphi_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 0, device=device),
                                                    indices = loc, deriv_axes=[0,], deriv_method = autograd, nn_output=1)
    contr_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = 0., device=device),
                                                 indices = loc, deriv_axes=[None,], deriv_method = autograd, nn_output=0)

    u_var_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=0)
    v_var_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=1)
    contr_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=0)    

    
    loss = control_utils.ConditionalLoss([(1., y_tar_constr, 0),
                                          (1., v_tar_constr, 0), 
                                          (0.001, contr_constr, 1),
                                          (10., u_var_non_neg, 0),
                                          (10., v_var_non_neg, 0),
                                          (10., contr_non_neg, 1)])
    optimizer = control_utils.ControlExp(loss=loss, device=device)
    
    def get_ode_bop(key, var, term, grid_loc, value):
        bop = epde.interface.solver_integration.BOPElement(axis = 0, key = key, term = term,
                                                           power = 1, var = var)
        if isinstance(grid_loc, float):
            bop_grd_np = np.array([[grid_loc,]])
            bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor)).to(device)
        elif isinstance(grid_loc, torch.Tensor):
            bop.set_grid(grid_loc.reshape((1, 1)).type(torch.FloatTensor)) # What is the correct shape here?
        else:
            raise TypeError('Incorret value type, expected float or torch.Tensor.')
        bop.values = torch.from_numpy(np.array([[value,]])).float().to(device)
        return bop

    bop_y = get_ode_bop('y', 0, [None], t[0, 0], u_init)
    bop_dy = get_ode_bop('y', 0, [0,], t[0, 0], v_init)

    optimizer.system = eq

    optimizer.set_control_optim_params()

    optimizer.set_solver_params(training_params = {'epochs': 200,})

    state_nn, ctrl_net, ctrl_pred, hist = optimizer.train_pinn(bc_operators = [(bop_u(device=device), 0.001), 
                                                                               (bop_v(device=device), 0.001)], 
                                                               grids = [t,], n_control = 1., 
                                                               state_net = state_nn_pretrained, 
                                                               opt_params = [0.005, 0.9, 0.999, 1e-8],
                                                               control_net = ctrl_nn_pretrained, epochs = 50,
                                                               fig_folder = fig_folder)

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

    return t, rollout_env(cart_env, two_cosine_policy, n_steps = 250, 
                          n_steps_reset=1000)

def general(single_sample: bool = True, discover: bool = True):
    env_config = {'domain_name': "cartpole",
                  'task_name': "swingup",
                  'frame_skip': 1,
                  'from_pixels': False}

    cart_env = DMCEnvWrapper(env_config)
    random_policy = RandomPolicy(cart_env.action_space)
    cosine_policy = CosinePolicy(period=180, amplitude=0.004)
    cosine_signum_policy = CosineSignPolicy(period=180, amplitude=0.002)
    two_cosine_policy = TwoCosinePolicy(180, 90, 0.002)

    if single_sample:
        traj_obs, traj_acts, traj_rews = rollout_env(cart_env, two_cosine_policy, n_steps = 250, 
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

            device = 'cpu'
            try:
                if device == 'cpu':
                    fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cpu.pickle"
                else:
                    fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cuda.pickle"
                with open(fname, 'rb') as data_input_file:  
                    data_nn = pickle.load(data_input_file)
            except:
                data_nn = None

            if discover:
                res = epde_discovery(t, x, angle, u, derivs, 'FD', data_nn = data_nn, device = device)
            else:
                res = translate_equation(t, x, angle, u, derivs, 'FD')
    else:
        traj_obs_sc, traj_acts_sc, traj_rews_sc = rollout_env(cart_env, cosine_signum_policy, n_steps = 250, 
                                                            n_steps_reset=1000)

        traj_obs_tc, traj_acts_tc, traj_rews_tc = rollout_env(cart_env, two_cosine_policy, n_steps = 250, 
                                                            n_steps_reset=1000)
        
        traj_obs_c, traj_acts_c, traj_rews_c = rollout_env(cart_env, cosine_policy, n_steps = 250, 
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

        # samples = [[tc_comb[0], tc_comb[1], tc_comb[2]], 
        #            [sc_comb[0], sc_comb[1], sc_comb[2]],
        #            [c_comb[0],  c_comb[1],  c_comb[2]]]

        samples_t = [[tc_comb[0],], [sc_comb[0],], [c_comb[0],]]
        samples_pos = [tc_comb[1], sc_comb[1], c_comb[1]]
        samples_angle = [tc_comb[2],  sc_comb[2],  c_comb[2]]

        samples_derivs = [[tc_comb[3]['y'], sc_comb[3]['y'], c_comb[3]['y']],
                          [tc_comb[3]['phi'], sc_comb[3]['phi'], c_comb[3]['phi']]]

        samples_u = [tc_comb[4], tc_comb[4], tc_comb[4]]

        return epde_multisample_discovery(samples_t, samples_pos, samples_angle, samples_derivs, samples_u, 'FD')

if __name__ == '__main__':
    only_prepare = False
    if only_prepare:
        t, traj_data = prepare_trajectories() # traj_data = (traj_obj, traj_acts, traj_rews)
        print(f'Acts len is {len(traj_data[0])}, while shape {traj_data[0][0].shape}')
        assert(t.size == traj_data[0][0].shape[0]), 'Incorrect t detected'
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/t.npy', arr = t)
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/state.npy', arr = traj_data[0][0])
        np.save(file = '/home/mikemaslyaev/Documents/EPDE/projects/control/reserve/acts.npy', arr = traj_data[1][0])
    else:
        res = general(discover=True)