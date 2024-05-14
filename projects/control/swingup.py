#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:43:03 2024

@author: maslyaev
"""
import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import List

import gymnasium
from gymnasium.spaces.box import Box
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
try:
    from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
except ImportError:
    from projects.control.ray_replacement import DMCEnv

import matplotlib.pyplot as plt

import epde

def safe_reset(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if isinstance(res[-1], dict):
        return res[0]
    else:
        return res
    
def safe_step(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if len(res)==5:
        return res[0], res[1], res[2] or res[3], res[4]
    else:
        return res
    
def replace_with_inf(arr, neg):
    '''helper function to replace an array with inf. Used for Box bounds'''
    replace_with = np.inf
    if neg:
        replace_with *= -1
    return np.nan_to_num(arr, nan=replace_with)

def rollout_env(env, policy, n_steps, n_steps_reset=np.inf, seed=None, verbose = False, env_callback = None):
    '''
    Step through an environment and produce rollouts.
    Arguments:
        env: gymnasium environment
        policy: sindy_rl.BasePolicy subclass
            The policy to rollout experience
        n_steps: int
            number of steps of experience to rollout
        n_steps_reset: int
            number of steps of experience to rollout before reset
        seed: int
            environment reset seed 
        verbose: bool
            whether to provide tqdm progress bar
        env_callback: fn(idx, env)
            optional function that is called after every step
    Returns:
        lists of obs, acts, rews trajectories
    '''
    if seed is not None:    
        obs_list = [safe_reset(env.reset(seed=seed))]
    else:
        obs_list = [safe_reset(env.reset())]

    act_list = []
    rew_list = []
    
    trajs_obs = []
    trajs_acts = []
    trajs_rews = []

    for i in tqdm(range(n_steps), disable=not verbose):
        
        # collect experience
        action = policy.compute_action(obs_list[-1])
        step_val = env.step(action)
        # print(f'step_val {step_val} for action {action}')
        obs, rew, done, info = safe_step(step_val)
        act_list.append(action)
        obs_list.append(obs)
        rew_list.append(rew)
        
        # handle resets
        if done or len(obs_list) > n_steps_reset:
            obs_list.pop(-1)
            trajs_obs.append(np.array(obs_list))
            trajs_acts.append(np.array(act_list))
            trajs_rews.append(np.array(rew_list))

            obs_list = [safe_reset(env.reset())]
            act_list = []
            rew_list = []
        
        # env callback
        if env_callback:
            env_callback(i, env)
    
    # edge case if never hit reset
    if len(act_list) != 0:
        obs_list.pop(-1)
        trajs_obs.append(np.array(obs_list))
        trajs_acts.append(np.array(act_list))
        trajs_rews.append(np.array(rew_list))
    return trajs_obs, trajs_acts, trajs_rews

class DMCEnvWrapper(DMCEnv):
    '''
    A wrapper for all dm-control environments using RLlib's 
    DMCEnv wrapper. 
    '''
    # need to wrap with config dict instead of just passing kwargs
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)
        
class BasePolicy:
    '''Parent class for policies'''
    def __init__(self):
        raise NotImplementedError
    def compute_action(self, obs):
        '''given observation, output action'''
        raise NotImplementedError        
        
class RandomPolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, action_space = None, low = -1, high = 1, seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        if action_space: 
            self.action_space = action_space
        else:
            self.action_space = Box(low=low, high=high) #action_space
        self.action_space.seed(seed)
        self.magnitude = 0.08
#        self.inertia = 0.9
#        self._prev_action = 0.
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = self.action_space.sample()
        # print(f'action is {action}, type {type(action)}')
        return self.magnitude * action # + self.inertia * self._prev_action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag        

class CosinePolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, period = 300., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-25, high=25)
        self.time_counter = 0
        self.period = period
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        # print(1.*self.time_counter/self.period)
        action = np.array([self.amplitude * self.time_counter * np.sin(2*np.pi*self.time_counter/self.period),], dtype=np.float32)
        self.time_counter += 1
        # print(f'action is {action}')
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag


class CosineSignPolicy(BasePolicy):
    '''
    A policy, that employs cosine signum as control function
    '''
    def __init__(self, period = 300., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-2500, high=2500)
        self.time_counter = 0
        self.period = period
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = np.array([self.amplitude * self.time_counter * np.sign(np.sin(2*np.pi*self.time_counter/self.period)),], dtype=np.float32)
        self.time_counter += 1
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag        

class TwoCosinePolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, period1 = 300., period2 = 150., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-25, high=25)
        self.time_counter = 0
        self.period = [period1, period2]
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        # print(1.*self.time_counter/self.period)
        action = np.array([self.amplitude * self.time_counter * (np.sin(2*np.pi*self.time_counter/self.period[0]) + np.sin(2*np.pi*self.time_counter/self.period[1])),], 
                          dtype=np.float32)
        self.time_counter += 1
        # print(f'action is {action}')
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag

def epde_discovery(t, x, angle, u, derivs, diff_method = 'FD'):
    dimensionality = x.ndim - 1
    
    epde_search_obj = epde.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 30,
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
    
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    
    angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
                                               {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
                                               OrderedDict([('power', (1, 3))]), {'power': 0}, meaningful=True)
    control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u}, OrderedDict([('power', (1, 1))]),
                                                {'power': 0}, meaningful=True)
    sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
                                        {'sgn(dy)' : np.sign(derivs['y'][:, 0]),
                                         'sgn(ddy)' : np.sign(derivs['y'][:, 1]),}, 
                                        OrderedDict([('power', (1, 1))]), {'power': 0}, meaningful=True)    

    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=200)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)
    
    epde_search_obj.fit(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,),
                        equation_terms_max_number=10, data_fun_pow = 2, derivs = [derivs['y'], derivs['phi']],
                        additional_tokens=[custom_grid_tokens, control_var_tokens, angle_trig_tokens, sgn_tokens], # , control_var_tokens, 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-7, 1e-5)) # TODO: narrow sparsity interval, reduce the population size
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
    
    angle_cos = np.cos(np.concatenate([angle_arr for angle_arr in angle]))
    angle_sin = np.sin(np.concatenate([angle_arr for angle_arr in angle]))
    
    angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
                                               {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
                                               OrderedDict([('power', (1, 3))]), {'power': 0}, meaningful=True, 
                                               unique_token_type=False, unique_specific_token=False, non_default_power=True)
    u_concat = np.concatenate(u)
    print(f'u_concat.shape is {u_concat.shape}')
    control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u_concat}, OrderedDict([('power', (1, 1))]),
                                                {'power': 0}, meaningful=True)
    der_y = np.concatenate(derivs[0])
    print(f'der_y shape is {der_y.shape}')
    sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
                                        {'sgn(dy)' : np.sign(der_y[:, 0]),
                                         'sgn(ddy)' : np.sign(der_y[:, 1]),}, 
                                        OrderedDict([('power', (1, 1))]), {'power': 0}, meaningful=True)    

    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=150)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)

    epde_search_obj.fit(samples = samples, variable_names = ['y', 'phi'], max_deriv_order = (2,),
                        equation_terms_max_number = 15, data_fun_pow = 2, deriv_fun_pow=2, derivs = derivs,
                        additional_tokens = [custom_grid_tokens, control_var_tokens, angle_trig_tokens, sgn_tokens], # , control_var_tokens, 
                        equation_factors_max_number = factors_max_number,
                        eq_sparsity_interval = (1e-7, 1e-5)) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj

def translate_equation(t, x, angle, u, derivs: dict, diff_method = 'FD'):
    dimensionality = x.ndim - 1
    
    lp_terms = [['ctrl{power: 1}',], 
                ['d^2phi/dx0^2{power: 1}', 'cos(phi){power: 1}'],
                ['dphi/dx0{power: 2}', 'sin(phi){power: 1}'],
                ['sgn(dy){power: 1}']]
    rp_term  = ['d^2y/dx0^2{power: 1}',]
    

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
    
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    
    angle_trig_tokens = epde.CacheStoredTokens('angle_trig', ['sin(phi)', 'cos(phi)'], 
                                               {'sin(phi)' : angle_sin, 'cos(phi)' : angle_cos}, 
                                               OrderedDict([('power', (1, 3))]), {'power': 0})
    control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : u}, OrderedDict([('power', (1, 1))]),
                                                {'power': 0})
    sgn_tokens = epde.CacheStoredTokens('signum of y', ['sgn(dy)', 'sgn(ddy)'], 
                                        {'sgn(dy)' : np.sign(derivs['y'][:, 0]),
                                         'sgn(ddy)' : np.sign(derivs['y'][:, 1]),}, 
                                        OrderedDict([('power', (1, 1))]), {'power': 0})
    

    epde_search_obj.create_pool(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,), derivs = [derivs['y'], derivs['phi']],
                                additional_tokens = [angle_trig_tokens, control_var_tokens, sgn_tokens])

    test = epde.interface.equation_translator.CoeffLessEquation(lp_terms, rp_term, 
                                                                 epde_search_obj.pool)
    val, target, features = test.equation.evaluate(normalize = False, return_val = True)
    print(np.mean(val), np.mean(np.abs(val)), features.shape)
    plt.plot(target, color = 'r', label = 'Equation target')
    plt.plot(val, '-', color = 'k', label = 'Equation discrepancy')
    plt.plot((test.equation.weights_final.reshape((1, -1))[:, :-1] @ features.T + test.equation.weights_final[-1]).reshape(-1), color = 'b', label = 'Equation features')
    plt.grid()
    plt.legend()
    plt.show()

    return test

if __name__ == '__main__':
    env_config = {'domain_name': "cartpole",
                  'task_name': "swingup",
                  'frame_skip': 1,
                  'from_pixels': False}
    cart_env = DMCEnvWrapper(env_config)
    random_policy = RandomPolicy(cart_env.action_space)
    cosine_policy = CosinePolicy(period=180, amplitude=0.004)
    cosine_signum_policy = CosineSignPolicy(period=180, amplitude=0.002)
    two_cosine_policy = TwoCosinePolicy(180, 90, 0.002)

    single_sample = False
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

            discover = False
            if discover:
                res = epde_discovery(t, x, angle, u, derivs, 'FD')
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

        res = epde_multisample_discovery(samples_t, samples_pos, samples_angle, samples_derivs, samples_u, 'FD')
