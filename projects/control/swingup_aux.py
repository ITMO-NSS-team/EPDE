import os
import sys
import torch
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
from typing import Union, Callable, Tuple, List
from collections import OrderedDict

from epde.evaluators import CustomEvaluator, EvaluatorTemplate, sign_evaluator, \
     trigonometric_evaluator
from epde.interface.prepared_tokens import PreparedTokens
from epde.interface.token_family import TokenFamily
import epde.globals as global_var

import gymnasium
from gymnasium.spaces.box import Box
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
try:
    from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
except ImportError:
    from projects.control.ray_replacement import DMCEnv

class DerivSignFunction(PreparedTokens):
    def __init__(self, var_name: str, token_labels: list):
        """
        Class for tokens, representing arbitrary functions of the modelled variable passed in `var_name` or its derivatives.  
        """
        token_type = f'signum of {var_name}`s derivatives'
        max_order = len(token_labels)

        deriv_solver_orders: list = [[0,]*(order+1) for order in range(max_order)]
        params_ranges = OrderedDict([('power', (1, 1))])
        params_equality_ranges = {'power': 0}

        self._token_family = TokenFamily(token_type = token_type, variable = var_name,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = False)

        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges,
                                      derivs_solver_orders=deriv_solver_orders)
        self._token_family.set_evaluator(sign_evaluator)

class VarTrigTokens(PreparedTokens):
    """
    Class for prepared tokens, that belongs to the trigonometric family
    """
    def __init__(self, var_name: str, freq_center: float = 1., 
                 max_power: int = 2, freq_eps = 1e-8):
        """
        Initialization of class

        Args:

        """
        freq = (freq_center - freq_eps, freq_center + freq_eps)

        self._token_family = TokenFamily(token_type=f'trigonometric of {var_name}', variable=var_name,
                                         family_of_derivs=True)
        self._token_family.set_status(demands_equation=False, unique_specific_token=True, unique_token_type=True,
                                      meaningful=True, non_default_power = True)
            
        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''
            form = label + r'^{{{0}}}'.format(params["power"][1]) + \
                    r'(' + params["freq"][1] + r' x_{' + params["dim"][1] + r'})'
            return form
        
        self._token_family.set_latex_form_constructor(latex_form)
        trig_token_params = OrderedDict([('power', (1, max_power)),
                                         ('freq', freq)])
        
        trig_equal_params = {'power': 0, 'freq': 2*freq_eps}

        adapted_labels = [f'sin({var_name})', f'cos({var_name})']
        deriv_solver_orders = [[None,] for label in adapted_labels]

        trig_eval_fun_np = {adapted_labels[0]: lambda *args, **kwargs: np.sin(kwargs['freq'] * args[0]) ** kwargs['power'],
                            adapted_labels[1]: lambda *args, **kwargs: np.cos(kwargs['freq'] * args[0]) ** kwargs['power']}

        trig_eval_fun_torch = {adapted_labels[0]: lambda *grids, **kwargs: torch.cos(kwargs['freq'] 
                                                                                     * grids[0]) ** kwargs['power'],
                               adapted_labels[1]: lambda *grids, **kwargs: torch.sin(kwargs['freq']
                                                                                     * grids[0]) ** kwargs['power']}

        eval = CustomEvaluator(evaluation_functions_np = trig_eval_fun_np,
                               evaluation_functions_torch = trig_eval_fun_torch,
                               eval_fun_params_labels = ['power', 'freq'])

        self._token_family.set_params(adapted_labels, trig_token_params, trig_equal_params, 
                                      derivs_solver_orders=deriv_solver_orders)
        self._token_family.set_evaluator(eval)


class ControlVarTokens(PreparedTokens):
    def __init__(self, sample: np.ndarray, ann: torch.nn.Sequential = None, var_name: str = 'ctrl',
                 arg_var: List[Tuple[Union[int, List]]] = [(0, [None,]),]):
        vars, der_ords = zip(*arg_var)

        token_params = OrderedDict([('power', (1, 1)),])
        
        equal_params = {'power': 0}

        self._token_family = TokenFamily(token_type = var_name, variable = vars,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = False)
        
        self._token_family.set_params([var_name,], token_params, equal_params,
                                      derivs_solver_orders=der_ords)
        
        def nn_eval_torch(*args, **kwargs):
            inp = torch.stack([torch.reshape(tensor, -1) for tensor in args]) # Validate correctness
            return global_var.control_nn(inp)**kwargs['power']

        def nn_eval_np(*args, **kwargs):
            return nn_eval_torch(*args, **kwargs).detach().numpy()**kwargs['power']

        eval = CustomEvaluator(evaluation_functions_np=nn_eval_torch,
                               evaluation_functions_torch=nn_eval_np,
                               eval_fun_params_labels = ['power'])

        global_var.reset_control_nn(ann=ann, n_var=len(vars))
        global_var.tensor_cache.add(tensor=sample, label = (var_name, (1.0,)))

        self._token_family.set_evaluator(eval)


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

    for i in tqdm.tqdm(range(n_steps), disable=not verbose):
        
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