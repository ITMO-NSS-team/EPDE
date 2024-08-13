import os
import sys
from collections import OrderedDict
from typing import List, Union
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
from matplotlib.collections import LineCollection

import epde
import epde.interface.control_utils as control_utils
import epde.globals as global_var
from projects.control.swingup_aux import DMCEnvWrapper, RandomPolicy, CosinePolicy, CosineSignPolicy, \
                                         TwoCosinePolicy, rollout_env, VarTrigTokens # ,, DerivSignFunction
from epde.interface.prepared_tokens import DerivSignFunction

import numpy as np
import scipy
import matplotlib.pyplot as plt

import time


import gym
import scipy.special

def get_additional_token_families(ctrl):
    angle_trig_tokens = VarTrigTokens('phi', max_power=2, freq_center=1.)
    # sgn_tokens = DerivSignFunction(token_type = 'speed_sign', var_name = 'y', token_labels=['sign(dy/dx1)',],
    #                                deriv_solver_orders = [[0,],])
    
    ctrl_keys = ['ctrl_main', 'ctrl_thrust']

    def main_thrust_filtering(thrust: Union[np.ndarray, torch.tensor]):
        # acts[0, :] = np.where(acts[0, :] > 0.5, acts[0, :], 0.)    
        # acts[1, :] = np.where(np.abs(acts[1, :]) < 0.5, 0., acts[1, :])
        if isinstance(thrust, np.ndarray):
            return np.where(thrust > 0.5, thrust, 0.)
        else:
            return torch.where(thrust > 0.5, thrust, 0.)
    
    def manuever_thrust_filtering(thrust: Union[np.ndarray, torch.tensor]):
        if isinstance(thrust, np.ndarray):
            return np.where(np.abs(thrust) < 0.5, 0., thrust)
        else:
            return torch.where(torch.abs(thrust) < 0.5, 0., thrust)

    def main_thrust_nn_eval_torch(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1)  # Validate correctness
        else:
            inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1)
        return main_thrust_filtering(global_var.control_nn.net(inp)[..., 0])

    def manuever_thrust_nn_eval_torch(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1)  # Validate correctness
        else:
            inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1)
        return manuever_thrust_filtering(global_var.control_nn.net(inp)[..., 1])

    # def nn_eval_torch(*args, **kwargs):
    #     if isinstance(args[0], torch.Tensor):
    #         inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1) # Validate correctness
    #     else:
    #         inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1)
    #     # print(f'passing tensor, stored at {inp.get_device()}')
    #     # print(f'inp shape is {inp.shape}, args are {args}, kwargs are {kwargs}')
    #     if kwargs['axis'] == 0:
    #         return main_thrust_filtering(global_var.control_nn.net(inp)[..., kwargs['axis']])
    #     elif kwargs['axis'] == 1:
    #         return manuever_thrust_filtering(global_var.control_nn.net(inp)[..., kwargs['axis']])
    #     else:
    #         raise IndexError(f'Incorrect index selected: got kwargs["axis"] {kwargs["axis"]}.')
    #     # return global_var.control_nn.net(inp)[..., kwargs['axis']]#**kwargs['power']

    def main_thrust_nn_eval_np(*args, **kwargs):
        # if kwargs['axis'] == 0:        
        return main_thrust_nn_eval_torch(*args, **kwargs).detach().cpu().numpy()  #**kwargs['power']
    
    def manuever_thrust_nn_eval_np(*args, **kwargs):
        # if kwargs['axis'] == 0:
        return manuever_thrust_nn_eval_torch(*args, **kwargs).detach().cpu().numpy()  #**kwargs['power']
    
    nn_eval_torch = {ctrl_keys[0] : main_thrust_nn_eval_torch, 
                     ctrl_keys[1] : manuever_thrust_nn_eval_torch}
    
    nn_eval_np = {ctrl_keys[0] : main_thrust_nn_eval_np, 
                  ctrl_keys[1] : manuever_thrust_nn_eval_np}

    control_var_tokens = epde.interface.prepared_tokens.ControlVarTokens(sample = [ctrl[0, ...], ctrl[1, ...]], 
                                                                         var_name = ctrl_keys,
                                                                         arg_var = [(0, [None,]), 
                                                                                    (1, [None,]),
                                                                                    (2, [None,]), 
                                                                                    (0, [0,]), 
                                                                                    (1, [0,]),
                                                                                    (2, [0,])], 
                                                                         eval_torch = nn_eval_torch, eval_np = nn_eval_np)
    
    return [angle_trig_tokens, control_var_tokens] # sgn_tokens, 

def epde_discovery(t, y, z, angle, u, derivs = None, diff_method = 'FD', data_nn: torch.nn.Sequential = None, 
                   device: str = 'cpu', use_solver = True):
    dimensionality = t.ndim - 1
    
    epde_search_obj = epde.EpdeSearch(use_solver = use_solver, dimensionality = dimensionality, boundary = 30,
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
    popsize = 10
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs = 30)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.4, 0.4, 0.2]}

    # custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)
    # if use_solver:
    #     epde_search_obj.create_pool(data=[x, angle], variable_names=['y', 'phi'], max_deriv_order=(2,),
    #                                 data_fun_pow = 2, derivs = [derivs['y'], derivs['phi']],
    #                                 additional_tokens=get_additional_token_families(ctrl=u), data_nn=data_nn)
    
    # if data_nn is None and use_solver:
    #     data_nn = global_var.solution_guess_nn
    #     if device == 'cpu':
    #         # fname = 'C://Users//Mike//Documents//Work//EPDE//projects//control//swingup_ann_cpu.pickle'
    #         fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cpu.pickle"
    #     else:
    #         fname = r"/home/mikemaslyaev/Documents/EPDE/projects/control/swingup_ann_cuda.pickle"
    #     with open(fname, 'wb') as output_file:
    #         pickle.dump(data_nn, output_file)

    if derivs is not None:
        derivs = [derivs['y'], derivs['phi']]
    epde_search_obj.fit(data=[y, z, angle], variable_names=['y', 'z', 'phi'], max_deriv_order=(2,),
                        equation_terms_max_number=6, data_fun_pow = 2, derivs = derivs,
                        additional_tokens=get_additional_token_families(ctrl=u),
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-7, 1e-4), data_nn=data_nn, device=device) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj


def prepare_data(*args, **kwargs):
    env = gym.make("LunarLander-v2",
                continuous = True,
                gravity = -9.8,
                enable_wind = False,
                wind_power = 1.0,
                turbulence_power = 1.5,
                render_mode="rgb_array")

    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    import time 

    # Number of steps you run the agent for 
    num_steps = 200

    k = 3
    main_thrust = lambda x: (np.power(x, k/2.-1)*np.exp(-x/2.))/(2**(k/2.)*scipy.special.gamma(k/2.))/3.
    test_range = np.linspace(0, 5, 100)
    plt.plot(test_range, main_thrust(test_range))
    plt.show()

    obs = env.reset()
    observations = [obs[0],]
    print(f'Initial state is {observations}')
    acts = [[0, 0],]
    left_landed = False; right_landed = False
    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        if not (left_landed or right_landed):
            print(observations[-1][3])
            hor_thrust = -np.sign(observations[-1][0]) * (np.abs(observations[-1][0]))
            action = [main_thrust(observations[-1][1]), hor_thrust]#env.action_space.sample()
        else:
            action = [0., 0.]

        acts.append(action)

        obs, reward, left_landed, right_landed, info = env.step(action)
        observations.append(obs)

        if left_landed or right_landed:
            print(f'Ouch: l {left_landed} r {right_landed} on {step}')
        if left_landed or right_landed:
            print('Touched!')
            env.reset()
            break

    # Close the env
    env.close()
    # print(len(observations))
    observations = np.stack(observations, axis = 1)
    acts = np.array(acts).T
    acts[0, :] = np.where(acts[0, :] > 0.5, acts[0, :], 0.)    
    acts[1, :] = np.where(np.abs(acts[1, :]) < 0.5, 0., acts[1, :])    
    # print('Obs:', observations.shape, ' Acts:', acts.shape)
    # print(acts[0, :])
    # print(acts[1, :])

    

    # print([len(obs) for obs in observations])
    t    = np.arange(observations[0, :].size)
    y    = observations[0, :].reshape(-1)
    z    = observations[1, :].reshape(-1)
    cols = np.linspace(0,1, y.size)


    points = np.array([y, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots() # 1, 1, sharex=True, sharey=True

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(cols.min(), cols.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(cols)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)

    eps = 0.1
    axs.set_xlim(-max(np.abs(y.min()), np.abs(y.max()))- eps, max(np.abs(y.min()), np.abs(y.max()))+eps)
    axs.set_ylim(z.min()-eps, z.max()+eps)

    every = int(observations[0, :].size / 10.)
    origins = np.stack([observations[0, :].reshape(-1)[::every], observations[1, :].reshape(-1)[::every]], axis = 0)

    plt.quiver(*origins, np.sin(observations[4, :][::every]), np.cos(observations[4, :][::every]), color='r', scale=10)
    plt.quiver(*origins, observations[2, :][::every], observations[3, :][::every], color='b', scale=5)

    plt.show()

    plt.plot(np.arange(observations[2, :].size), observations[2, :], color = 'k')
    plt.plot(np.arange(observations[3, :].size), observations[3, :], color = 'r')
    plt.show()

    plt.plot(np.arange(acts[0, :].size), acts[0, :], color = 'k')
    plt.plot(np.arange(acts[1, :].size), acts[1, :], color = 'r')
    plt.show()
    print(observations.shape, acts.shape)
    return t, observations, acts, env.moon, (env.helipad_x1, env.helipad_x2, env.helipad_y)

# def discover_equations(observations: np.ndarray, actions: np.ndarray):
    

if __name__ == '__main__':
    t, obs, acts, moon, helipad = prepare_data()

    # derivs = np.stack([obs[1, ...], obs[3, ...], obs[5, ...]])

    epde_discovery(t = t, y = obs[0, ...], z = obs[2, ...], angle = obs[4, ...], u = acts, device = 'cuda')