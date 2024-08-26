import os
import sys
from collections import OrderedDict
from typing import List, Union, Tuple
import faulthandler

faulthandler.enable()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
import torch
import dill as pickle

from tqdm import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import epde
from epde.control import ControlExp, ConstrLocation, ConditionalLoss, ControlConstrNEq, ControlConstrEq
import epde.globals as global_var
from projects.control.swingup_aux import VarTrigTokens

from epde.interface.interface import ExperimentCombiner
from epde.interface.prepared_tokens import DerivSignFunction

import numpy as np
import scipy
import matplotlib.pyplot as plt

import time


import gym
import scipy.special

def get_additional_token_families(ctrl, device = 'cpu'):
    angle_trig_tokens = VarTrigTokens('phi', max_power=1, freq_center=1.)

    ctrl_keys = ['ctrl_main', 'ctrl_thrust']

    def main_thrust_filtering(thrust: Union[np.ndarray, torch.tensor]):
        # print('In main thrust', thrust.shape)        
        if isinstance(thrust, np.ndarray):
            return np.where(thrust > 0.5, thrust, 0.)
        else:
            return torch.where(thrust > 0.5, thrust, 0.).to(device)
    
    def manuever_thrust_filtering(thrust: Union[np.ndarray, torch.tensor]):
        # print('In manuever thrust', thrust.shape)
        if isinstance(thrust, np.ndarray):
            return np.where(np.abs(thrust) < 0.5, 0., thrust)
        else:
            return torch.where(torch.abs(thrust) < 0.5, 0., thrust).to(device)

    def main_thrust_nn_eval_torch(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1).to(device)  # Validate correctness
        else:
            inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1).to(device)
        # print('In main thrust: shape of glob_net_output', inp.shape, global_var.control_nn.net(inp).shape)
        return main_thrust_filtering(global_var.control_nn.net(inp)[..., 0])

    def manuever_thrust_nn_eval_torch(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            inp = torch.cat([torch.reshape(tensor, (-1, 1)) for tensor in args], dim = 1).to(device)  # Validate correctness
        else:
            inp = torch.cat([torch.reshape(torch.Tensor([elem,]), (-1, 1)) for elem in args], dim = 1).to(device)
        # print('In manuever thrust: shape of glob_net_output', inp.shape, global_var.control_nn.net(inp).shape)            
        return manuever_thrust_filtering(global_var.control_nn.net(inp)[..., 1])

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
                                                                         eval_torch = nn_eval_torch, 
                                                                         eval_np = nn_eval_np,
                                                                         device = device)
    
    return [angle_trig_tokens, control_var_tokens]

def epde_discovery(t, y, z, angle, u, derivs = None, diff_method = 'FD', data_nn: torch.nn.Sequential = None, 
                   device: str = 'cpu', use_solver = True):
    dimensionality = t.ndim - 1
    
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
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs = 1)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.4, 0.5, 0.1]}



    if derivs is not None:
        derivs = [derivs['y'], derivs['phi']]
    epde_search_obj.fit(data=[y, z, angle], variable_names=['y', 'z', 'phi'], max_deriv_order=(2,),
                        equation_terms_max_number=9, data_fun_pow = 2, derivs = derivs,
                        additional_tokens=get_additional_token_families(ctrl=u, device = device),
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-4, 1e-0), data_nn=data_nn) # TODO: narrow sparsity interval, reduce the population size
    epde_search_obj.equations()
    return epde_search_obj

def prepare_data(num_steps: int = 200, *args, **kwargs):
    env = gym.make("LunarLander-v2",
                continuous = True,
                gravity = -9.8,
                enable_wind = False,
                wind_power = 1.0,
                turbulence_power = 1.5,
                render_mode = "rgb_array")

    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    import time 

    # Number of steps you run the agent for 
    num_steps = 200

    k = 3
    def main_thrust(x):
        thrust = 0.4 + (np.power(x, k/2.-1)*np.exp(-x/2.))/(2**(k/2.)*scipy.special.gamma(k/2.))
        print(x, thrust)
        return thrust
    test_range = np.linspace(0, 5, 100)

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
            hor_thrust = 0.5 + observations[-1][4] * 2
            action = [main_thrust(observations[-1][1]), hor_thrust]
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

    env.close()

    observations = np.stack(observations, axis = 1)
    acts = np.array(acts).T
    acts_res = np.copy(acts)
    acts[0, :] = np.where(acts[0, :] > 0.5, acts[0, :], 0.)    
    acts[1, :] = np.where(np.abs(acts[1, :]) < 0.5, 0., acts[1, :])    

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
    plt.plot(np.arange(acts_res[0, :].size), acts_res[0, :], '--',  color = 'k')

    plt.plot(np.arange(acts[1, :].size), acts[1, :], color = 'r')
    plt.plot(np.arange(acts_res[1, :].size), acts_res[1, :], '--',  color = 'r')
    plt.show()
    print(observations.shape, acts.shape)
    return t, observations, acts, env.moon, (env.helipad_x1, env.helipad_x2, env.helipad_y)

# def discover_equations(observations: np.ndarray, actions: np.ndarray):
    
def optimize_ctrl(eq: epde.structure.main_structures.SoEq, t: torch.tensor,
                  y_init: float, z_init: float, dy_init: float, dz_init: float, phi_init: float, dphi_init: float,
                  y_left: float, y_right: float, dy_max: float, dz_max: float, stab_der_ord: int, helipad: Tuple[int], # tuple as (y_left, y_right, z) 
                  state_nn_pretrained: torch.nn.Sequential,
                  ctrl_nn_pretrained: torch.nn.Sequential, fig_folder: str, device = 'cpu'):
    from epde.supplementary import AutogradDeriv
    autograd = AutogradDeriv()

    loc_domain = ConstrLocation(domain_shape = (t.size()[0],), device=device) # Declaring const in the entire domain
    loc_end = ConstrLocation(domain_shape = (t.size()[0],), axis = 0, loc = -1, device=device) # Check format
    print(f'loc_end.flat_idxs : {loc_end.flat_idxs}, device {device}')

    def cosine_cond(x, ref):
        return torch.abs(torch.cos(x) - ref)
    
    halflength = (helipad[1] - helipad[0])/2.
    ycenter = (helipad[1] + helipad[0])/2.

    def allowed_cosine_cond(x, ref):
        return torch.abs(torch.cos(x) - ref)

    def landing_x_cond(x, ref):
        return torch.abs(x - ref) - halflength
        
    y_constr = ControlConstrNEq(val = torch.full_like(input = t[-1], fill_value = ycenter, device=device), # Better processing for periodic
                                indices = loc_end, deriv_axes=[None,], deriv_method = autograd, nn_output=0,
                                estim_func=landing_x_cond, device=device, sign='<')
    
    dy_constr = ControlConstrNEq(val = torch.full_like(input = t[-1], fill_value = dy_max, device=device), # Better processing for periodic
                                 indices = loc_end, deriv_axes=[0,], deriv_method = autograd, nn_output=0,
                                 device=device, sign='<')

    z_constr = ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = helipad[2], device=device), # Better processing for periodic
                               indices = loc_end, deriv_axes=[None,], deriv_method = autograd, nn_output=1,
                               device=device)
    
    dz_constr = ControlConstrNEq(val = torch.full_like(input = t[-1], fill_value = dz_max, device=device), # Better processing for periodic
                                 indices = loc_end, deriv_axes=[0,], deriv_method = autograd, nn_output=1,
                                 device=device, sign='<')

    phi_tar_constr = ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 1., device=device), # Better processing for periodic
                                     indices = loc_end, deriv_axes=[None,], deriv_method = autograd, nn_output=2, 
                                     estim_func=cosine_cond, device=device)
    dphi_tar_constr = ControlConstrEq(val = torch.full_like(input = t[-1], fill_value = 0, device=device),
                                      indices = loc_end, deriv_axes=[0,], deriv_method = autograd, nn_output=2, device=device)
    contr_constr = ControlConstrEq(val = torch.full_like(input = t, fill_value = 0., device=device),
                                   indices = loc_domain, deriv_axes=[None,], deriv_method = autograd, nn_output=0, device=device)

    y_right_bnd = ControlConstrNEq(val = torch.full_like(input = t, fill_value = y_right, device=device), sign='<',
                                   indices = loc_domain, deriv_method = autograd, nn_output=0, device=device)
    y_left_bnd = ControlConstrNEq(val = torch.full_like(input = t, fill_value = y_left, device=device), sign='>',
                                  indices = loc_domain, deriv_method = autograd, nn_output=0, device=device)
    
    loss = ConditionalLoss([(1., y_constr, 0),
                            (1., dy_constr, 0),
                            (1., z_constr, 0),
                            (1., dz_constr, 0),
                            (1., phi_tar_constr, 0),
                            (1., dphi_tar_constr, 0), 
                            (0.001, contr_constr, 1),
                            (100., y_right_bnd, 0),
                            (100., y_left_bnd, 0)])
    
    optimizer = ControlExp(loss=loss, device=device)
    
    def get_ode_bop(key, var, term, grid_loc, value):
        bop = epde.interface.solver_integration.BOPElement(axis = 0, key = key, term = term,
                                                           power = 1, var = var)
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

    bop_z = get_ode_bop('z', 0, [None], t[0, 0], z_init)
    bop_dz = get_ode_bop('z', 0, [0,], t[0, 0], dz_init)

    bop_phi = get_ode_bop('phi', 0, [None], t[0, 0], phi_init)
    bop_dphi = get_ode_bop('phi', 0, [0,], t[0, 0], dphi_init)

    optimizer.system = eq.system

    # optimizer.set_control_optim_params()

    solver_params = {'full':     {'training_params': {'epochs': 1500,}, 'optimizer_params': {'params': {'lr': 1e-5}}}, 
                     'abridged': {'training_params': {'epochs': 300,}, 'optimizer_params': {'params': {'lr': 5e-5}}}}
    
    state_nn, ctrl_net, ctrl_pred, hist = optimizer.train_pinn(bc_operators = [(bop_y(device=device), 0.1),
                                                                               (bop_dy(device=device), 0.1),
                                                                               (bop_z(device=device), 0.1),
                                                                               (bop_dz(device=device), 0.1),
                                                                               (bop_phi(device=device), 0.1),
                                                                               (bop_dphi(device=device), 0.1)],
                                                               grids = [t,], n_control = 2., 
                                                               state_net = state_nn_pretrained, 
                                                               opt_params = [0.005, 0.9, 0.999, 1e-8],
                                                               control_net = ctrl_nn_pretrained, epochs = 55,
                                                               fig_folder = fig_folder, eps=2e0,
                                                               solver_params = solver_params)

    return state_nn, ctrl_net, ctrl_pred, hist


if __name__ == '__main__':
    # import pickle
    import datetime

    experiment = 'lander'
    explicit_cpu = False
    device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'
    print(f'Working on {device}')

    res_folder = '/home/mikemaslyaev/Documents/EPDE/projects/control'
    fig_folder = os.path.join(res_folder, 'figs')

    traj_filename = os.path.join(res_folder, f'training_traj_{experiment}.pickle')

    try:
        with open(traj_filename, 'rb') as data_input_file:  
            traj_info = pickle.load(data_input_file)
            t, obs, acts, helipad = traj_info # , moon
    except FileNotFoundError:
        t, obs, acts, _, helipad = prepare_data()
        traj_info = (t, obs, acts, helipad) # , moon
        with open(traj_filename, 'wb') as output_file:
            pickle.dump(traj_info, output_file)


    print('acts.shape:', acts.shape)

    print(f'Observations {obs.shape}, acts {acts.shape}')

    plt.plot(t, obs[0, ...], color = 'k', label = 'y')
    plt.plot(t, obs[2, ...], '--', color = 'k', label = 'dy')    
    plt.plot(t, obs[1, ...], color = 'r', label = 'z')
    plt.plot(t, obs[3, ...], '--', color = 'r', label = 'dz')    
    plt.plot(t, obs[4, ...], color = 'b', label = 'phi')
    plt.plot(t, obs[5, ...], '--', color = 'b', label = 'd phi')
    plt.legend()
    plt.show()

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

    use_solver = True
    models = epde_discovery(t = t[:-5], y = obs[0, :-5], z = obs[1, :-5], angle = obs[4, :-5], diff_method='poly',
                            u = acts[..., :-5], device = 'cuda', data_nn=data_nn, use_solver = use_solver)
    
    # exp_comb = 
    model = ExperimentCombiner(models.equations(False)[0]).create_best(models.pool)

    import epde.loader as Loader

    path_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'EPDE', 'projects', 'control')

    loader = Loader.EPDELoader()
    loader.save(obj=model, filename=os.path.join(path_dir, f'{experiment}_solver_model.pickle')) # Pickle-saving an equation
    loader.save(obj = models.pool, filename=os.path.join(path_dir, f'{experiment}_solver_pool.pickle')) # Pickle-saving the pool
    # raise NotImplementedError()

    if save_nn:
        if device == 'cpu':
            fname =  os.path.join(res_folder, f"data_ann_{experiment}_cpu.pickle")
        else:
            fname = os.path.join(res_folder, f"data_ann_{experiment}_cuda.pickle")
        with open(fname, 'wb') as output_file:
            pickle.dump(epde.globals.solution_guess_nn, output_file)


    # model.equations()
    
    example_sol = epde.globals.solution_guess_nn(torch.from_numpy(t).reshape((-1, 1)).float().to(device))
    epde.globals.solution_guess_nn.to(device)
    print(f'example_sol: {type(example_sol)}, {example_sol.shape}, {example_sol.get_device()}')

    def create_shallow_nn(arg_num: int = 1, output_num: int = 1, device = 'cpu') -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
        hidden_neurons = 25
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


    nn = 'shallow'
    load_ctrl = True

    if device == 'cpu':
        ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cpu.pickle")
    else:
        ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cuda.pickle")
    if load_ctrl:
        with open(ctrl_fname, 'rb') as ctrl_input_file:  
            ctrl_ann = pickle.load(ctrl_input_file)
    else:
        nn_method = create_shallow_nn if nn == 'shallow' else create_deep_nn
        ctrl_args = [obs[0, :-5], obs[2, :-5], obs[1, :-5], obs[3, :-5], obs[4, :-5], obs[5, :-5]]
        ctrl_ann = epde.supplementary.train_ann(args=ctrl_args,
                                                data = acts.T, 
                                                epochs_max = 5e6, 
                                                dim = len(ctrl_args), 
                                                model = nn_method(len(ctrl_args), 1, device=device),
                                                device = device)

    with open(ctrl_fname, 'wb') as ctrl_output_file:  
            pickle.dump(ctrl_ann, file = ctrl_output_file)
            
                #   (eq: epde.structure.main_structures.SoEq, t: torch.tensor,
                #   y_init: float, z_init: float, dy_init: float, dz_init: float, phi_init: float, dphi_init: float,
                #   y_left: float, y_right: float, dy_max: float, dz_max: float, stab_der_ord: int, helipad: Tuple[int], # tuple as (y_left, y_right, z) 
                #   state_nn_pretrained: torch.nn.Sequential,
                #   ctrl_nn_pretrained: torch.nn.Sequential, fig_folder: str, device = 'cpu')

    res = optimize_ctrl(model, torch.from_numpy(t).reshape((-1, 1)).float().to(device),
                        y_init=obs[0, 0], dy_init=obs[1, 0], z_init=obs[2, 0], dz_init=obs[3, 0], 
                        phi_init=obs[4, 0], dphi_init=obs[5, 0],
                        y_left = -1, y_right = 1, dy_max = 0.1, dz_max = 0.1, stab_der_ord = 2, helipad = helipad,
                        state_nn_pretrained=epde.globals.solution_guess_nn, ctrl_nn_pretrained=ctrl_ann, 
                        fig_folder=fig_folder, device=device)

    savename = f'res_{time_exp_start.month}_{time_exp_start.day}_at_{time_exp_start.hour}_{time_exp_start.minute}_{experiment}.pickle'

    with open(os.path.join(res_folder, savename), 'wb') as output_file:  
        pickle.dump(res, output_file)                
    


