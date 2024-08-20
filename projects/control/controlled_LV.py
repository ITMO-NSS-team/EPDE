import os
import sys
import datetime
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
import torch

from typing import Callable, Union
from collections import OrderedDict

import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

import epde
import epde.globals as global_var
import epde.interface.control_utils as control_utils


def controlled_lv_by_RK_t(initial : tuple, timestep : float, steps : int, alpha : float, 
                          beta : float, delta : float, gamma : float, c1: float = 0, c2: float = 0,
                          conrol_intensity: Union[np.ndarray, Callable] = lambda x: 1):
    res = np.full(shape = (steps, 2), fill_value = initial, dtype=np.float64)
    if not isinstance(conrol_intensity, np.ndarray):
        conrol_intensity_vect = np.vectorize(conrol_intensity)
        conrol_intensity = conrol_intensity_vect(np.linspace(0, steps*timestep, steps*2 - 1))
    assert conrol_intensity.size == steps*2-1, 'Incorrect shape of control, not taking Runge-Kutta half-steps into account'

    for step in range(steps-1):
        alpha_1 = alpha - c1*conrol_intensity[2*step]
        alpha_2 = alpha - c1*conrol_intensity[2*step + 1]
        alpha_3 = alpha - c1*conrol_intensity[2*step + 2]

        gamma_1 = gamma + c2*conrol_intensity[2*step]
        gamma_2 = gamma + c2*conrol_intensity[2*step + 1]
        gamma_3 = gamma + c2*conrol_intensity[2*step + 2]


        k1 = alpha_1 * res[step, 0] - beta * res[step, 0] * res[step, 1]; x1 = res[step, 0] + timestep/2. * k1
        l1 = delta * res[step, 0] * res[step, 1] - gamma_1 * res[step, 1]; y1 = res[step, 1] + timestep/2. * l1

        k2 = alpha_2 * x1 - beta * x1 * y1; x2 = res[step, 0] + timestep/2. * k2
        l2 = delta * x1 * y1 - gamma_2 * y1; y2 = res[step, 1] + timestep/2. * l2

        k3 = alpha_2 * x2 - beta * x2 * y2
        l3 = delta * x2 * y2 - gamma_2 * y1
        
        x3 = res[step, 0] + timestep * k1 - 2 * timestep * k2 + 2 * timestep * k3
        y3 = res[step, 1] + timestep * l1 - 2 * timestep * l2 + 2 * timestep * l3
        k4 = alpha_3 * x3 - beta * x3 * y3
        l4 = delta * x3 * y3 - gamma_3 * y3
        
        res[step+1, 0] = res[step, 0] + timestep / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        res[step+1, 1] = res[step, 1] + timestep / 6. * (l1 + 2 * l2 + 2 * l3 + l4)
    return res

def controlled_lv_by_RK(initial : tuple, timestep : float, steps : int, alpha : float, 
                        beta : float, delta : float, gamma : float, c1: float = 0, c2: float = 0,
                        control_intensity: Union[Callable, torch.nn.Sequential] = lambda x: 1):
    res = np.full(shape = (steps, 2), fill_value = initial, dtype=np.float64)
    controls = np.zeros(shape = steps)

    if isinstance(control_intensity, torch.nn.Sequential):
        prepare_input = lambda *args: torch.from_numpy(np.array(args))
    else:
        prepare_input = lambda *args: args

    for step in range(steps-1):
        ctrl_val_1 = control_intensity(prepare_input(res[step, 0], res[step, 1]))
        controls[step] = ctrl_val_1

        alpha_1 = alpha - c1 * ctrl_val_1
        gamma_1 = gamma + c2 * ctrl_val_1

        k1 = alpha_1 * res[step, 0] - beta * res[step, 0] * res[step, 1]; x1 = res[step, 0] + timestep/2. * k1
        l1 = delta * res[step, 0] * res[step, 1] - gamma_1 * res[step, 1]; y1 = res[step, 1] + timestep/2. * l1

        ctrl_val_2 = control_intensity(prepare_input(x1, y1))
        alpha_2 = alpha - c1 * ctrl_val_2
        gamma_2 = gamma + c2 * ctrl_val_2

        k2 = alpha_2 * x1 - beta * x1 * y1; x2 = res[step, 0] + timestep/2. * k2
        l2 = delta * x1 * y1 - gamma_2 * y1; y2 = res[step, 1] + timestep/2. * l2

        ctrl_val_3 = control_intensity(prepare_input(x2, y2))
        alpha_3 = alpha - c1 * ctrl_val_3
        gamma_3 = gamma + c2 * ctrl_val_3

        k3 = alpha_3 * x2 - beta * x2 * y2
        l3 = delta * x2 * y2 - gamma_3 * y1
        
        x3 = res[step, 0] + timestep * k1 - 2 * timestep * k2 + 2 * timestep * k3
        y3 = res[step, 1] + timestep * l1 - 2 * timestep * l2 + 2 * timestep * l3

        alpha_4 = alpha - c1*control_intensity(prepare_input(x3, y3))
        gamma_4 = gamma + c2*control_intensity(prepare_input(x3, y3))

        k4 = alpha_4 * x3 - beta * x3 * y3
        l4 = delta * x3 * y3 - gamma_4 * y3
        
        res[step+1, 0] = res[step, 0] + timestep / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        res[step+1, 1] = res[step, 1] + timestep / 6. * (l1 + 2 * l2 + 2 * l3 + l4)
    return controls, res

def prepare_data(steps_num: int = 151, t_max: float = 0.5, ctrl_fun: Callable = lambda x: x[0]*x[1]):
    # def get_sine_control(ampl: float = 1, period: float = 1., 
    #                      phase_shift: float = 0.) -> Callable:
    #     return lambda x: ampl*(np.sin(2*np.pi/period*(x + phase_shift)) + 1.)
    
    step = t_max/steps_num
    t = np.arange(start = 0, stop = step * steps_num, step = step)
    # ctrl = get_sine_control(ampl = 15., period=0.3)(np.linspace(0, 1, t.size*2 - 1))
    ctrl, solution = controlled_lv_by_RK(initial=(4., 2.), timestep=step, steps=steps_num, 
                                   alpha=20., beta=20., delta=20., gamma=20., 
                                   c1 = 1., c2 = 1., control_intensity = ctrl_fun)

    return t, ctrl, solution

def epde_discovery(t: np.ndarray, u: np.ndarray, v: np.ndarray, control: np.ndarray, diff_method = 'FD',
                   control_ann: torch.nn.Sequential = None, bnd = 30):
    dimensionality = t.ndim - 1
    
    epde_search_obj = epde.EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = bnd,
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

    control_var_tokens = epde.interface.prepared_tokens.ControlVarTokens(sample = control, ann = control_ann, 
                                                                         arg_var = [(0, [None]),
                                                                                    (1, [None])])

    eps = 5e-7
    popsize = 24
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=200)

    factors_max_number = {'factors_num' : [1, 2, 3,], 'probas' : [0.2, 0.65, 0.15]}

    custom_grid_tokens = epde.GridTokens(dimensionality = dimensionality, max_power=1)
    
    epde_search_obj.fit(data=[u, v], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow = 2,
                        additional_tokens=[custom_grid_tokens, control_var_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-7, 1e-5))
    epde_search_obj.equations()
    return epde_search_obj

def translate_dummy_eqs(t: np.ndarray, u: np.ndarray, v: np.ndarray, control: np.ndarray, diff_method = 'FD',
                        bnd = 30, control_ann: torch.nn.Sequential = None, data_nn: torch.nn.Sequential = None, 
                        device = 'cpu'):
    dimensionality = t.ndim - 1
    
    epde_search_obj = epde.EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = bnd,
                                      coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : True})    

    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max': 50000, 'device': device})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 4}) 
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})
    else:
        raise ValueError('Incorrect preprocessing tool selected.')


    control_var_tokens = epde.interface.prepared_tokens.ControlVarTokens(sample = control, ann = control_ann, 
                                                                         arg_var = [(0, [None]),
                                                                                    (1, [None])])#,
                                                                                    # (0, [0,]),
                                                                                    # (1, [0,])])   

    epde_search_obj.create_pool(data=[u, v], variable_names=['u', 'v'], max_deriv_order=(1,),
                                additional_tokens = [control_var_tokens,], data_nn = data_nn, device = device)


    eq_u = '20. * u{power: 1} + -1. * ctrl{power: 1} * u{power: 1} + -20. * u{power: 1} * v{power: 1} + 0 = du/dx0{power: 1}'
    eq_v = '-20. * v{power: 1} + -1. * ctrl{power: 1} * v{power: 1} + 20. * u{power: 1} * v{power: 1} + 0 = dv/dx0{power: 1}'

    test = epde.interface.equation_translator.translate_equation({'u': eq_u, 'v': eq_v}, pool = epde_search_obj.pool,
                                                                  all_vars = ['u', 'v'])
    return test

def optimize_ctrl(eq: epde.structure.main_structures.SoEq, t: torch.tensor,
                  u_tar: float, v_tar: float, u_init: float, v_init: float,
                  state_nn_pretrained: torch.nn.Sequential, ctrl_nn_pretrained: torch.nn.Sequential, 
                  fig_folder: str, device = 'cpu'):
    
    from epde.supplementary import AutogradDeriv
    autograd = AutogradDeriv()

    loc = control_utils.ConstrLocation(domain_shape = (t.size()[0],), device=device) # Declaring const in the entire domain
    u_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = u_tar, device=device),
                                                 indices = loc, deriv_method = autograd, nn_output=0, device=device)
    v_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = v_tar, device=device),
                                                 indices = loc, deriv_method = autograd, nn_output=1, device=device)
    contr_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = 0., device=device),
                                                 indices = loc, deriv_method = autograd, nn_output=0, device=device)

    u_var_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=0, device=device)
    v_var_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=1, device=device)
    contr_non_neg = control_utils.ControlConstrNEq(val = torch.full_like(input = t, fill_value = 0., device=device), sign='>',
                                                   indices = loc, deriv_method = autograd, nn_output=0, device=device)

    
    loss = control_utils.ConditionalLoss([(1., u_tar_constr, 0),
                                          (1., v_tar_constr, 0), 
                                        #   (0.1, contr_constr, 1),
                                          (100., u_var_non_neg, 0),
                                          (100., v_var_non_neg, 0),
                                          (100., contr_non_neg, 1)])
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

    bop_u = get_ode_bop('u', 0, [None], t[0, 0], u_init)
    bop_v = get_ode_bop('u', 1, [None], t[0, 0], v_init)

    optimizer.system = eq

    solver_params = {'full':     {'training_params': {'epochs': 1500,}, 'optimizer_params': {'params': {'lr': 1e-5}}}, 
                     'abridged': {'training_params': {'epochs': 300,}, 'optimizer_params': {'params': {'lr': 5e-5}}}}

    state_nn, ctrl_net, ctrl_pred, hist = optimizer.train_pinn(bc_operators = [(bop_u(device=device), 0.3),
                                                                               (bop_v(device=device), 0.3)],
                                                               grids = [t,], n_control = 1., 
                                                               state_net = state_nn_pretrained, 
                                                               opt_params = [0.0001, 0.9, 0.999, 1e-8],
                                                               control_net = ctrl_nn_pretrained, epochs = 150,
                                                               fig_folder = fig_folder, eps = 1e0, 
                                                               solver_params = solver_params)

    return state_nn, ctrl_net, ctrl_pred, hist


if __name__ == '__main__':
    import pickle

    experiment = 'LV'
    explicit_cpu = False
    device = 'cuda' if (torch.cuda.is_available and not explicit_cpu) else 'cpu'
    print(f'Working on {device}')

    res_folder = '/home/mikemaslyaev/Documents/EPDE/projects/control'
    fig_folder = os.path.join(res_folder, 'figs')

    t, ctrl, solution = prepare_data(steps_num = 201, ctrl_fun = lambda x: 12*x[1] + 0.05*x[0] + 0.2) # x[0]
    t, ctrl, solution = t[:-1], ctrl[:-1], solution[:-1, ...]

    print(t.shape, ctrl.shape, solution.shape)
    plt.plot(t, solution[:, 0], color = 'k', label = 'Prey, relative units')
    plt.plot(t, solution[:, 1], color = 'r', label = 'Hunters, relative units')
    plt.plot(t, ctrl, '*', color = 'y', label = 'control variable')
    plt.legend()
    plt.show()

    try:
        if device == 'cpu':
            fname = os.path.join(res_folder, f"data_ann_{experiment}_cpu.pickle")
        else:
            fname = os.path.join(res_folder, f"data_ann_{experiment}_cuda.pickle")
        with open(fname, 'rb') as data_input_file:  
            data_nn = pickle.load(data_input_file)
        data_nn = data_nn.to(device)
        save_nn = False
    except FileNotFoundError:
        print('No model located, ')
        data_nn = None
        save_nn = True
    # print(f'next(data_nn.parameters()).is_cuda {next(data_nn.parameters()).is_cuda}')
    # data_nn = None

    model = translate_dummy_eqs(t, solution[:, 0], solution[:, 1], ctrl, data_nn = data_nn, device = device) # , 
    example_sol = epde.globals.solution_guess_nn(torch.from_numpy(t).reshape((-1, 1)).float().to(device))
    epde.globals.solution_guess_nn.to(device)
    print(f'example_sol: {type(example_sol)}, {example_sol.shape}, {example_sol.get_device()}')
    if save_nn:
        if device == 'cpu':
            fname =  os.path.join(res_folder, f"data_ann_{experiment}_cpu.pickle")
        else:
            fname = os.path.join(res_folder, f"data_ann_{experiment}_cuda.pickle")
        with open(fname, 'wb') as output_file:
            pickle.dump(epde.globals.solution_guess_nn, output_file)



    def create_shallow_nn(arg_num: int = 1, output_num: int = 1, device = 'cpu') -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
        hidden_neurons = 50
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


    # preprocessor_kwargs = {'epochs_max' : 10000}
    
    def prepare_derivs(var_name: str, var_array: np.ndarray, grid: np.ndarray, max_order: tuple = (1,)):
        default_preprocessor_type = 'FD'
        preprocessor_kwargs = {}#{'use_smoothing' : False,
                              #  'include_time' : True}        
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
    
    der_names_u, derivatives_u = prepare_derivs('u', var_array = solution[:, 0], grid = t)
    der_names_v, derivatives_v = prepare_derivs('v', var_array = solution[:, 1], grid = t)
    
    args = torch.from_numpy(solution).float().to(device)
    # args = torch.cat([args, torch.from_numpy(derivatives_u).float().to(device), 
                    #   torch.from_numpy(derivatives_v).float().to(device)], dim=1)
    print(f'args.shape is {args.shape}')
    # print(f'derivatives_u.shape {derivatives_u.shape, type(derivatives_u)}')
    # plt.plot(t, derivatives_u, color = 'k')
    # plt.plot(t, derivatives_v, color = 'r')
    # plt.plot(t, solution[:, 0], '*', color = 'k')
    # plt.plot(t, solution[:, 1], '*', color = 'r')
    # plt.show()

    # raise NotImplementedError('Fin!')

    nn = 'deep' # nn = 'shallow'
    load_ctrl = False

    if device == 'cpu':
        ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cpu.pickle")
        # f"/home/mikemaslyaev/Documents/EPDE/projects/control/control_ann_{nn}_cpu.pickle"
    else:
        ctrl_fname = os.path.join(res_folder, f"control_ann_{nn}_{experiment}_cuda.pickle")
    if load_ctrl:
        with open(ctrl_fname, 'rb') as ctrl_input_file:  
            ctrl_ann = pickle.load(ctrl_input_file)
    else:
        nn_method = create_shallow_nn if nn == 'shallow' else create_deep_nn
        ctrl_ann = epde.supplementary.train_ann(args=[solution[:, 0], solution[:, 1]],#, 
                                                    #   derivatives_u.reshape(-1), 
                                                    #   derivatives_v.reshape(-1)], 
                                                data = ctrl, epochs_max = 1e5, dim = 2, 
                                                model = nn_method(2, 1, device=device),
                                                device = device)

        with open(ctrl_fname, 'wb') as ctrl_output_file:
            pickle.dump(ctrl_ann, file = ctrl_output_file)

    # ctrl_vals = ctrl_ann(example_sol)


    @torch.no_grad()
    def eps_increment_diff(input_params: OrderedDict, loc, forward: bool = True, eps = 1e-4):
        print(loc[1:])
        if forward:
            input_params[loc[0]][tuple(loc[1:])] += eps
        else:
            input_params[loc[0]][tuple(loc[1:])] -= 2*eps
        return input_params
    
    play_with_params = True
    if play_with_params:
        eps = 5e-1
        ctrl_alt_p = deepcopy(ctrl_ann)
        state_dict_alt = ctrl_alt_p.state_dict()
        print(f'state_dict_alt["0.weight"] {state_dict_alt["0.weight"].shape}, with value of {state_dict_alt["0.weight"][10, 0]}')        
        # print(f'ctrl_alt.keys()[0] {list(state_dict_prev.keys())[0]}')
        # state_dict_prev['0.weight'][0, 0]
        state_dict_alt = eps_increment_diff(state_dict_alt, loc = ['0.weight', 10, 0], forward=True, eps = eps)
        print(f'state_dict_alt["0.weight"] {state_dict_alt["0.weight"].shape}, with value of {state_dict_alt["0.weight"][10, 0]}')
        ctrl_alt_p.load_state_dict(state_dict_alt)
        print(f'ctrl_alt_p(args).cpu().detach().numpy() min val is {np.min(ctrl_alt_p(args).cpu().detach().numpy())}, max is {np.max(ctrl_alt_p(args).cpu().detach().numpy())}')        

        ctrl_alt_m = deepcopy(ctrl_ann)
        state_dict_alt = ctrl_alt_m.state_dict()
        # print(f'ctrl_alt.keys()[0] {list(state_dict_prev.keys())[0]}')
        # state_dict_prev['0.weight'][0, 0]
        state_dict_alt = eps_increment_diff(state_dict_alt, loc = ['0.weight', 10, 0], forward=True, eps = -eps)
        print(f'state_dict_alt["0.weight"] {state_dict_alt["0.weight"].shape}, with value of {state_dict_alt["0.weight"][10, 0]}')
        ctrl_alt_m.load_state_dict(state_dict_alt)
        print(f'ctrl_alt_m(args).cpu().detach().numpy() min val is {np.min(ctrl_alt_m(args).cpu().detach().numpy())}, max is {np.max(ctrl_alt_m(args).cpu().detach().numpy())}')    
    # raise NotImplementedError

    print(f'ctrl_ann is on cuda: {next(ctrl_ann.parameters()).is_cuda}, ')
    plt.plot(t, ctrl_ann(args).cpu().detach().numpy(), color = 'b', label = 'control variable, nn approx on input states')
    if play_with_params:
        plt.plot(t, ctrl_alt_m(args).cpu().detach().numpy(), color = 'g', label = 'control variable, altered -')
        plt.plot(t, ctrl_alt_p(args).cpu().detach().numpy(), color = 'r', label = 'control variable, altered +')        
    # plt.plot(t, ctrl_ann(example_sol).cpu().detach().numpy(), '--', color = 'k', label = 'control variable, nn approx with nn approx state')
    plt.plot(t, ctrl, '*', color = 'y', label = 'control variable')
    plt.legend()
    plt.show()

    # raise NotImplementedError()
    res = optimize_ctrl(model, torch.from_numpy(t).reshape((-1, 1)).float().to(device), u_tar = 1.5, v_tar = 0.5,
                        u_init=solution[0, 0], v_init=solution[0, 1],
                        state_nn_pretrained=epde.globals.solution_guess_nn, ctrl_nn_pretrained=ctrl_ann, 
                        fig_folder=fig_folder, device=device)

    savename = f'res_{time_exp_start.month}_{time_exp_start.day}_at_{time_exp_start.hour}_{time_exp_start.minute}_{experiment}.pickle'
    # plt.savefig(os.path.join(fig_folder, frame_name))
    with open(os.path.join(res_folder, savename), 'wb') as output_file:  
        pickle.dump(res, output_file)

    # u_plots, v_plots = torch.linspace(0, 6, 61), torch.linspace(0, 6, 61)
    # UU, VV = torch.meshgrid(u_plots, v_plots)
    # ctrl_args = torch.stack(tensors = (UU.reshape(-1), VV.reshape(-1)), dim = 0).T

    # ctrl_landscape = res[1](ctrl_args).cpu().detach().numpy().reshape((u_plots.size()[0], v_plots.size()[0]))
    # np.save(file = "/home/mikemaslyaev/Documents/EPDE/projects/control/ctrl_landscape.npy", arr=ctrl_landscape)