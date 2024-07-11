import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
import torch

from typing import Callable, Union
from collections import OrderedDict

import matplotlib as mpl
mpl.use('TkAgg')
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

def prepare_data(steps_num: int = 151, t_max: float = 1, ctrl_fun: Callable = lambda x: x[0]*x[1]):
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

    
    # control_var_tokens = epde.CacheStoredTokens('control', ['ctrl',], {'ctrl' : control}, OrderedDict([('power', (1, 1))]),
    #                                             {'power': 0}, meaningful=True)

    # control_ann = global_var.control_nn.net
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
                        bnd = 30, control_ann: torch.nn.Sequential = None, data_nn: torch.nn.Sequential = None):
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

    epde_search_obj.create_pool(data=[u, v], variable_names=['u', 'v'], max_deriv_order=(1,),
                                additional_tokens = [control_var_tokens,], data_nn = data_nn)


    eq_u = '20. * u{power: 1} + -1. * ctrl{power: 1} * u{power: 1} + -20. * u{power: 1} * v{power: 1} + 0 = du/dx0{power: 1}'
    eq_v = '-20. * v{power: 1} + 1. * ctrl{power: 1} * v{power: 1} + 20. * u{power: 1} * v{power: 1} + 0 = dv/dx0{power: 1}'

    test = epde.interface.equation_translator.translate_equation({'u': eq_u, 'v': eq_v}, pool = epde_search_obj.pool)
    return test

def optimize_ctrl(eq: epde.structure.main_structures.SoEq, t: torch.tensor,
                  u_tar: float, v_tar: float, u_init: float, v_init: float,
                  state_nn_pretrained: torch.nn.Sequential, ctrl_nn_pretrained: torch.nn.Sequential):
    
    from epde.supplementary import AutogradDeriv
    autograd = AutogradDeriv()

    loc = control_utils.ConstrLocation(domain_shape = (t.size()[0],)) # Declaring const in the entire domain
    u_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = u_tar),
                                                 indices = loc, deriv_method = autograd, nn_output=0)
    v_tar_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = v_tar),
                                                 indices = loc, deriv_method = autograd, nn_output=1)
    contr_constr = control_utils.ControlConstrEq(val = torch.full_like(input = t, fill_value = 0.),
                                                 indices = loc, deriv_method = autograd, nn_output=0)
    
    loss = control_utils.ConditionalLoss([(100., u_tar_constr, 0),
                                          (100., v_tar_constr, 0),
                                          (0.001, contr_constr, 1)])
    optimizer = control_utils.ControlExp(loss=loss)
    
    def get_ode_bop(key, var, term, grid_loc, value):
        bop = epde.interface.solver_integration.BOPElement(axis = 0, key = key, term = term,
                                                           power = 1, var = var)
        if isinstance(grid_loc, float):
            bop_grd_np = np.array([[grid_loc,]])
            bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        elif isinstance(grid_loc, torch.Tensor):
            bop.set_grid(grid_loc.reshape((1, 1)).type(torch.FloatTensor)) # What is the correct shape here?
        else:
            raise TypeError('Incorret value type, expected float or torch.Tensor.')
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        return bop

    bop_u = get_ode_bop('u', 0, [None], t[0, 0], u_init)
    bop_v = get_ode_bop('u', 0, [None], t[0, 0], v_init)

    optimizer.system = eq

    optimizer.set_control_optim_params()
    optimizer.set_solver_params()

    state_nn, ctrl_net, ctrl_pred = optimizer.train_pinn(bc_operators = [bop_u(), bop_v()], grids = [t,], 
                                                         n_control = 1., state_net = state_nn_pretrained, 
                                                         control_net = ctrl_nn_pretrained, epochs = 1e2)

    return state_nn, ctrl_net, ctrl_pred


if __name__ == '__main__':
    import pickle

    t, ctrl, solution = prepare_data(ctrl_fun = lambda x: 17*x[1] + 0.05*x[0] + 0.2) # x[0]
    t, ctrl, solution = t[:-1], ctrl[:-1], solution[:-1, ...]

    print(t.shape, ctrl.shape, solution.shape)
    plt.plot(t, solution[:, 0], color = 'k', label = 'Prey, relative units')
    plt.plot(t, solution[:, 1], color = 'r', label = 'Hunters, relative units')
    plt.plot(t, ctrl, '*', color = 'y', label = 'control variable')
    plt.legend()
    plt.show()

    with open(r"/home/maslyaev/Documents/EPDE/projects/control/data_ann.pickle", 'rb') as data_input_file:  
        data_nn = pickle.load(data_input_file)
    # data_nn = None

    model = translate_dummy_eqs(t, solution[:, 0], solution[:, 1], ctrl, data_nn = data_nn) # , 
    # with open(r"/home/maslyaev/Documents/EPDE/projects/control/data_ann.pickle", 'wb') as output_file:  
    #     pickle.dump(epde.globals.solution_guess_nn, output_file)

    args = torch.from_numpy(solution).float()

    def create_shallow_nn(arg_num: int = 1, output_num: int = 1) -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
        hidden_neurons = 180
        layers = [torch.nn.Linear(arg_num, hidden_neurons),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_neurons, output_num)]
        return torch.nn.Sequential(*layers)
    
    ctrl_ann = epde.supplementary.train_ann(args=[solution[:, 0], solution[:, 1]], data = ctrl, 
                                            epochs_max = 1e4, dim = 2, model = create_shallow_nn(2, 1))
    
    # with open(r"/home/maslyaev/Documents/EPDE/projects/control/control_ann.pickle", 'rb') as ctrl_input_file:  
    #     ctrl_ann = pickle.load(ctrl_input_file)
    with open(r"/home/maslyaev/Documents/EPDE/projects/control/control_ann_shallow.pickle", 'wb') as ctrl_output_file:  
        pickle.dump(ctrl_ann, file = ctrl_output_file)

    plt.plot(t, ctrl_ann(args).detach().numpy(), color = 'b', label = 'control variable, nn approx')
    plt.plot(t, ctrl, '*', color = 'y', label = 'control variable')
    plt.legend()
    plt.show()   

    res = optimize_ctrl(model, torch.from_numpy(t).reshape((-1, 1)).float(), u_tar = 1, v_tar = 0,
                         u_init=solution[0, 0], v_init=solution[0, 1],
                         state_nn_pretrained=epde.globals.solution_guess_nn, ctrl_nn_pretrained=ctrl_ann)

    with open(r"/home/maslyaev/Documents/EPDE/projects/control/control_opt_res.pickle", 'wb') as output_file:  
        pickle.dump(res, output_file)