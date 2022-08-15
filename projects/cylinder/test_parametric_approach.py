#import pandas as pd

import os
import numpy as np
import pandas as pd

import epde.interface.interface as epde_alg
import pickle

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens, TrigonometricTokens
from epde.interface.prepared_tokens import Velocity_HEQ_tokens, ConstantToken
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

from epde.prep.interpolation_oversampling import BesselInterpolator
from epde.parametric.parametric_eq_translator import optimize_parametric_form

def test_parametric():
    filename = '/home/maslyaev/epde/EPDE_parametric/projects/parametric_experiments/data/0_6_down_exp_N.dat'
    file = np.loadtxt(filename, skiprows = 2)
    file = file[:1000, :]
    dimensionality = file.ndim
    
    time = file[:, 0]/1000.
    x = np.array([0.1, 0.6, 1.1, 1.6, 2.1])

    temp_idxs = range(1, 16, 3)
    u_smol = file[:, temp_idxs]

    boundary = [10, 5]
    
    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)    
    
    # print(x, oversampling_x, u_smol.shape)
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                   for idx in range(u_smol.shape[0])])
    

    grids = np.meshgrid(time, oversampling_x, indexing = 'ij')

    boundary = 10

    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)
                                                                     
    # custom_grid_tokens = CacheStoredTokens(token_type = 'grid', 
    #                                          boundary = boundary,
    #                                          token_labels = ['t', 'r'], 
    #                                          token_tensors={'t' : grids[0], 'r' : grids[1]},
    #                                          params_ranges = {'power' : (1, 1)},
    #                                          params_equality_ranges = None)

    velocity_params = [(-0.0390729, 0.0390729), (-2., 2.), (-100., 100.), 
                       (-0.00294, 0.00294), (-0.218, 0.218), (-20., 20.), 
                       (-0.00003102, 0.00003102), (-0.001, 0.001), (-0.01, 0.01),
                       (-3., 3.), (-80., 80.), (-50000., 50000.), 
                       (-0.0006, 0.0006), (-0.04, 0.04), (-5., 5.)] # Insert parameter ranges
    velocity_tokens = Velocity_HEQ_tokens(velocity_params)

    const_tokens = ConstantToken(values_range = (-10, 10))

    # trig_tokens = TrigonometricTokens(freq = (0.95, 1.05))

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
                                               eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = CustomTokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                          token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                      # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                          evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                          params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                          params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.

    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary, coordinate_tensors = grids, 
                        additional_tokens = [velocity_tokens, const_tokens, custom_inv_fun_tokens], method='poly', 
                        method_kwargs = {'smooth': True, 'grid': grids, 'sigma' : 3},
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    print('kek')
    
    velocity_str = 'v{power : 1, p1 : None, p2 : None, p3 : None, p4 : None, p5 : None, p6 : None, p7 : None, p8 : None, p9 : None, p10 : None, p11 : None, p12 : None, p13 : None, p14 : None, p15 : None}'
    equation_form = [['const{power: 1, value : None}', '1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], #'const{power: 1, value : None}', 
                     ['const{power: 1, value : None}', 'd^2u/dx2^2{power: 1}'], # 'const{power: 1, value : None}', 
                     [velocity_str, 'du/dx2{power: 1}'], # 'const{power: 1, value : None}', 
                     ['du/dx1{power: 1}']]
    
    random_initial_params = np.array([0.5, 0.5] + [np.random.uniform(bounds[0], bounds[1]) for bounds in velocity_params])
    
    return optimize_parametric_form(equation_form, epde_search_obj.pool, 
                                    initial_params = random_initial_params)
    
def experimental_diffusion_only():
    dirname = '/home/maslyaev/epde/EPDE_parametric/projects/parametric_experiments/data/Diffusion/'
    temp_initial = {7 : 24.86, 
                    12 : 25.0307, 
                    17 : 25.15106, 
                    22 : 26.0285,
                    27 : 26.05125}    
    
    def load_data(directory, data_file_format = 'dat', loader_kwargs = {}, 
                  temp_initial = {}, voltage_prefix = '06'):
        data = []
        t = None
        x = 1e-4 * np.array(list(temp_initial.keys()))
        
        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith(".dat"):
                temp = pd.read_csv(directory + '/' + filename, **loader_kwargs)
                if t is None:
                    t = temp.loc[temp.tc >= 0].ts.to_numpy()
                dist = int(filename.replace(voltage_prefix + 'V_', '').replace('mm_u.dat', ''))
                      
                data.append((dist, temp.loc[temp.tc >= 0].Tr.to_numpy()))
        len_min = min([series.size for _, series in data])
        data = sorted([(dist, series[:len_min]) for dist, series in data], 
                      key = lambda x: x[0])
        
        return t, x, np.stack([series - temp_initial[dist] for dist, series in data])    

    t_min = 15000
    t_max = 25000  
    t, x, u_smol = load_data(dirname, 
                             loader_kwargs={'skiprows':2, 
                                            'sep' : '\t', 
                                            'error_bad_lines' : False, 
                                            'names' : ["ts","tc","Tr","Tm"]},
                             temp_initial = temp_initial)
    u_smol = u_smol.T[t_min:t_max, :]; t = t[t_min:t_max]
    # file = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Data_32_points_.dat', 
    #                   delimiter=' ', usecols=range(33))
    
    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)
    
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                    for idx in range(u_smol.shape[0])])

    grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    boundary = [10, 1]

    dimensionality = u.ndim
        
    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}
    
    custom_inv_fun_tokens = CustomTokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                         token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                         # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                         evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                         params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                         params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.

    velocity_params = [(-0.0390729, 0.0390729), (-2., 2.), (-100., 100.), 
                       (-0.00294, 0.00294), (-0.218, 0.218), (-20., 20.), 
                       (-0.00003102, 0.00003102), (-0.001, 0.001), (-0.01, 0.01),
                       (-3., 3.), (-80., 80.), (-50000., 50000.), 
                       (-0.0006, 0.0006), (-0.04, 0.04), (-5., 5.)] # Insert parameter ranges
    velocity_tokens = Velocity_HEQ_tokens(velocity_params)

    const_tokens = ConstantToken(values_range = (-10, 10))

    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary, coordinate_tensors = grids, 
                                additional_tokens = [velocity_tokens, const_tokens, custom_inv_fun_tokens], method='poly', 
                                method_kwargs = {'smooth': True, 'grid': grids, 'sigma' : 3},
                                memory_for_cache=5, prune_domain = True,
                                division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    print('kek')
    
    velocity_str = 'v{power : 1, p1 : None, p2 : None, p3 : None, p4 : None, p5 : None, p6 : None, p7 : None, p8 : None, p9 : None, p10 : None, p11 : None, p12 : None, p13 : None, p14 : None, p15 : None}'
    equation_form = [['const{power: 1, value : None}', '1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], #'const{power: 1, value : None}', 
                     ['const{power: 1, value : None}', 'd^2u/dx2^2{power: 1}'], # 'const{power: 1, value : None}', 
                     [velocity_str, 'du/dx2{power: 1}'], # 'const{power: 1, value : None}', 
                     ['du/dx1{power: 1}']]
    
    random_initial_params = np.array([0.5, 0.5] + [np.random.uniform(bounds[0], bounds[1]) for bounds in velocity_params])
    
    return optimize_parametric_form(equation_form, epde_search_obj.pool, 
                                    initial_params = random_initial_params)    

res = experimental_diffusion_only()