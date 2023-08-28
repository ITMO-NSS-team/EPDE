import numpy as np
import pandas as pd

import time
import torch
import os
import sys
from functools import reduce

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('C:/Users/Mike/Documents/Work/EPDE')

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import lotka

import pysindy as ps

from epde.interface.logger import Logger
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, CustomEvaluator, CustomTokens

from epde.interface.equation_translator import translate_equation

def translate_sindy_eq(equation):
    print(equation)
    correspondence = {"0" : "u{power: 1.0}",
                      "0_1" : "du/dx2{power: 1.0}",
                      "0_11" : "d^2u/dx2^2{power: 1.0}",
                      "0_111" : "d^3u/dx2^3{power: 1.0}",
                      "0_1" : "du/dx1{power: 1.0}",
                      "1" : "v{power: 1.0}",
                      "1_1" : "dv/dx1{power: 1.0}",}
                        
     # Check EPDE translator input format
    
    def replace(term):
        term = term.replace(' ', '').split('x')
        for idx, factor in enumerate(term[1:]):
            try:
                if '^' in factor:
                    factor = factor.split('^')
                    term[idx+1] = correspondence[factor[0]].replace('{power: 1.0}', '{power: '+str(factor[1])+'.0}')
                else:
                    term[idx+1] = correspondence[factor]
            except KeyError:
                print(f'Key of term {factor} is missing')
                raise KeyError()
        return term
                
    if isinstance(equation, str):
        terms = []        
        split_eq = equation.split('+')
        const = split_eq[0][:-2]
        if 'x' in const:
            for term in split_eq:
                print('To replace:', term, replace(term))
                terms.append(reduce(lambda x, y: x + ' * ' + y, replace(term)))
            terms_comb = reduce(lambda x, y: x + ' + ' + y, terms) + ' + 0.0 = du/dx1{power: 1.0}'
        else:
            for term in split_eq[1:]:
                print('To replace:', term, replace(term))
                terms.append(reduce(lambda x, y: x + ' * ' + y, replace(term)))
            terms_comb = reduce(lambda x, y: x + ' + ' + y, terms) + ' + ' + const + ' = du/dx1{power: 1.0}'
        return terms_comb        
    elif isinstance(equation, list):
        assert len(equation) == 2
        var_list = ['u', 'v', 'w']
        #rp_term_list = [' + 0.0 = du/dx1{power: 1.0}', ' + 0.0 = dv/dx1{power: 1.0}']
        eqs_tr = []
        for idx, eq in enumerate(equation):
            terms = []            
            split_eq = eq.split('+')
            const = split_eq[0][:-2]
            if 'x' in const:
                for term in split_eq:
                    print('To replace:', term, replace(term))                
                    terms.append(reduce(lambda x, y: x + ' * ' + y, replace(term)))
                terms_comb = reduce(lambda x, y: x + ' + ' + y, terms) + ' + 0.0 = d' + var_list[idx] + '/dx1{power: 1.0}'
            else:
                for term in split_eq[1:]:
                    print('To replace:', term, replace(term))                
                    terms.append(reduce(lambda x, y: x + ' * ' + y, replace(term)))
                terms_comb = reduce(lambda x, y: x + ' + ' + y, terms) + ' + ' + const + ' = d' + var_list[idx] + '/dx1{power: 1.0}'
            eqs_tr.append(terms_comb)
        print('Translated system:', eqs_tr)
        return eqs_tr
    else:
        raise NotImplementedError()


def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.title(title)
    plt.show()
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')

def epde_discovery(x, t, u, use_ann = False, smooth = False): #(grids, data, use_ann = False):
    grids = np.meshgrid(t, x, indexing = 'ij')
    print(u.shape, grids[0].shape, grids[1].shape)
    multiobjective_mode = True
    dimensionality = u.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 35000})# 35000
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : smooth, 'sigma' : 1, 
                                                              'polynomial_window' : 5, 'poly_order' : 4}) # 'epochs_max' : 10000})# 
    popsize = 9
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=55)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize,
                                                   training_epochs=85)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)   

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                            eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                      token_labels=['cos(t)sin(x)'],
                                      evaluator=custom_trig_evaluator,
                                      params_ranges=trig_params_ranges,
                                      params_equality_ranges=trig_params_equal_ranges,
                                      meaningful=True, unique_token_type=False)    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-9, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    print(u.shape, grids[0].shape)
    epde_search_obj.create_pool(data = u, variable_names=['u'], max_deriv_order=(1, 3),  
                                additional_tokens=[custom_trig_tokens, custom_grid_tokens])
    

    epde_search_obj.fit(data=u, variable_names=['u',], max_deriv_order=(1, 3),
                        equation_terms_max_number=6, data_fun_pow = 1, additional_tokens=[custom_trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    
    equation_obtained = False; compl = [6.,]; attempt = 0
    
    iterations = 4
    while not equation_obtained:
        if attempt < iterations:
            try:
                sys = epde_search_obj.get_equations_by_complexity(compl)
                res = sys[0]
            except IndexError:
                compl[0] += 0.5
                attempt += 1
                continue
        else:
            res = epde_search_obj.equations(only_print = False)[0][0]
        equation_obtained = True
    
    return epde_search_obj, res

def get_epde_pool(x, t, u, use_ann = False):
    grids = np.meshgrid(t, x, indexing = 'ij')
    print(u.shape, grids[0].shape, grids[1].shape)
    multiobjective_mode = True
    dimensionality = u.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 20,
                                          coordinate_tensors = grids)    
    
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 35000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : True, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 

    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)   

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                            eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                      token_labels=['cos(t)sin(x)'],
                                      evaluator=custom_trig_evaluator,
                                      params_ranges=trig_params_ranges,
                                      params_equality_ranges=trig_params_equal_ranges,
                                      meaningful=True, unique_token_type=False)    
    
    opt_val = 1e-1
    bounds = (1e-9, 1e0) if multiobjective_mode else (opt_val, opt_val)    

    
    epde_search_obj.create_pool(data = u, variable_names=['u'], max_deriv_order=(1, 3),  
                                additional_tokens=[custom_trig_tokens, custom_grid_tokens])

    return epde_search_obj.pool

def sindy_discovery(grids, u):
    t = np.unique(grids[0])
    x = np.unique(grids[1])    
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    u = u.T.reshape(len(x), len(t), 1)
    
    # задаем свои токены через лямбда выражения
    library_functions = [lambda x: x, lambda x: x * x]#, lambda x: np.cos(x)*np.cos(x)]#, lambda x: 1/x]
    library_function_names = [lambda x: x, lambda x: x + x]#, lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']#, lambda x: '1/'+x]
    
    # ? проблема с использованием multiindices
    # multiindices=np.array([[0,1],[1,1],[2,0],[3,0]])
    
    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=3, spatial_grid=x,
                            # multiindices=multiindices,
                            implicit_terms=True, temporal_grid=t,
                            include_bias=True, is_uniform=True, include_interaction=True)
    feature_library = ps.feature_library.PolynomialLibrary(degree=3)
    optimizer = ps.SR3(threshold=0, max_iter=10000, tol=1e-15, nu=1e2,
                        thresholder='l0', normalize_columns=True)
    
    # optimizer = ps.STLSQ(threshold=50, alpha=1e-15, 
    #                       normalize_columns=True, max_iter=200)    
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)  
    return model

def sindy_provided(grids, u):
    t = np.unique(grids[0])
    x = np.unique(grids[1])        
    u = u.T.reshape(len(x), len(t), 1)
    
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        is_uniform=True,
    )
    
    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()
    return model
            
def sindy_provided_l0_old(grids, u):
    t = np.unique(grids[0])
    print('t.shape', t.shape)
    x = np.unique(grids[1])        
    u = u.T.reshape(len(x), len(t), 1)
    
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        is_uniform=True,
    )    
    
    print('SR3 model, L0 norm: ')
    optimizer = ps.SR3(
        threshold=0.5,
        max_iter=10000,
        tol=1e-15,
        nu=1e2,
        thresholder="l0",
        normalize_columns=True,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=t[1] - t[0])
    model.print()    
    return model

def sindy_provided_l0(grids, u):
    t = np.unique(grids[0])
    print('t.shape', t.shape)
    x = np.unique(grids[1])        
    u = u.T.reshape(len(x), len(t), 1) # 
    
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        is_uniform=True,
    )    
    
    print('SR3 model, L0 norm: ')
    optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
                   thresholder='l0', normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=t[1] - t[0])
    model.print()    
    return model

if __name__ == "__main__":
    kdV = loadmat('C:\\Users\\Mike\\Documents\\Work\\EPDE\\projects\\wSINDy\\data\\KdV\\kdv.mat')
    t = np.ravel(kdV['t'])
    x = np.ravel(kdV['x'])
    u = np.real(kdV['usol']).T

    print(u.shape, t.shape, t[1]-t[0], x[1]- x[0])
    raise NotImplementedError()
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    train_max = 200    
    grids = np.meshgrid(t, x, indexing = 'ij')
    grids_training = (grids[0][:train_max, ...], grids[1][:train_max, ...])
    grids_test = (grids[0][train_max:, ...], grids[1][train_max:, ...])    

    t_train, t_test = t[:train_max], t[train_max:]
    data_train, data_test = u[:train_max, ...], u[train_max:, ...]
    '''
    EPDE side
    '''

    run_epde = True #True
    run_sindy = False

    exps = {}
    test_launches = 5
    magnitudes = [0, 1.*1e-2, 2.5*1e-2, 5.*1e-2, 1.*1e-1]# 1.5 * 1e-1, 2. * 1e-1, 2.5 * 1e-1]
    for magnitude in magnitudes:
        data_train_n = data_train + np.random.normal(scale = magnitude * np.abs(data_train), size = data_train.shape)
        
        #Heatmap(data_train_n, title = 'Data')
        
        errs_epde = []
        models_epde = []
        calc_epde = []
        pool = None
        
        if run_epde:
            for idx in range(test_launches):
                t1 = time.time()
                epde_search_obj, sys = epde_discovery(x, t_train, data_train_n, False)
                t2 = time.time()
                if pool is None:
                    pool = epde_search_obj.pool                
                try:
                    logger.add_log(key = f'KdV_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('epde', magnitude), time = t2 - t1)
                except NameError:
                    logger = Logger(name = 'logs/KdV_0_from_mat.json', referential_equation = '1.0 * d^3u/dx2^3{power: 1.0} + 6.0 * u{power: 1.0} * du/dx2{power: 1.0}  + 0.0 = du/dx1{power: 1.0}', 
                                    pool = epde_search_obj.pool)
                    logger.add_log(key = f'KdV_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('epde', magnitude), time = t2 - t1)
        if run_sindy:
            if pool is None:
                pool = get_epde_pool(x, t_train, data_train_n)
            t1 = time.time()
            model_base = sindy_provided_l0(grids_training, data_train_n)
            t2 = time.time()
            sys = translate_equation(translate_sindy_eq(model_base.equations()[0]), pool)            
            try:
                logger.add_log(key = f'Burgers_sindy_{magnitude}', entry = sys, aggregation_key = ('sindy', magnitude), time = t2 - t1)
            except NameError:
                logger = Logger(name = 'logs/Burgers_SINDy_new.json', referential_equation = '1.0 * d^3u/dx2^3{power: 1.0} + 6.0 * u{power: 1.0} * du/dx2{power: 1.0}  + 0.0 = du/dx1{power: 1.0}', 
                                pool = pool)
                logger.add_log(key = f'Burgers_sindy_{magnitude}', entry = sys, aggregation_key = ('sindy', magnitude), time = t2 - t1)
            errs_sindy, calc_sindy = None, None

        else:
            model_base, errs_sindy, calc_sindy = None, None, None
        
        exps[magnitude] = {'epde': (models_epde, errs_epde, calc_epde),
                           'SINDy': (model_base, errs_sindy, calc_sindy)}
    logger.dump()
        
