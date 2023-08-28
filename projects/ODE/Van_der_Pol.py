#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:55:18 2023

@author: maslyaev
"""
import numpy as np
import time

import torch
import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('C:/Users/Mike/Documents/Work/EPDE')

import matplotlib.pyplot as plt
import matplotlib

from typing import Callable

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import lotka
from functools import reduce

import pysindy as ps

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement
from epde.interface.logger import Logger

from epde.supplementary import define_derivatives
from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

from epde.interface.equation_translator import translate_equation

def second_order_ODE_by_RK(initial: tuple, timestep: float, steps: int, epsilon: float):
    res = np.full(shape = (steps, 2), fill_value = initial, dtype=np.float64)
    for step in range(steps-1):
        # print(res[step, :])
        t = step*timestep
        k1 = res[step, 1] ; x1 = res[step, 0] + timestep/2. * k1
        l1 =  - epsilon*(res[step, 0]**2 - 1)*res[step, 1] - res[step, 0]; y1 = res[step, 1] + timestep/2. * l1

        k2 = y1; x2 = res[step, 0] + timestep/2. * k2
        l2 = - epsilon*(x1**2 - 1)*y1 - x1; y2 = res[step, 1] + timestep/2. * l2

        k3 = y2
        l3 = - epsilon*(x2**2 - 1)*y2 - x2
        
        x3 = res[step, 0] + timestep * k1 - 2 * timestep * k2 + 2 * timestep * k3
        y3 = res[step, 1] + timestep * l1 - 2 * timestep * l2 + 2 * timestep * l3
        k4 = y3
        l4 = - epsilon*(x3**2 - 1)*y3 - x3
        
        res[step+1, 0] = res[step, 0] + timestep / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        res[step+1, 1] = res[step, 1] + timestep / 6. * (l1 + 2 * l2 + 2 * l3 + l4)
    return res

# if __name__ == "__main__":
def prepare_data(initial = (np.sqrt(3)/2., 1./2.), step = 0.05, steps_num = 640, epsilon = 0.2):
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    solution = second_order_ODE_by_RK(initial=initial, timestep=step, steps=steps_num, 
                                      epsilon=epsilon)
    return t, solution

def translate_sindy_eq(equation):
    print(equation)
    correspondence = {"0" : "u{power: 1.0}",
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

def get_epde_pool(t, x, y, use_ann = False):
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t,])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 35000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : True, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                     # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 12
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=85)
    # trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    
    epde_search_obj.create_pool(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,)) 
                                # additional_tokens=[trig_tokens, custom_grid_tokens])

    return epde_search_obj.pool

def sindy_discovery(t, x, y, sparsity = 0.05):
    poly_order = 4
    # threshold = 0.05
    
    x_train = np.array([x, y]).T
    print(x_train.shape)
    
    # library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
    # library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
    model = ps.SINDy(
        optimizer=ps.STLSQ(alpha=sparsity),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(
        x_train,
        t=t[1] - t[0],
        # x_dot=x_dot_train_measured
        # + np.random.normal(scale=eps, size=x_train.shape),
        quiet=True,
    )
    return model

def epde_discovery_as_system(t, x, y, use_ann = False):
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t,])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 35000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : True, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                     # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 12
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=85)
    # trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2, 3], 'probas' : [0.4, 0.3, 0.3]}
    
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=6, data_fun_pow = 3, #additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4))
    '''
    Смотрим на найденное Парето-множество, 
    
    "идеальное уравнение" имеет вид:
     / -1.3048888807580532 * u{power: 1.0} * v{power: 1.0} + 0.3922851274813135 * u{power: 1.0} + -0.0003917278536547386 = du/dx1{power: 1.0}
     \ -0.9740492564964498 * v{power: 1.0} + 0.9717873909925121 * u{power: 1.0} * v{power: 1.0} + 0.0003500773115704403 = dv/dx1{power: 1.0}
    {'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.8, 0.2]}}, 
     ('sparsity', 'u'): {'optimizable': True, 'value': 0.00027172388370453704}, ('sparsity', 'v'): {'optimizable': True, 'value': 0.00019292375116125682}} , with objective function values of [0.2800438  0.18041074 4.         4.        ]         
    '''
    epde_search_obj.equations(only_print = True, num = 1)
    equation_obtained = False; compl = [5,]; attempt = 0
    
    
    while not equation_obtained:    
        try:
            sys = epde_search_obj.get_equations_by_complexity(compl)
            res = sys[0]
        except IndexError:
            compl[attempt % 2] += 1
            attempt += 1
            continue
        equation_obtained = True
    return epde_search_obj, res

def epde_discovery_as_ode(t, x, y, use_ann = False):
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 50,
                                           coordinate_tensors = [t,])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 35000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : True, 'sigma' : 1, 
                                                              'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                     # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 12
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=100)
    # trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2, 3], 'probas' : [0.4, 0.3, 0.3]}
    
    epde_search_obj.fit(data=[x,], variable_names=['u',], max_deriv_order=(2,),
                        equation_terms_max_number=6, data_fun_pow = 2, #additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-3))

    epde_search_obj.equations(only_print = True, num = 1)
    equation_obtained = False; compl = [5,]; attempt = 0
    
    
    while not equation_obtained:
        if attempt > 5:
            res = epde_search_obj.equations(False)[0][0]
        try:
            sys = epde_search_obj.get_equations_by_complexity(compl)
            res = sys[0]
        except IndexError:
            compl[0] += 0.5
            attempt += 1
            continue
        equation_obtained = True
    return epde_search_obj, res

if __name__ == "__main__":
    as_system = False
    t, x_stacked = prepare_data(steps_num=640)
    t_max = 320
    x, y = x_stacked[:, 0], x_stacked[:, 1]
    t_train, t_test = t[:t_max], t[:t_max]
    x_train, x_test = x[:t_max], x[:t_max]
    y_train, y_test = y[:t_max], y[:t_max]
    
    aux_preprocessor_type = 'poly'
    aux_preprocessor_kwargs = {'use_smoothing' : False,
                               'include_time' : True}
    # aux_preprocessor_kwargs = {'epochs_max' : 10000}
    
    setup = PreprocessorSetup()
    builder = ConcretePrepBuilder()
    setup.builder = builder    
    
    if aux_preprocessor_type == 'ANN':
        setup.build_ANN_preprocessing(**aux_preprocessor_kwargs)
    elif aux_preprocessor_type == 'poly':
        setup.build_poly_diff_preprocessing(**aux_preprocessor_kwargs)
    elif aux_preprocessor_type == 'spectral':
        setup.build_spectral_preprocessing(**aux_preprocessor_kwargs)
    else:
        raise NotImplementedError('Incorrect default preprocessor type. Only ANN or poly are allowed.')
    aux_preprocessor_pipeline = setup.builder.prep_pipeline

    if 'max_order' not in aux_preprocessor_pipeline.deriv_calculator_kwargs.keys():
        aux_preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None
        
#    max_order = (1,)
#    deriv_names, deriv_orders = define_derivatives('u', dimensionality=u.ndim,
#                                                   max_order=max_order)
      
    run_epde = True
    run_sindy = True
    pool = None
    pred = False

    exps = {}
    magnitudes = [0, ]#0.5*1e-2, 1.*1e-2, 2.5*1e-2,]# 5.*1e-2,]# 10.*1e-2, 15.*1e-2] # # 
    for magnitude in magnitudes:
        x_train_n = x_train + np.random.normal(scale = np.abs(magnitude*x_train), 
                                              size = x_train.shape)
        _, dx_train_n = aux_preprocessor_pipeline.run(x_train_n, grid=[t_test,],
                                                            max_order=(1,))
        dx_train_n = dx_train_n.reshape(-1)               
        plt.plot(t_train, x_train, color = 'k', label = 'Initial data')
        plt.plot(t_train, x_train_n, color = 'r', label = 'Corrupted data')
        plt.grid()
        plt.legend()
        plt.show()

        errs_epde = []
        models_epde = []
        calc_epde = []        
        if run_epde:
            test_launches = 1

        
            for idx in range(test_launches):
                t1 = time.time()
                if as_system:
                    epde_search_obj = epde_discovery_as_system(t_train, x_train_n, y_train, True)
                else:
                    epde_search_obj, sys = epde_discovery_as_ode(t_train, x_train_n, y_train, True)
                t2 = time.time()

                def get_ode_bop(key, grid_loc, value, var = 0, term = [None]):
                    bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var)
                    bop_grd_np = np.array([[grid_loc,]])
                    bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
                    bop.values = torch.from_numpy(np.array([[value,]])).float()
                    return bop

                if pred:
                    plt.plot(t_train[50:-50], x_train[50:-50], color = 'k', label = "x, input")
                    plt.plot(t_train[50:-50], epde_search_obj.cache[1].get(('u', (1.0,)))[50:-50], 
                            color = 'r', label = "x, calculated")
                    plt.grid()
                    plt.legend()
                    plt.show()

                    plt.plot(t_train[50:-50], y_train[50:-50], color = 'k', label = "x', input")
                    plt.plot(t_train[50:-50], epde_search_obj.saved_derivaties['u'][50:-50, 0], 
                            color = 'r', label = "x', calculated")
                    plt.grid()
                    plt.legend()
                    plt.show()
                    
                    
                    bop_u = get_ode_bop('u', t_test[0], x_test[0], term = [None])
                    bop_dudt = get_ode_bop('dudt', t_test[0], y_test[0], term = [0])
                    pred_u_v = epde_search_obj.predict(system=sys, boundary_conditions=[bop_u(), bop_dudt()], 
                                                        grid = [t_test,], strategy='autograd')
                    pred_u_v = pred_u_v.reshape(x_test.shape)
                    
                    
                    _, pred_derivatives = aux_preprocessor_pipeline.run(pred_u_v, grid=[t_test,],
                                                                        max_order=(1,))                
                    plt.figure(figsize=(11, 6))
                    plt.plot(t_test, x_test, '+', color = 'b', label = 'x, test data')
                    plt.plot(t_test, y_test, '*', color = 'r', label = "x', test data")
                    plt.plot(t_test, pred_u_v, color = 'b', label='x, solution')
                    plt.plot(t_test, pred_derivatives[:, 0], color = 'r', label="x', solution")
                    plt.xlabel('Time')
                    plt.ylabel('x')
                    plt.grid()
                    plt.legend(loc='upper right')
                    plt.savefig(f'/home/maslyaev/epde/EPDE_main/projects/wSINDy/figs/van_der_pol_ANN_{magnitude}_attempt_{idx}.png', dpi = 300)
                    plt.show()            
                    
                    plt.figure(figsize=(7, 6))
                    plt.plot(pred_u_v, pred_derivatives[:, 0], color = 'r')
                    plt.scatter(x_test, y_test, s = 3, color = 'k')
                    plt.xlabel("x")
                    plt.ylabel("x'")
                    plt.savefig(f'/home/maslyaev/epde/EPDE_main/projects/wSINDy/figs/van_der_pol_ANN_{magnitude}_attempt_{idx}_traj.png', dpi = 300)
                    plt.show()            
                else:
                    pred_u_v = 0

                    
                models_epde.append(epde_search_obj)
                errs_epde.append(np.mean(np.abs(x_test - pred_u_v)))
                calc_epde.append(pred_u_v)            
                try:
                    logger.add_log(key = f'Van_der_Pol_noise_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('epde', magnitude), 
                                error_pred = np.mean(np.abs(x_test - pred_u_v)), time = t2 - t1)
                except NameError:
                    logger = Logger(name = 'logs/Van_der_Pol_time_check.json', referential_equation = '-0.2 * du/dx1{power: 1.0} * u{power: 2.0} + 0.2 * du/dx1{power: 1.0} + -1.000 * u{power: 1.0} + 0.0 * u{power: 1.0} * d^2u/dx1^2{power: 2.0} + 0.0 = d^2u/dx1^2{power: 1.0}', 
                                    pool = epde_search_obj.pool)
                    logger.add_log(key = f'Van_der_Pol_noise_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('epde', magnitude),
                                error_pred = np.mean(np.abs(x_test - pred_u_v)), time = t2 - t1)
            
        errs_SINDy = []
        models_SINDy = []
        calc_SINDy = []
        if run_sindy:            
            test_launches = 1
            for idx in range(test_launches):             
                '''
                Basic SINDy
                '''
                model_quality = np.inf; model_container = (None, None, None)
                for sparsity_thr in [50.,]: # 7., 12., ]: # 1.2, 1.5,, 1.5, 2., 2.5]: # 0.05, 0.1,  0.2, 0.5, 1.*1e-2
                    pool = get_epde_pool(t_train, x_train_n, dx_train_n)

                    t1 = time.time()                       
                    model_base = sindy_discovery(t_train, x_train_n, dx_train_n, sparsity=sparsity_thr)
                    t2 = time.time()
                    
                    if pred:
                        print('Initial conditions', np.array([x_test[0], y_test[0]]))
                        eq_translated = translate_sindy_eq(model_base.equations())
                        sys = translate_equation({'u': eq_translated[0],
                                                'v': eq_translated[1]}, pool)                    
                        print(sys.text_form)
                        # try:
                        pred_u_v = model_base.simulate(np.array([x_test[0], y_test[0]]), t_test)
                        
                        plt.plot(t_test, x_test, '+', label = 'preys_odeint')
                        plt.plot(t_test, y_test, '*', label = "predators_odeint")
                        plt.plot(t_test, pred_u_v[:, 0], color = 'b', label='preys_NN')
                        plt.plot(t_test, pred_u_v[:, 1], color = 'r', label='predators_NN')
                        plt.xlabel('Time t, [days]')
                        plt.ylabel('Population')
                        plt.grid()
                        plt.legend(loc='upper right')
                        plt.title(f'Basic SINDy {magnitude}')
                        plt.show()
                        err_u, err_v = np.mean(np.abs(x_test - pred_u_v[:, 0])), np.mean(np.abs(y_test - pred_u_v[:, 1]))
                    else:
                        pred_u_v = 0
                        err_u, err_v = 0, 0
                # except:
                    #     # errs_SINDy.append((np.inf, np.inf))
                    #     # calc_SINDy.append(np.zeros((x_test.size, 2)))
                    #     err_u, err_v = np.inf, np.inf
                    #     pred_u_v = np.zeros((t_test.size, 2))
                    if err_u + err_v < model_quality:
                        model_quality = err_u + err_v
                        model_container = (model_base, (err_u, err_v), pred_u_v)
    
                models_SINDy.append(model_container[0])
                errs_SINDy.append(model_container[1])
                calc_SINDy.append(model_container[2])                
                print('Discovered by SINDy:', sys.text_form)
                # print('Discovered by SINDy in sindy form:', model_base)
                
                try:
                    logger.add_log(key = f'VdP_SINDy_noise_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('sindy', magnitude), 
                                   error_pred = (err_u, err_v), time = t2 - t1)
                except NameError:
                    logger = Logger(name = 'logs/Van_der_Pol_time_check.json', referential_equation = {'u' : '20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} + 0.0 = du/dx1{power: 1.0}',
                                                                                                   'v' : '-20.0 * v{power: 1.0} + 20.0 * u{power: 1.0} * v{power: 1.0} + 0.0 = dv/dx1{power: 1.0}'}, 
                                    pool = pool)
                    logger.add_log(key = f'VdP_noise_{magnitude}_attempt_{idx}', entry = sys, aggregation_key = ('sindy', magnitude), 
                                   error_pred = (err_u, err_v), time = t2 - t1) 
        exps[magnitude] = {'epde' : (models_epde, errs_epde, calc_epde),
                           'sindy' : (models_SINDy, errs_SINDy, calc_SINDy)}
                    
    logger.dump()