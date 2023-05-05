#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:53:27 2023

@author: maslyaev
"""

import numpy as np

import torch
import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import lotka

import pysindy as ps

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

SOLVER_STRATEGY = 'autograd'

def write_pareto(dict_of_exp):
    for key, item in dict_of_exp.items():
        test_key = str(key[0]).replace('.', '_') + '__' + str(key[1]).replace('.', '_')
        with open('/home/maslyaev/epde/EPDE_main/projects/hunter-prey/param_var/'+test_key+'.txt', 'w') as f:
            for iteration in range(len(item)):
                f.write(f'Iteration {iteration}\n\n')
                for ind in [pareto.text_form for pareto in item[iteration][0]]:
                    f.write(ind + '\n\n')

def epde_discovery(t, x, y, use_ann = False):
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
    factors_max_number = {'factors_num' : [1, 2, 3], 'probas' : [0.6, 0.3, 0.1]}
    
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow = 2, #additional_tokens=[trig_tokens,], 
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
    epde_search_obj.equation_search_results(only_print = True, num = 1)
    equation_obtained = False; compl = [4, 4]; attempt = 0
    
    
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
    

def sindy_discovery(t, x, y):
    poly_order = 5
    threshold = 0.05
    
    x_train = np.array([x, y]).T
    print(x_train.shape)
    
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
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

# def save_launch_res(filename):
#     if filename[0] == '\':

if __name__ == '__main__':
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    try:
        t_file = os.path.join(os.path.dirname( __file__ ), 'projects/hunter-prey/t_20.npy')
        t = np.load(t_file)
    except FileNotFoundError:
        t_file = '/home/maslyaev/epde/EPDE_main/projects/hunter-prey/t_20.npy'
        t = np.load(t_file)
    
    try:
        data_file =  os.path.join(os.path.dirname( __file__ ), 'projects/hunter-prey/data_20.npy')
        data = np.load(data_file)
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/hunter-prey/data_20.npy'
        data = np.load(data_file)
    
    large_data = False
    if large_data:
        t_max = 400; t_size_raw = 100; t_size_dense = 1000
        t_train = t[:t_max]; t_test_interval = t[t_max:t_max+t_size_raw] 
        t_test_interval_pred = np.linspace(t_test_interval[0], t_test_interval[-1], 
                                           num = t_size_dense)
    else:
        t_max = 150
        t_train = t[:t_max]; t_test = t[t_max:] 
        t_test_interval_pred = t_test
        
    x = data[:t_max, 0]; x_test = data[t_max:, 0]
    y = data[:t_max, 1]; y_test = data[t_max:, 1]

    magnitude = 5.*1e-2
    x_n = x + np.random.normal(scale = magnitude*x, size = x.shape)
    y_n = y + np.random.normal(scale = magnitude*y, size = y.shape)
    plt.plot(t_train, x_n)
    plt.plot(t_train, y_n)
    plt.show()
    epde = True
    
    if epde:
        test_launches = 1
        errs = []
        models = []
        for idx in range(test_launches):
            epde_search_obj, sys = epde_discovery(t_train, x_n, y_n, True)
            
            def get_ode_bop(key, var, grid_loc, value):
                bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
                bop_grd_np = np.array([[grid_loc,]])
                bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
                bop.values = torch.from_numpy(np.array([[value,]])).float()
                return bop
                
            
            bop_x = get_ode_bop('u', 0, t_test_interval_pred[0], x_test[0])
            bop_y = get_ode_bop('v', 1, t_test_interval_pred[0], y_test[0])
            
            
            pred_u_v = epde_search_obj.predict(system=sys, boundary_conditions=[bop_x(), bop_y()], 
                                                grid = [t_test_interval_pred,], strategy=SOLVER_STRATEGY)
            plt.plot(t_test_interval_pred, x_test, '+', label = 'preys_odeint')
            plt.plot(t_test_interval_pred, y_test, '*', label = "predators_odeint")
            plt.plot(t_test_interval_pred, pred_u_v[..., 0], color = 'b', label='preys_NN')
            plt.plot(t_test_interval_pred, pred_u_v[..., 1], color = 'r', label='predators_NN')
            plt.xlabel('Time t, [days]')
            plt.ylabel('Population')
            plt.grid()
            plt.legend(loc='upper right')
            plt.show()
            
            models.append(epde_search_obj)
            errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])), 
                         np.mean(np.abs(y_test - pred_u_v[:, 1]))))
    else:
        errs = []
        models = []
        test_launches = 10
        for idx in range(test_launches):
            model = sindy_discovery(t_train, x_n, y_n)
            print('Initial conditions', np.array([x_test[0], y_test[0]]))
            pred_u_v = model.simulate(np.array([x_test[0], y_test[0]]), t_test)
            models.append(model)
            
            plt.plot(t_test, x_test, '+', label = 'preys_odeint')
            plt.plot(t_test, y_test, '*', label = "predators_odeint")
            plt.plot(t_test, pred_u_v[:, 0], color = 'b', label='preys_NN')
            plt.plot(t_test, pred_u_v[:, 1], color = 'r', label='predators_NN')
            plt.xlabel('Time t, [days]')
            plt.ylabel('Population')
            plt.grid()
            plt.legend(loc='upper right')       
            plt.show()
            errs.append((np.mean(np.abs(x_test - pred_u_v[:, 0])),
                         np.mean(np.abs(y_test - pred_u_v[:, 1]))))
        