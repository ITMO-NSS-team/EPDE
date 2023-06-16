import numpy as np
import pandas as pd

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

from epde.interface.logger import Logger
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, CustomEvaluator, CustomTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

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

def epde_discovery(x, t, u, derivs, use_ann = False): #(grids, data, use_ann = False):
    grids = np.meshgrid(t, x, indexing = 'ij')
    print(u.shape, grids[0].shape, grids[1].shape)
    multiobjective_mode = True
    dimensionality = u.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True
                                          preprocessor_kwargs={'epochs_max' : 2})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=65)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize,
                                                   training_epochs=65)
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
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.6, 0.4]}
    
    opt_val = 1e-1
    bounds = (1e-9, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    print(u.shape, grids[0].shape)
    epde_search_obj.fit(data=u, variable_names=['u',], max_deriv_order=(1, 3), derivs = [derivs,],
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[custom_trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    
    equation_obtained = False; compl = [5,]; attempt = 0
    
    iterations = 4
    while not equation_obtained:
        if attempt < iterations:
            try:
                sys = epde_search_obj.get_equations_by_complexity(compl)
                res = sys[0]
            except IndexError:
                compl[0] += 1
                attempt += 1
                continue
        else:
            res = epde_search_obj.equations(only_print = False)[0][0]
        equation_obtained = True
    
    return epde_search_obj, res

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
            
def sindy_provided_l0(grids, u):
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
    model.fit(u, t=dt)
    model.print()    
    return model

if __name__ == '__main__':
    path = '/home/maslyaev/epde/EPDE_main/projects/wSINDy/data/KdV/'

    try:
        df = pd.read_csv(f'{path}KdV_sln_100.csv', header=None)
        dddx = pd.read_csv(f'{path}ddd_x_100.csv', header=None)
        ddx = pd.read_csv(f'{path}dd_x_100.csv', header=None)
        dx = pd.read_csv(f'{path}d_x_100.csv', header=None)
        dt = pd.read_csv(f'{path}d_t_100.csv', header=None)
    except (FileNotFoundError, OSError):
        df = pd.read_csv('/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/KdV_sln_100.csv', header=None)
        dddx = pd.read_csv('/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/ddd_x_100.csv', header=None)
        ddx = pd.read_csv('/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/dd_x_100.csv', header=None)
        dx = pd.read_csv('/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/d_x_100.csv', header=None)
        dt = pd.read_csv('/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/d_t_100.csv', header=None)

    def train_test_split(tensor, time_index):
        return tensor[:time_index, ...], tensor[time_index:, ...]

    train_max = 50

    u = df.values
    u = np.transpose(u)
    u_train, u_test = train_test_split(u, train_max)

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')

    
    grid_t_train, grid_t_test = train_test_split(grids[0], train_max)
    grid_x_train, grid_x_test = train_test_split(grids[1], train_max)
    
    grids_training = (grid_t_train, grid_x_train)
    grids_test = (grid_t_test, grid_x_test)    

    t_train, t_test = train_test_split(t, train_max)
    data_train, data_test = train_test_split(u, train_max)
    
    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    ddd_x_train, ddd_x_test = train_test_split(ddd_x, train_max)
    
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    dd_x_train, dd_x_test = train_test_split(dd_x, train_max)
    
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_x_train, d_x_test = train_test_split(d_x, train_max)    
    
    d_t = dt.values
    d_t = np.transpose(d_t)
    d_t_train, d_t_test = train_test_split(d_t, train_max)

    derivs = np.zeros(shape=(u_train.shape[0], u_train.shape[1], 4))
    derivs[:, :, 0] = d_t_train
    derivs[:, :, 1] = d_x_train
    derivs[:, :, 2] = dd_x_train
    derivs[:, :, 3] = ddd_x_train
    derivs = derivs.reshape((-1, 4))   
    print(derivs.shape)
    '''
    EPDE side
    '''

    exps = {}
    test_launches = 10
    magnitudes = [0,]# 1.*1e-2, 2.5*1e-2, 5.*1e-2, 1.*1e-1] # 
    for magnitude in magnitudes:
        data_train_n = data_train + np.random.normal(scale = magnitude * np.abs(data_train), size = data_train.shape)
        
        Heatmap(data_train_n, title = 'Data')
        
        errs_epde = []
        models_epde = []
        calc_epde = []
        for idx in range(test_launches):
            epde_search_obj, sys = epde_discovery(x, t_train, data_train_n, derivs)
            print(sys.text_form)
            models_epde.append(sys)
        exps['kek'] = models_epde
        #     bnd_t = torch.cartesian_prod(torch.from_numpy(np.array([t[train_max + 1]], dtype=np.float64)),
        #                                   torch.from_numpy(x)).float()
            
        #     bop_1 = BOPElement(axis = 0, key = 'u_t', term = [None], power = 1, var = 0)
        #     bop_1.set_grid(bnd_t)
        #     bop_1.values = torch.from_numpy(data_test[0, ...]).float()
            
        #     t_der = epde_search_obj.saved_derivaties['u'][..., 0].reshape(grids_training[0].shape)
        #     bop_2 = BOPElement(axis = 0, key = 'dudt', term = [0], power = 1, var = 0)
        #     bop_2.set_grid(bnd_t)
        #     bop_2.values = torch.from_numpy(t_der[-1, ...]).float()
            
        #     bnd_x1 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
        #                                   torch.from_numpy(np.array([x[0]], dtype=np.float64))).float()
        #     bnd_x2 = torch.cartesian_prod(torch.from_numpy(t[train_max:]),
        #                                   torch.from_numpy(np.array([x[-1]], dtype=np.float64))).float()            
            
        #     bop_3 = BOPElement(axis = 1, key = 'u_x1', term = [None], power = 1, var = 0)
        #     bop_3.set_grid(bnd_x1)
        #     bop_3.values = torch.from_numpy(data_test[..., 0]).float()            
    
        #     bop_4 = BOPElement(axis = 1, key = 'u_x2', term = [None], power = 1, var = 0)
        #     bop_4.set_grid(bnd_x2)
        #     bop_4.values = torch.from_numpy(data_test[..., -1]).float()            
            
        #     # bop_grd_np = np.array([[,]])
            
        #     # bop = get_ode_bop('u', 0, t_test[0], x_test[0])
        #     # bop_y = get_ode_bop('v', 1, t_test[0], y_test[0])
            
            
        #     pred_u_v = epde_search_obj.predict(system=sys, boundary_conditions=[bop_1(), bop_2(), bop_3(), bop_4()], 
        #                                         grid = grids_test, strategy='NN')
        #     pred_u_v = pred_u_v.reshape(data_test.shape)
        #     models_epde.append(epde_search_obj)
        #     errs_epde.append(np.mean(np.abs(data_test - pred_u_v)))
        #     calc_epde.append(pred_u_v)
            
        # model_base = sindy_provided_l0(grids_training, data_train_n)    
        # exps[magnitude] = {'epde': (models_epde, errs_epde, calc_epde),
        #                    'SINDy': (model_base,)}