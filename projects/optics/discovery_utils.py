import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CustomEvaluator, CustomTokens, GridTokens, CacheStoredTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement


def get_polynomial_family(tensor, order, token_type = 'polynomials'):
    '''
    Get family of tokens for polynomials of orders from second up to order argument.
    '''
    assert order > 1
    labels = [f'p^{idx+1}' for idx in range(1, order)]
    tensors = {label : tensor ** (idx + 2) for idx, label in enumerate(labels)}
    return CacheStoredTokens(token_type = token_type,
                                token_labels = labels,
                                token_tensors = tensors,
                                params_ranges = {'power' : (1, 1)},
                                params_equality_ranges = None)


def epde_discovery(t, x, boundary=0,use_ann = False, derivs=None):
    dimensionality = x.ndim - 1
    

    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = boundary,
                                           coordinate_tensors = [t,])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 50000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1, 'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                     # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 12
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=100)

    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.65, 0.35]}

    custom_grid_tokens = GridTokens(dimensionality = dimensionality)

    polynomial_tokens=get_polynomial_family(x, 4, token_type = 'polynomials')
    
    # custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power']) 
    # custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
    #                                            eval_fun_params_labels = ['dim', 'power'], 
    #                                            use_factors_grids = True)    

    # grid_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality)}
    
    # custom_grid_tokens = CustomTokens(token_type = 'grid', # Выбираем название для семейства токенов - обратных функций.
    #                                   token_labels = ['1/x_{dim}',], # Задаём названия токенов семейства в формате python-list'a.
    #                                                                  # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
    #                                   evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
    #                                   params_ranges = grid_params_ranges, # Используем заявленные диапазоны параметров
    #                                   params_equality_ranges = None)    
    if derivs is None:
        epde_search_obj.fit(data=[x,], variable_names=['u',], max_deriv_order=(2,),
                            equation_terms_max_number=7, data_fun_pow = 1, 
                            additional_tokens=[polynomial_tokens], 
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-12, 1e-4))
    else:
        epde_search_obj.fit(data=[x,], variable_names=['u',], max_deriv_order=(2,),
                    derivs=[derivs,],
                    equation_terms_max_number=7, data_fun_pow = 1, 
                    additional_tokens=[polynomial_tokens], 
                    equation_factors_max_number=factors_max_number,
                    eq_sparsity_interval=(1e-12, 1e-4))

    epde_search_obj.equation_search_results(only_print = True, num = 1)
    
    # syss = epde_search_obj.equation_search_results(only_print = False, num = 1) 
    '''
    Having insight about the initial ODE structure, we are extracting the equation with complexity of 5
    
    In other cases, you should call sys.equation_search_results(only_print = True),
    where the algorithm presents Pareto frontier of optimal equations.
    '''
    #sys = epde_search_obj.get_equations_by_complexity(4)[0]
    return epde_search_obj

# from tedeous.input_preprocessing import Equation
# from tedeous.solver import Solver, grid_format_prepare
# from tedeous.metrics import Solution
# from tedeous.device import solver_device,check_device
import torch
import numpy as np
import os

# def solver_solution(eq,rv,m_grid):
    
#     # solver_device('gpu')

#     coord_list = [m_grid]

#     coord_list=torch.tensor(coord_list)
#     grid=coord_list.reshape(-1,1).float()

#     # point t=0
#     bnd1 = torch.from_numpy(np.array([[float(m_grid[0])]], dtype=np.float64)).float()
    
    
#     #  So u(0)=-1/2
#     bndval1 = torch.from_numpy(np.array([[float(rv[0])]], dtype=np.float64))

#     # point t=0
#     bnd3 = torch.from_numpy(np.array([[float(m_grid[1])]], dtype=np.float64)).float()
    
    
#     #  So u(0)=-1/2
#     bndval3 = torch.from_numpy(np.array([[float(rv[1])]], dtype=np.float64))


#     # point t=0
#     bnd2 = torch.from_numpy(np.array([[float(m_grid[-1])]], dtype=np.float64)).float()
    
    
#     #  So u(0)=-1/2
#     bndval2 = torch.from_numpy(np.array([[float(rv[-1])]], dtype=np.float64))    



#         # Putting all bconds together
#     bconds = [[bnd1, bndval1, 'dirichlet'],
#                 [bnd2, bndval2, 'dirichlet'],
#                 #[bnd3, bndval3, 'dirichlet']
#                 ]

#     equation = Equation(grid, eq, bconds).set_strategy('autograd')

#     img_dir=os.path.join(os.path.dirname( __file__ ), 'optics_intermediate')

#     model = torch.nn.Sequential(
#     torch.nn.Linear(1, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 1)
#     )

#     model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1000,verbose=1, learning_rate=1e-4,
#                                             eps=1e-6, tmin=1000, tmax=1e6,use_cache=True,cache_verbose=True,
#                                             save_always=True,print_every=None,model_randomize_parameter=1e-4,
#                                             optimizer_mode='Adam',no_improvement_patience=1000,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)

#     return model