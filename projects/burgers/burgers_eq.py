# import pandas as pd
# import numpy as np
# import torch
# import winsound

# from epde_wave import epde_equation
import time
import numpy as np
import scipy.io
import pandas as pd
import epde.interface.interface as epde_alg
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CacheStoredTokens, CustomTokens, TrigonometricTokens


def out_formatting(string):
    """
    Formats a string to improve readability of discovered equations by replacing symbolic representations and rounding numerical values.
    
        This function specifically targets the output format of the equation discovery process,
        making it easier to interpret the resulting mathematical expressions. It performs
        substitutions for common derivative notations and ensures that floating-point numbers
        are displayed with a reasonable level of precision.
    
        Args:
            string: The input string representing the discovered equation.
    
        Returns:
            str: The formatted string with replaced patterns and rounded numbers,
                 suitable for human interpretation.
    """
    string1 = str.split(string, '\n')
    string = string1[0]
    string = string.replace("u{power: 1.0}", "u")
    string = string.replace("d^3u/dx2^3{power: 1.0}", "u_{xxx}")
    string = string.replace("d^2u/dx2^2{power: 1.0}", "u_{xx}")
    string = string.replace("d^2u/dx1^2{power: 1.0}", "u_{tt}")
    string = string.replace("du/dx1{power: 1.0}", "u_{t}")
    string = string.replace("du/dx2{power: 1.0}", "u_{x}")
    string = string.replace("cos(t)sin(x){power: 1.0}", "cos(t)sin(x)")
    string = string.replace("cos(t){power: 1.0}", "cos(t)")
    string = string.replace("sin(x){power: 1.0}", "sin(x)")

    ls = string.split()
    string2 = ''
    for elem in ls:
        if elem.find('.') > -1:
            elem = round(float(elem), 4)
        elif elem == '+' or elem == '=':
            elem = ' ' + elem + ' '
        string2 += str(elem)
    return string2




if __name__ == '__main__':

    # initial params before fit-EPDE (global params)
    grid_res = 70
    title = 'wave_equation'  # name of the problem/equation
    test_iter_limit = 1 # how many times to launch algorithm (one time - 2-3 equations)
    # Load data
    # df = pd.read_csv(f'wolfram_sln/wave_sln_{grid_res}.csv', header=None)

    ################ передача данных из матлаба
    # t = pd.read_csv('t.csv', header=None)
    # x = pd.read_csv('x.csv', header=None)
    # t = t.to_numpy()
    # x = x.to_numpy()
    # t = t.reshape(t.size)
    # x = x.reshape(x.size)
    # df = pd.read_csv('U_exact.csv', header=None)

    path = "projects/burgers/"
    mat = scipy.io.loadmat(f'{path}burgers.mat')
    u = mat['u']
    u = np.transpose(u)
    t = np.ravel(mat['t'])
    x = np.ravel(mat['x'])
    ################ end

    # u = df.values
    # u = np.transpose(u)

    # ddd_x = dddx.values
    # ddd_x = np.transpose(ddd_x)
    # dd_x = ddx.values
    # dd_x = np.transpose(dd_x)
    # d_x = dx.values
    # d_x = np.transpose(d_x)
    # d_t = dt.values
    # d_t = np.transpose(d_t)


    # derivs = np.zeros(shape=(u.shape[0],u.shape[1],4))
    # derivs[:, :, 0] = d_t
    # derivs[:, :, 1] = d_x
    # derivs[:, :, 2] = dd_x
    # derivs[:, :, 3] = ddd_x
    # derivs = [d_t, d_x, dd_x, ddd_x]

    # t = np.linspace(0, 1, u.shape[0])
    # x = np.linspace(0, 1, u.shape[1])

    # t = np.linspace(0, 1, u.shape[0])
    # x = np.linspace(0, 10, u.shape[1])
    print(len(t))
    print(len(x))

    boundary = 50
    dimensionality = u.ndim #- 1
    grids = np.meshgrid(t, x, indexing='ij')

    for test_idx in np.arange(test_iter_limit):
        epde_search_obj = epde_alg.epde_search(use_solver=False, boundary=boundary,
                                               dimensionality=dimensionality, coordinate_tensors=grids)

        ################# eval_fun вариант 1
        # custom_trigonometric_eval_fun = {
        #     'cos': lambda *grids, **kwargs: np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
        #     'sin': lambda *grids, **kwargs: np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}
        # custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
        #                                         eval_fun_params_labels=['freq', 'dim', 'power'])
        # trig_params_ranges = {'power': (1, 1), 'freq': (0.95, 1.05), 'dim': (0, dimensionality-1)}
        # trig_params_equal_ranges = {'freq': 0.05}


        ################# eval_fun вариант 2
        # custom_trigonometric_eval_fun = {
        #     'cos(t)': lambda *grids, **kwargs: np.cos(grids[0]) ** kwargs['power'],
        #     'sin(x)': lambda *grids, **kwargs: np.sin(grids[1]) ** kwargs['power']}
        # custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
        #                                         eval_fun_params_labels=['power'])
        #
        # trig_params_ranges = {'power': (1, 1)}
        # trig_params_equal_ranges = {}


        ################# eval_fun вариант 3
        custom_trigonometric_eval_fun = {
            'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
        custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                                eval_fun_params_labels=['power'])

        trig_params_ranges = {'power': (1, 1)}
        trig_params_equal_ranges = {}

        trig_tokens = TrigonometricTokens(freq=(1 - 1e-7, 1 + 1e-7), dimensionality=dimensionality-1)
        # trig_tokens = TrigonometricTokens(dimensionality=dimensionality - 1)

        custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                           token_labels=['cos(t)sin(x)'],
                                           evaluator=custom_trig_evaluator,
                                           params_ranges=trig_params_ranges,
                                           params_equality_ranges=trig_params_equal_ranges,
                                           meaningful=True, unique_token_type=False)




        epde_search_obj.set_memory_properties(u, mem_for_cache_frac=10)
        epde_search_obj.set_moeadd_params(population_size=2, training_epochs=100)

        custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                 # boundary=boundary,
                                                 token_labels=['t', 'x'],
                                                 token_tensors={'t': grids[0], 'x': grids[1]},
                                                 params_ranges={'power': (1, 1)},
                                                 params_equality_ranges=None,
                                                 unique_token_type=False)

        # custom_grid_tokens = CacheStoredTokens(token_type='grid',
        #                                          # boundary=boundary,
        #                                          token_labels=['t',],
        #                                          token_tensors={'t': grids[0]},
        #                                          params_ranges={'power': (1, 3)},
        #                                          params_equality_ranges=None,
        #                                          unique_token_type=False,
        #                                          unique_specific_token = True,
        #                                          meaningful=False)

        '''
        Method epde_search.fit() is used to initiate the equation search.
        '''
        start = time.time()

        epde_search_obj.fit(data=u, max_deriv_order=(1, 1),
                            equation_terms_max_number=2, equation_factors_max_number=2,
                            eq_sparsity_interval=(1e-08, 1e-06),
                            #derivs=[derivs],
                            # deriv_method_kwargs={'smooth': False, 'grid': grids},
                            # deriv_method_kwargs={'epochs_max': 20000},
                            # additional_tokens=[custom_grid_tokens, ],# custom_trig_tokens],trig_tokens   custom_grid_tokens
                            memory_for_cache=25, prune_domain=False, data_fun_pow=1,
                            # custom_cross_prob={('d^3u/dx2^3',): 0.1, ('du/dx2', 'u'): 0.1, ('cos(t)sin(x)', ): 0.1}
                            )



        # epde_search_obj.fit(data=u, max_deriv_order=(1, 3),
        #                     equation_terms_max_number=4, equation_factors_max_number=2,
        #                     coordinate_tensors=grids, eq_sparsity_interval=(1e-08, 1e-06),
        #                     deriv_method='ANN', derivs=[derivs],
        #                     # deriv_method_kwargs={'smooth': False, 'grid': grids},
        #                     deriv_method_kwargs={'epochs_max': 40000},
        #                     additional_tokens=[custom_trig_tokens, ],# custom_trig_tokens],trig_tokens   custom_grid_tokens
        #                     memory_for_cache=25, prune_domain=True, data_fun_pow=2,
        #                     custom_cross_prob={('d^3u/dx2^3',): 0.1, ('du/dx2', 'u'): 0.1, ('cos(t)sin(x)', ): 0.1}
        #                     )
                            #)custom_prob_terms={('d^3u/dx2^3',): 5, ('du/dx2', 'u'): 2, ('sin', 'cos'): 4},)
        #('cos(t)sin(x)',): 4,
        end = time.time()

        '''
        The results of the equation search have the following format: if we call method 
        .equation_search_results() with "only_print = True", the Pareto frontiers 
        of equations of varying complexities will be shown, as in the following example:

        If the method is called with the "only_print = False", the algorithm will return list 
        of Pareto frontiers with the desired equations.
        '''

        # epde_search_obj.equation_search_results(only_print=True, level_num=1)

        res = epde_search_obj.equation_search_results(only_print=True, level_num=2)
        # sys1 = res[0][1]
        # res_text = sys1.text_form()
        # print('\n\n\nTEXT FORM\n', res_text)

        time1 = end-start
        print('Overall time is:', time1 / 60.)
        # for idx in range(len(epde_search_obj.equations_pareto_frontier)):
        #     print('\n')
        #     print(f'{idx}-th non-dominated level')
        #     print('\n')
        #     [print(f'{out_formatting(solution.text_form)}\nwith objective function values of {solution.obj_fun} \n')
        #     # [print(f'{solution.text_form}\nwith objective function values of {solution.obj_fun} \n')
        #                                                 for solution in epde_search_obj.equations_pareto_frontier[idx]]


        # def equation_search_results(self, only_print: bool = True, level_num=1):
        #     if only_print:
        #         for idx in range(min(level_num, len(self.equations_pareto_frontier))):
        #             print('\n')
        #             print(f'{idx}-th non-dominated level')
        #             print('\n')
        #             [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')
        #              for solution in self.equations_pareto_frontier[idx]]
        #     else:
        #         return self.optimizer.pareto_levels.levels[:level_num]

        # print('Overall time is:', time1 / 60.)
    # winsound.PlaySound('C:\\Windows\\Media\\Windows Proximity Notification.wav', winsound.SND_NOWAIT)
    # winsound.PlaySound('C:\\Windows\\Media\\Windows Proximity Connection.wav', winsound.SND_LOOP)



