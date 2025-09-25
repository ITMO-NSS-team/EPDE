#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:50:09 2021

@author: maslyaev
"""



import numpy as np
import matplotlib
import epde.interface.interface as epde_alg
import os
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, ExternalDerivativesTokens

import matplotlib.pyplot as plt
import torch
import epde.solver

# from TEDEouS import solver


# from TEDEouS import config 


os.chdir(os.path.dirname( __file__ ))

import pandas as pd




def equation_fit(grid,data):
    """
    Defines the parameter value ranges within which functions are considered identical.
    
            This is crucial for constructing unique equation structures and their components. For instance, in the context of equation discovery,
            the difference between `sin(3.135 * x)` and `sin(3.145 * x)` might be deemed negligible, treating them as equivalent. This allows to reduce number of similar equations.
    
            Args:
                grid (np.ndarray): The grid on which the data is defined.
                data (np.ndarray): The data to fit the equation to.
    
            Returns:
                list: A list of tuples, where each tuple contains the solver form of an equation and its boundary conditions.
    """
       
        dimensionality = data.ndim # - 1
    
        epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 200, dimensionality=dimensionality) #verbose_params={'show_moeadd_epochs' : True}
        
        #custom_trigonometric_eval_fun =  {'cos' : lambda *grids, **kwargs: np.cos(2*np.pi*f0*grids[int(kwargs['dim'])]) ** kwargs['power'], 
        #           'sin' : lambda *grids, **kwargs: np.sin(2*np.pi*f0*grids[int(kwargs['dim'])]) ** kwargs['power']}
    
        '''
        --------------------------------------------------------------------------------------------------------------------------------
        Задаём объект для оценки значений токенов в эволюционном алгоритме. Аргументы - заданная выше функция/функции оценки значений 
        токенов и лист с названиями параметров. 
        '''
        #custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels = ['dim', 'power'])
        
        '''
        --------------------------------------------------------------------------------------------------------------------------------
        Задам через python-словарь диапазоны, в рамках которых могут браться параметры функций оценки токенов.
        
        Ключи должны быть в формате str и соотноситься с аргументами лямда-функций для оценки значения токенов. Так, для введённой
        выше функции для оценки значений тригонометрических функций, необходимы значения частоты, степени функции и измерения сетки, по 
        которому берётся аргумент с ключами соответственно 'freq', 'power' и 'dim'. Значения, соответствующие этим ключам, должны быть
        границы, в пределах которых будут искаться значения параметров функции при оптимизации, заданные в формате python-tuple из 
        2-ух элементов: левой и правой границы.
        Целочисленное значение границ соответствует дискретным значеням (например, при 'power' : (1, 3), 
        будут браться степени со значениями 1, 2 и 3); при действительных значениях (типа float) значения параметров 
        берутся из равномерного распределения с границами из значения словаря. Так, например, при значении 'freq' : (1., 3.), 
        значения будут выбираться из np.random.uniform(low = 1., high = 3.), например, 2.7183... 
        '''
        #trig_params_ranges = {'power' : (1, 1), 'dim' : (0, 0)} 
        
        '''
        --------------------------------------------------------------------------------------------------------------------------------
        Далее необходимо определить различия в значениях параметров, в пределах которых функции считаются идентичными, чтобы строить 
        уникальные структуры уравнений и слагаемых в них. Например, для эволюционного алгоритма можно считать, что различия между 
        sin(3.135 * x) и sin(3.145 * x) незначительны и их можно считать равными. 
        
        Задание значений выполняется следующим образом: ключ словаря - название параметра, значение - максимальный интервал, при котором токены
        счиатются идентичными.
        
        По умолчанию, для дискретных параметров равенство выполняется только при полном соответствии, а для действительно-значных аргументов
        равенство выполняется при разнице меньше, чем 0.05 * (max_param_value - min_param_value).
        '''
        #trig_params_equal_ranges = {}
        
        #custom_trig_tokens = Custom_tokens(token_type = 'trigonometric', # Выбираем название для семейства токенов.
        #                                   token_labels = ['sin', 'cos'], # Задаём названия токенов семейства в формате python-list'a.
        #                                   meaningful=True,              # Названия должны соответствовать тем, что были заданы в словаре с лямбда-ф-циями.
        #                                   evaluator = custom_trig_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
        #                                   params_ranges = trig_params_ranges, # Используем заявленные диапазоны параметров
        #                                   params_equality_ranges = trig_params_equal_ranges) # Используем заявленные диапазоны "равенства" параметров
            
            
        ExternalDerivativesTokens
        
        epde_search_obj.set_moeadd_params(population_size=20)
    
        
    
        #epde_search_obj.fit(data=phi, max_deriv_order=(2,), boundary=(0,), equation_terms_max_number=7, data_fun_pow = 2,
         #                   equation_factors_max_number=2, deriv_method='poly', eq_sparsity_interval=(1e-7, 1000),
         #                   deriv_method_kwargs={'smooth': False, 'grid': [t, ]}, coordinate_tensors=[t, ],
         #                   additional_tokens = [custom_trig_tokens])

        epde_search_obj.fit(data=data, max_deriv_order=(2,), boundary=(0,), equation_terms_max_number=5, data_fun_pow = 2,
                    equation_factors_max_number=2, deriv_method='poly', eq_sparsity_interval=(1e-7, 1000),
                    deriv_method_kwargs={'smooth': False, 'grid': [grid, ]}, coordinate_tensors=[grid, ])
    
        res = epde_search_obj.equation_search_results(only_print = False, level_num = 1) # showing the Pareto-optimal set of discovered equations 
        solver_inp = []
    
        for eq in res[0]:
            solver_inp.append((eq.structure[0].solver_form(), eq.structure[0].boundary_conditions()))
    
        epde_search_obj.equation_search_results(only_print = True, level_num = 1)
        return solver_inp


if __name__ == '__main__':
    for nruns in range(1):
        # file_number='8202026016'
        # file_number='8202026032'
        # file_number='8202026034'
        file_number='8202026000' 
        
        exp_name = "electroduce"
        # df=pd.read_csv('8202026000_halfper_0.csv',index_col=0)
        # phi = df['deg'].values
        # tick_title = "time, ms"
        
        # t = df['t_arc'].values
        
        
        # df=pd.read_csv('prepared_data/mean_arc_{}.csv'.format(file_number),index_col=0)
        # phi = df['mean_arc'].values
        
        # t = df['mean_arc_time'].values
        
        df=pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'prepared_data/median_arc_{}.csv'.format(file_number))),index_col=0)
        phi = df['median_arc'].values
        phi=phi*np.pi/180
        print(phi)
        t = df['median_arc_time'].values.astype(np.float64)
        t*=0.25*1e-3
        
        print(t)
        raise NotImplementedError()
        f0=50
        tick_title = "time, ms"
        matplotlib.rcParams.update({'font.size': 20})
        plt.rcParams["figure.figsize"] = [14, 7]
    
        plt.plot(t,phi, 'bo')
        plt.xlabel(tick_title)
        plt.ylabel("angle, deg")
    
        solver_inp=equation_fit(t,phi)

        # print(solver_inp)
        
    
        #print('bnds={}'.format(solver_inp[0][1]))
        models = []
        
        
        for eqn_n, s_inp in enumerate(solver_inp): 
            
            coord_list = [t]
            
            cfg=config.Config(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'ODE_prediction.json')))
            
            #cfg.set_parameter("Matrix.cache_model",model_arch)

            cfg.set_parameter("Cache.use_cache",True)

            cfg.set_parameter('NN.h',float(t[1]-t[0]))

            model = solver.optimization_solver(coord_list, None, s_inp[0], s_inp[1], cfg,mode='NN')
            
            models.append(model)

        import os
        if not(os.path.isdir('results')):
            os.mkdir('results')
         
        
        
        for n in range(len(s_inp[0])):
            plt.figure()
            
            print(res[0][n].structure[0].text_form)
            
            with open('results/eqn_{}_{}_run_{}.txt'.format(file_number, n+1,nruns), 'w') as the_file:
                the_file.write(res[0][n].structure[0].text_form)
            
            plt.title('Eqn #{}'.format(n+1))
            
            plt.plot(t,phi)
            plt.xlabel(tick_title)
            plt.ylabel("angle, deg")
        
        
            if callable(models[1]):
                plt.plot(t,models[n](torch.from_numpy(t).reshape(-1,1).float()).detach().numpy().reshape(-1))
            else:
                plt.plot(t,models[n].detach().numpy().reshape(-1))
            plt.xlabel(tick_title)
            plt.ylabel("angle, deg")
        
            plt.savefig('results/solution_{}_eqn_{}_run_{}.png'.format(file_number,n+1,nruns))

    
    