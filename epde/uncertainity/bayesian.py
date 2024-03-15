#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:47 2024

@author: maslyaev
"""
import os
import numpy as np
import pandas as pd
import pickle

from epde.interface.interface import EpdeSearch

# def learn_fit(data, grid, derivs = None):
#     self.

class BayesianApproach(object):
    def __init__(self):
        pass
        
    def _get_equations(self, cfg, title, data, grid, derives = None, ):
        if not (os.path.exists(f'data/{title}/epde_result')):
            os.mkdir(f'data/{title}/epde_result')
    
        if cfg.params["glob_epde"]["load_result"]:
            # Need to check the existence of the file or send the path
            return pd.read_csv(f'data/{title}/epde_result/output_main_{title}.csv', index_col='Unnamed: 0', sep='\t', encoding='utf-8'), False
    
        k = 0  # number of equations (final)
        variable_names = cfg.params["fit"]["variable_names"] # list of objective function names
        table_main = [{i: [{}, {}]} for i in variable_names]  # dict/table coefficients left/right parts of the equation
    
        # Loading temporary data (for saving temp results)
        if os.path.exists(f'data/{title}/epde_result/table_main_general.pickle'):
            with open(f'data/{title}/epde_result/table_main_general.pickle', 'rb') as f:
                table_main = pickle.load(f)
            with open(f'data/{title}/epde_result/k_main_general.pickle', 'rb') as f:
                k = pickle.load(f)
    
        for test_idx in np.arange(cfg.params["glob_epde"]["test_iter_limit"]):
            epde_obj = equation_fit(u, grid_u, derives, cfg)
            res = epde_obj.equation_search_results(only_print=False, level_num=cfg.params["results"]["level_num"])  # result search
    
            table_main, k = collection.object_table(res, variable_names, table_main, k)
            # To save temporary data
            with open(f'data/{title}/epde_result/table_main_general.pickle', 'wb') as f:
                pickle.dump(table_main, f, pickle.HIGHEST_PROTOCOL)
    
            with open(f'data/{title}/epde_result/k_main_general.pickle', 'wb') as f:
                pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)
    
            print(test_idx)
    
        frame_main = collection.preprocessing_bamt(variable_names, table_main, k)
    
        if cfg.params["glob_epde"]["save_result"]:
            if os.path.exists(f'data/{title}/epde_result/output_main_{title}.csv'):
                frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/epde_result/"))}.csv', sep='\t', encoding='utf-8')
            else:
                frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}.csv', sep='\t', encoding='utf-8')
    
        return frame_main, epde_obj