#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:47 2024

@author: maslyaev
"""
import os
import numpy as np
import pandas as pd

import warnings
import itertools
import tempfile
import pickle

from sklearn import preprocessing

from epde.interface.interface import EpdeSearch
from epde.loader import EPDELoader, temp_pickle_save
from epde.uncertainity.utils import equation_fit, object_table, preprocessing_bamt, token_check, \
    get_objects

try:
    import bamt.Network as Nets
    import bamt.Nodes as Nodes
    import bamt.Preprocessors as pp
except ModuleNotFoundError:
    warnings.warn('BAMT library is missing for the bayesian network training. If you plan to use uncertainty \
                   estimation please, follow installation instructions in https://github.com/aimclub/BAMT repository.')

try:
    from pgmpy.estimators import K2Score
except ImportError:
    warnings.warn('pgmpy library is missing. Installation instructions are present in \
                   https://github.com/pgmpy/pgmpy repository.')


class BayesianApproach(object):
    def __init__(self):
        pass
    
    def _get_equations(self, cfg, title, data, grid, derives = None):
        k = 0
        variable_names = cfg.params["fit"]["variable_names"] # list of objective function names
        table_main = [{i: [{}, {}]} for i in variable_names]  # dict/table coefficients left/right parts of the equation
    
        # Loading temporary data (for saving temp results)
        if os.path.exists(f'data/{title}/epde_result/table_main_general.pickle'):
            with open(f'data/{title}/epde_result/table_main_general.pickle', 'rb') as f:
                table_main = pickle.load(f)
            with open(f'data/{title}/epde_result/k_main_general.pickle', 'rb') as f:
                k = pickle.load(f)
    
        for test_idx in np.arange(cfg.params["glob_epde"]["test_iter_limit"]):
            epde_obj = equation_fit(data, grid, derives, cfg)
            res = epde_obj.equation_search_results(only_print=False, level_num=cfg.params["results"]["level_num"])  # result search
    
            table_main, k = object_table(res, variable_names, table_main, k)
            # To save temporary data
            with open(f'data/{title}/epde_result/table_main_general.pickle', 'wb') as f:
                pickle.dump(table_main, f, pickle.HIGHEST_PROTOCOL)
    
            with open(f'data/{title}/epde_result/k_main_general.pickle', 'wb') as f:
                pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)
    
            print(test_idx)
    
        frame_main = preprocessing_bamt(variable_names, table_main, k)
    
        if cfg.params["glob_epde"]["save_result"]:
            if os.path.exists(f'data/{title}/epde_result/output_main_{title}.csv'):
                frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/epde_result/"))}.csv', sep='\t', encoding='utf-8')
            else:
                frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}.csv', sep='\t', encoding='utf-8')
    
        return frame_main, epde_obj
    
    def _get_bayesian_network(self, df, cfg, title):
        if not (os.path.exists(f'data/{title}/bamt_result')):
            os.mkdir(f'data/{title}/bamt_result')
    
        if cfg.params["glob_bamt"]["load_result"]:
            with open(f'data/{title}/bamt_result/data_equations_{cfg.params["glob_bamt"]["sample_k"]}.pickle', 'rb') as f:
                return pickle.load(f)
    
        # Rounding values
        for col in df.columns:
            df[col] = df[col].round(decimals=10)
        # Deleting rows with condition
        df = df.loc[(df.sum(axis=1) != -len(cfg.params["fit"]["variable_names"])), (df.sum(axis=0) != 0)]
        # Deleting null columns
        df = df.loc[:, (df != 0).any(axis=0)]
        # (df != 0).sum(axis = 0)
    
        df_initial = df.copy()
    
        for col in df.columns:
            if '_r' in col:
                df = df.astype({col: "int64"})
                df = df.astype({col: "str"})
    
        all_r = df.shape[0]
        unique_r = df.groupby(df.columns.tolist(), as_index=False).size().shape[0]
    
        print(f'Из {all_r} полученных систем \033[1m {unique_r} уникальных \033[0m ({int(unique_r / all_r * 100)} %)')
    
        l_r, l_left = [], []
        for term in list(df.columns):
            if '_r' in term:
                l_r.append(term)
            else:
                l_left.append(term)
        df = df[l_left + l_r]
    
        discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        encoder = preprocessing.LabelEncoder()
        p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
        data, est = p.apply(df)
        info_r = p.info
    
        bn = Nets.HybridBN(has_logit=True, use_mixture=True)
        bn.add_nodes(info_r)
    
        df_temp = (df_initial[[col for col in df_initial.columns if '_r' in col]] != 0).copy()
        print(df_temp.sum(axis=0).sort_values(ascending=False)[:len(cfg.params["fit"]["variable_names"])])
        init_nodes_list = []
        for i in range(len(cfg.params["fit"]["variable_names"])):
            init_nodes = df_temp.sum(axis=0).idxmax()
            init_nodes_list.append(init_nodes)
            df_temp = df_temp.drop(init_nodes, axis=1)
        print(init_nodes_list)
        params = {"init_nodes": init_nodes_list} if not cfg.params["params"]["init_nodes"] else cfg.params[
            "params"]
    
        bn.add_edges(data, scoring_function=('K2', K2Score), params=params)
        bn.fit_parameters(df_initial)
    
        objects_res = []
        while len(objects_res) < cfg.params["glob_bamt"]["sample_k"]:
            synth_data = bn.sample(30, as_df=True)
            temp_res = get_objects(synth_data, cfg)
    
            if len(temp_res) + len(objects_res) > cfg.params["glob_bamt"]["sample_k"]:
                objects_res += temp_res[:cfg.params["glob_bamt"]["sample_k"] - len(objects_res)]
            else:
                objects_res += temp_res
    
        if cfg.params["correct_structures"]["list_unique"] is not None:
            token_check(df_initial.columns, objects_res, cfg)
    
        if cfg.params["glob_bamt"]["save_result"]:
            number_of_files = len(os.listdir(path=f"data/{title}/bamt_result/"))
            if os.path.exists(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.csv'):
                with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}_{number_of_files}.pickle', 'wb') as f:
                    pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.pickle', 'wb') as f:
                    pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)
    
        return objects_res
    
    def train_series(self, config, data, grid, derivs = None):      
        self._get_bayesian_network
        