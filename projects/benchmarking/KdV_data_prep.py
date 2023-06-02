#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:35:54 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd

'''

You can install EPDE directly from our github repo:
    pip install git+https://github.com/ITMO-NSS-team/EPDE@main    

'''

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, CustomEvaluator, CustomTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

if __name__ == "__main__":
    try:
        dir_path = os.path.join(os.path.dirname( __file__ ), 'KdV/Data/')
        # data = loadmat(u_file)
    except (FileNotFoundError, OSError):
        dir_path = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/Data/'
    
    df = pd.read_csv(f'{dir_path}KdV_sln_100.csv', header=None)
    dddx = pd.read_csv(f'{dir_path}ddd_x_100.csv', header=None)
    ddx = pd.read_csv(f'{dir_path}dd_x_100.csv', header=None)
    dx = pd.read_csv(f'{dir_path}d_x_100.csv', header=None)
    dt = pd.read_csv(f'{dir_path}d_t_100.csv', header=None)

    u = df.values
    u = np.transpose(u)

    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_t = dt.values
    d_t = np.transpose(d_t)
    
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    