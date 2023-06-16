#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:25:37 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd

import torch
device = torch.device('cpu')

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

from epde.preprocessing.smoothers import ANNSmoother

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

def autograd_derivs(model: torch.nn.modules.container.Sequential, grids: torch.Tensor, axis: list = [0,]):
    grids.requires_grad = True
    fi = model(grids).sum(0) # [:, var]
    for ax in axis:
        grads, = torch.autograd.grad(fi, grids, create_graph=True)
        fi = grads[:, ax].sum()
    gradient_full = grads[:, axis[-1]].reshape(-1, 1)
    return gradient_full


if __name__ == "__main__":
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
    magnitude = 0.05

    u = df.values
    u = np.transpose(u)
    
    u = u + np.random.normal(scale = magnitude * np.abs(u), size = u.shape)
    # u_train, u_test = train_test_split(u, train_max)

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    grids_flattened = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T
    grids_flattened.to(device)
    
    t_dense = np.linspace(0, 1, u.shape[0]*3)
    x_dense = np.linspace(0, 1, u.shape[1]*3)
    grids_dense = np.meshgrid(t_dense, x_dense, indexing = 'ij')
    grid_dense_flattened = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids_dense])).float().T
    grid_dense_flattened.to(device)


    original_shape = grids[0].shape
    dense_shape = grids_dense[0].shape

    smoother = ANNSmoother()
    
    init_params = {'loss_mean' : 1e3, 'batch_frac' : 0.5, 'learining_rate' : 1e-4}
    
    models = []
    epochs = np.logspace(2, 4, num = 3)
    for epochs_num in epochs:
        models.append(smoother(u, grids, epochs_max = epochs_num, return_ann=True, **init_params))
        
    preds = []
    for idx, epochs_num in enumerate(epochs):
        pred = models[idx][1](grid_dense_flattened).detach().numpy().reshape(dense_shape)
        preds.append(pred)
        Heatmap(pred, title=f'Train epochs: {epochs_num}')