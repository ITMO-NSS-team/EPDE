# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:06:41 2021

@author: Sashka
"""

from epde.solver.finite_diffs import scheme_build, sign_order
import numpy as np
#from finite_diffs import sign_order
import math
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def shift_points(grid,axis,shift):
    grid_shift=grid.clone()
    grid_shift[:,axis]=grid[:,axis]+shift
    return grid_shift


def finite_diff_to_model(finite_diff_scheme,s_order,model,grid,h=0.00001):
    der=0
    for i,shifts in enumerate(finite_diff_scheme):
        s_grid=grid
        for j,axis in enumerate(shifts):
            if axis!=0:
                s_grid=shift_points(s_grid,j,axis*h)
        der+=s_order[i]*model(s_grid)
    nshifts=int(math.log(len(s_order))/math.log(2))
    der=der/((2**nshifts)*(h**nshifts))
    return der


def take_derivative(model,grid,dif_type,nvars,h=0.0001):
    const = dif_type[0]
    vars_set = dif_type[1]
    power = dif_type[2]
    if type(power) is list:
        for k,term in enumerate(vars_set):
            s_order=sign_order(len(term))
            scheme=scheme_build(term,nvars)
            try:
                der=der*(finite_diff_to_model(scheme,s_order,model,grid,h=h)**power[k])
            except NameError:
                der=finite_diff_to_model(scheme,s_order,model,grid,h=h)**power[k]
    else:
        s_order=sign_order(len(vars_set))
        scheme=scheme_build(vars_set,nvars)
        der=finite_diff_to_model(scheme,s_order,model,grid,h=h)
        der=der**power
    const_t = torch.from_numpy(np.full(shape=tuple(der.shape), fill_value=const))
    der=const_t*der
    print(type(const_t), type(der))
    return der


def apply_const_operator(model,grid, operator,nvars,h=0.0001):
    for term in operator:
        dif=take_derivative(model,grid,term,nvars,h=h)
        try:
            total+=dif
        except NameError:
            total=dif
    return total



def operator_loss(grid,model,operator,bconds,lambda_bound=10):
    h=0.001
    op=apply_const_operator(model,grid, operator,2,h=h)
    b_val_list=[]
    true_b_val_list=[]
    for bcond in bconds:
        boundary_val=model(bcond[0])
        b_val_list.append(boundary_val)
        true_boundary_val=bcond[1].reshape(-1,1)
        true_b_val_list.append(true_boundary_val)
    b_val=torch.cat(b_val_list)
    true_b_val=torch.cat(true_b_val_list)
    loss = torch.mean((op)**2)+lambda_bound*torch.mean((b_val-true_b_val)**2)
    return loss



def solver(grid,model,operator,bconds,lambda_bound=10,verbose=False,learning_rate = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss =  operator_loss(grid,model,operator,bconds,lambda_bound=lambda_bound)
    t=0
    while abs(loss.item())>0.1:
        loss = operator_loss(grid,model,operator,bconds,lambda_bound=lambda_bound)
        if (t % 1000 == 0) and verbose:
            print(t, loss.item())
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2)
            plt.show()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t+=1
    return model