# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:06:41 2021

@author: Sashka
"""
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from epde.solver.input_preprocessing import grid_prepare,bnd_prepare,operator_prepare



def take_derivative_shift_op(model,term):
    coeff = term[0]
    shift_grid_list = term[1]
    s_order_norm_list = term[2]
    power = term[3]
    der_term=torch.zeros_like(model(shift_grid_list[0][0]))+1
    for j,scheme in enumerate(shift_grid_list):
        grid_sum=torch.zeros_like(model(scheme[0]))
        for k,grid in enumerate(scheme):
            grid_sum+=model(grid)*s_order_norm_list[j][k]
        #None if for non-differential, i.e. the function
        #Here we want to apply differential operators for every term in the product
        der_term=der_term*grid_sum**power[j]
    der_term=coeff*der_term
    return der_term

def apply_const_shift_operator(model,operator):
    for term in operator:
        dif=take_derivative_shift_op(model,term)
        try:
            total+=dif
        except NameError:
            total=dif
    return total






def apply_operator_set(model,operator_set):
    field_part=[]
    for operator in operator_set:
        field_part.append(apply_const_shift_operator(model,operator))
    field_part=torch.cat(field_part)
    return field_part

flatten_list = lambda t: [item for sublist in t for item in sublist]

def point_sort_shift_loss(model,grid,operator_set,bconds,lambda_bound=10):
    op=apply_operator_set(model,operator_set)
    true_b_val_list=[]
    b_val_list=[]
    b_pos_list=[]
    '''
    we apply boundary condition operator and compare it with the
    presctibed values in loss
    '''
    simpleform=False
    for bcond in bconds:
        if bcond[1]==None:
            simpleform=True
        if bcond[1]!=None:
            simpleform=False
            break
    if simpleform:
        for bcond in bconds:
            b_pos_list.append(bcond[0])
            true_boundary_val=bcond[2].reshape(-1,1)
            true_b_val_list.append(true_boundary_val)
        # print(flatten_list(b_pos_list))
        # b_pos=torch.cat(b_pos_list)
        true_b_val=torch.cat(true_b_val_list)
        b_op_val=model(grid)
        b_val=b_op_val[flatten_list(b_pos_list)]
    else:
        for bcond in bconds:
            b_pos=bcond[0]
            b_cond_operator=bcond[1]
            true_boundary_val=bcond[2].reshape(-1,1)
            true_b_val_list.append(true_boundary_val)
            if b_cond_operator==None or b_cond_operator==[[1,[None],1]]:
                b_op_val=model(grid)
            else:
                b_op_val=apply_operator_set(model,b_cond_operator)
            b_val_list.append(b_op_val[b_pos])
        true_b_val=torch.cat(true_b_val_list)
        b_val=torch.cat(b_val_list)
        
    '''
    actually, we can use L2 norm for the operator and L1 for boundary
    since L2>L1 and thus, boundary values become not so signifnicant, 
    so the NN converges faster. On the other hand - boundary conditions is the
    crucial thing for all that stuff, so we should increase significance of the
    coundary conditions
    '''
    loss = torch.mean((op)**2)+lambda_bound*torch.mean((b_val-true_b_val)**2)
    return loss


def point_sort_shift_solver(grid,model,operator,bconds,grid_point_subset=['central'],lambda_bound=10,verbose=False,learning_rate = 1e-3,eps=0.1,tmin=1000,tmax=1e5):
    nvars=model[0].in_features
    
    prepared_grid=grid_prepare(grid)
    bconds=bnd_prepare(bconds, prepared_grid)
    operator=operator_prepare(operator,prepared_grid,subset=grid_point_subset,true_grid=grid)
    
    # standard NN stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters())
    loss=point_sort_shift_loss(model,prepared_grid,operator,bconds,lambda_bound=lambda_bound)
    t=0
    def closure():
        optimizer.zero_grad()
        loss=point_sort_shift_loss(model,prepared_grid,operator,bconds,lambda_bound=lambda_bound)
        loss.backward()
        return loss
    while abs(loss.item())>eps or t<=tmin:
        optimizer.step(closure)
        loss=point_sort_shift_loss(model,prepared_grid,operator,bconds,lambda_bound=lambda_bound)
        if (t % 1000 == 0) and verbose:
            print(t, loss.item())
            if prepared_grid.shape[1]==2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(prepared_grid[:,0].reshape(-1), prepared_grid[:,1].reshape(-1), model(prepared_grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                plt.show()
            if prepared_grid.shape[1]==1:
                fig = plt.figure()
                plt.scatter(prepared_grid.reshape(-1),model(prepared_grid).detach().numpy().reshape(-1))
                plt.show()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        t+=1
        if t>tmax:
            break
    return model
