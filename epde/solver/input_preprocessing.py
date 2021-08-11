# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 21:31:03 2021

@author: user
"""
import torch

from epde.solver.points_type import point_typization
from epde.solver.points_type import grid_sort
from epde.solver.finite_diffs import scheme_build
from epde.solver.finite_diffs import sign_order
from epde.solver.finite_diffs import second_order_scheme_build
from epde.solver.finite_diffs import second_order_sign_order

# def operator_to_finite_diff_op(unified_operator,nvars):
#     fin_diff_op=[]
#     for term in unified_operator:
#         fin_diff_list=[]
#         s_order_list=[]
#         const = term[0]
#         vars_set = term[1]
#         power = term[2]
#         for k,term in enumerate(vars_set):
#             if term!= None:
#                 s_order=sign_order(len(term))
#                 scheme=scheme_build(term,nvars)
#             else:
#                 scheme=[None]
#                 s_order=[1]
#             fin_diff_list.append(scheme)
#             s_order_list.append(s_order)
#         fin_diff_op.append([const,fin_diff_list,s_order_list,power])
#     return fin_diff_op


def grid_prepare(grid):
    point_type=point_typization(grid)

    grid_dict=grid_sort(point_type)
    
    sorted_grid=torch.cat(list(grid_dict.values()))
    
    return sorted_grid

def operator_unify(operator):
    unified_operator=[]
    for term in operator:
        const = term[0]
        vars_set = term[1]
        power = term[2]
        if type(power) is list:
            unified_operator.append([const,vars_set,power])
        else:
            unified_operator.append([const,[vars_set],[power]])
    return unified_operator



def operator_to_type_op(unified_operator,nvars,axes_scheme_type,h=1/2,boundary_order=1):
    fin_diff_op=[]
    for term in unified_operator:
        fin_diff_list=[]
        s_order_list=[]
        const = term[0]
        vars_set = term[1]
        power = term[2]
        for k,term in enumerate(vars_set):
            if term!= [None]:
                if axes_scheme_type=='central':
                    scheme,direction_list=scheme_build(term,nvars,'central')
                    s_order=sign_order(len(term),'central',h=h)
                else:
                    if boundary_order==1:
                        scheme,direction_list=scheme_build(term,nvars,axes_scheme_type)
                        s_order=sign_order(len(term),direction_list,h=h)
                    elif boundary_order==2:
                        scheme,direction_list=second_order_scheme_build(term,nvars,axes_scheme_type)
                        s_order=second_order_sign_order(len(term),direction_list,h=h)
            else:
                scheme=[None]
                s_order=[1]
            fin_diff_list.append(scheme)
            s_order_list.append(s_order)
        fin_diff_op.append([const,fin_diff_list,s_order_list,power])
    return fin_diff_op



def shift_points(grid,axis,shift):
    grid_shift=grid.clone()
    grid_shift[:,axis]=grid[:,axis]+shift
    return grid_shift


def finite_diff_scheme_to_grid_list(finite_diff_scheme,grid,h=0.001):
    s_grid_list=[]
    for i,shifts in enumerate(finite_diff_scheme):
        s_grid=grid
        for j,axis in enumerate(shifts):
            if axis!=0:
                s_grid=shift_points(s_grid,j,axis*h)
        s_grid_list.append(s_grid)
    return s_grid_list

def type_op_to_grid_shift_op(fin_diff_op,grid,h=0.001,true_grid=None):
    shift_grid_op=[]
    for term1 in fin_diff_op:
        shift_grid_list=[]
        coeff1 = term1[0]
        if type(coeff1)==int:
            coeff=coeff1
        elif callable(coeff1):
            coeff=coeff1(grid)
            coeff=coeff.reshape(-1,1)
        elif type(coeff1)==torch.Tensor:
            if true_grid!=None:
                pos=bndpos(true_grid,grid)
            else:
                pos=bndpos(grid,grid)
            coeff=coeff1[pos].reshape(-1,1)
        finite_diff_scheme = term1[1]
        s_order=term1[2]
        power = term1[3]
        for k,term in enumerate(finite_diff_scheme):
            if term!= [None]:
                grid_op=finite_diff_scheme_to_grid_list(term,grid,h=h)
            else:
                grid_op=[grid]
            shift_grid_list.append(grid_op)
        shift_grid_op.append([coeff,shift_grid_list,s_order,power])
    return shift_grid_op   



def apply_all_operators(operator,grid,h=0.001,subset=None,true_grid=None):
    operator_list=[]
    nvars=grid.shape[1]
    point_type=point_typization(grid)
    grid_dict=grid_sort(point_type)
    a=operator_unify(operator)
    for operator_type in list(grid_dict.keys()):
        if subset==None or operator_type in subset:
            b=operator_to_type_op(a,nvars,operator_type,h=h)
            c=type_op_to_grid_shift_op(b,grid_dict[operator_type],h=h,true_grid=true_grid)
            operator_list.append(c)
    return operator_list


def operator_prepare(op,grid,subset=['central'],true_grid=None,h=0.001):
    op1=operator_unify(op)
    prepared_operator=apply_all_operators(op1, grid,subset=subset,true_grid=true_grid,h=h)
    return prepared_operator


def bndpos(grid,bnd):
    bndposlist=[]
    for point in bnd:
#        print('point: ', point)
#        print('grid:', grid.dtype)
        pos=int(torch.where(torch.all(torch.isclose(grid,point),dim=1))[0])
        bndposlist.append(pos)
    return bndposlist


def bnd_unify(bconds):
    unified_bconds=[]
    for bcond in bconds:
        if len(bcond)==2:
            unified_bconds.append([bcond[0],None,bcond[1]])
        elif len(bcond)==3:
            unified_bconds.append(bcond) 
    return unified_bconds

def bnd_prepare(bconds,grid,h=0.001):
    bconds=bnd_unify(bconds)
    prepared_bnd=[]    
    for bcond in bconds:
        b_coord=bcond[0]
        bop=bcond[1]
        bval=bcond[2]
        bpos=bndpos(grid,b_coord)
        if bop==[[1,[None],1]]:
            bop=None
        if bop!=None:
            bop1=operator_unify(bop)
            bop2=apply_all_operators(bop1, grid,h=h)
        else:
            bop2=None
        prepared_bnd.append([bpos,bop2,bval])
    
    return prepared_bnd
        



