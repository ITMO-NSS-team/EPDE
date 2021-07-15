# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:32:53 2021

@author: user
"""
from copy import copy

def finite_diff_shift(diff,axis):
    diff_p=copy(diff)
    diff_m=copy(diff)
    diff_p[axis]=diff_p[axis]+1
    diff_m[axis]=diff_m[axis]-1
    return [diff_p,diff_m]

def scheme_build(axes,varn):
    order=len(axes)
    finite_diff=[]
    for i in range(varn):
        finite_diff+=[0]
    finite_diff=[finite_diff]
    for i in range(order):
        diff_list=[]
        for diff in finite_diff:
            f_diff=finite_diff_shift(diff,axes[i])
            if len(diff_list)==0:
                diff_list=f_diff
            else:
                for diffs in f_diff:
                    diff_list.append(diffs)
        finite_diff=diff_list
    return finite_diff

flatten_list = lambda t: [item for sublist in t for item in sublist]
    

def sign_order(order):
    sign_list=[1]
    for i in range(order):
        start_list=[]
        for sign in sign_list:
            if sign==1:
                start_list.append([1,-1])
            else:
                start_list.append([-1,1])
        sign_list=flatten_list(start_list)
    return sign_list
    