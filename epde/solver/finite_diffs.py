# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:32:53 2021

@author: user
"""
from copy import copy

# the idea is simple - central difference changes [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
def finite_diff_shift(diff,axis,mode):
    diff_p=copy(diff)
    diff_m=copy(diff)
    if mode=='central':
        diff_p[axis]=diff_p[axis]+1
        diff_m[axis]=diff_m[axis]-1
    elif mode=='f':
        diff_p[axis]=diff_p[axis]+1
    elif mode=='b':
        diff_m[axis]=diff_m[axis]-1
    return [diff_p,diff_m]
'''
we do the [0]->([1]-[-1])/(2h) transitions to the axes we need
as an example d2u/dxdt
u=[0,0]
u-> du/dx:

[0,0]->([1,0]-[-1,0])/(2h)

du/dx->d2u/dxdt:

[1,0]->([1,1]-[1,-1])/(2h*2tau)

[-1,0]->([-1,1]-[-1,-1])/(2h*2tau)

But we do not want to take signs into account (too complex), so

u-> du/dx:

[0,0]->[[1,0],[-1,0]]

du/dx->d2u/dxdt:

[[1,0],[-1,0]]->[[1,1],[1,-1],[-1,1],[-1,-1]]

Since order is preserved we can compute signs afterwards

'''
def scheme_build(axes,varn,axes_mode):
    order=len(axes)
    finite_diff=[]
    direction_list=[]
    # we generate [0,0,0,...] for number of variables (varn)
    for i in range(varn):
        finite_diff+=[0]
    # just to make this [[0,0,...]]
    finite_diff=[finite_diff]
    # when we increase differential order
    for i in range(order):
        diff_list=[]
        for diff in finite_diff:
            #we use [0,0]->[[1,0],[-1,0]] rule for the axis
            if axes_mode=='central':
                f_diff=finite_diff_shift(diff,axes[i],'central')
            else:
                f_diff=finite_diff_shift(diff,axes[i],axes_mode[axes[i]])
            if len(diff_list)==0:
                #and put it to the pool of differentials if it is empty
                diff_list=f_diff
            else:
                # or add to the existing pool
                for diffs in f_diff:
                    diff_list.append(diffs)
        # the we go to the next differential if needed
        finite_diff=diff_list
        direction_list.append(axes_mode[axes[i]])
    return finite_diff,direction_list

# axiluary function to flatten list
flatten_list = lambda t: [item for sublist in t for item in sublist]
    
'''
From transormations above, we always start from +1 (1)

Every +1 changes to ->[+1,-1] when order of differential rises

[0,0] (+1) ->([1,0]-[-1,0]) ([+1,-1])

Every -1 changes to [-1,+1]

[[1,0],[-1,0]] ([+1,-1])->[[1,1],[1,-1],[-1,1],[-1,-1]] ([+1,-1,-1,+1])

'''
def sign_order(order,mode,h=1/2):
    sign_list=[1]
    for i in range(order):
        start_list=[]
        for sign in sign_list:
            if mode=='central':
                start_list.append([sign*(1/(2*h)),-sign*(1/(2*h))])
            else:
                start_list.append([sign/h,-sign/h])
        sign_list=flatten_list(start_list)
    return sign_list



'''
The following material is presented in illustrative matters
'''


"""
For boundaries we have to use at the one hand - 2nd order scheme, at the other - forward/backward scheme.

Tests show, that we cannot use center scheme even for infinite small h for boundary points, since it gives
function from other that Sobolev space for a given equation.

Thus, everything becomes more serious. Brace yourself, if you go further to the code
"""




'''
2nd order forward difference changes [0]->(a*[2]+b*[1]+c*[0]) (in terms of grid nodes position)
with coefficients a=-3/(2h), b=2/h, c=-1/2h
'''

'''
Lets play with shifts and coeffs separately.
One shift becomes three different instead of 2.
'''
# def second_order_f_shift(diff,axis):
#     diff_1=copy(diff)
#     diff_2=copy(diff)
#     diff_3=copy(diff)
#     diff_3[axis]=diff_3[axis]+2
#     diff_2[axis]=diff_2[axis]+1
#     return [diff_3,diff_2,diff_1]
'''
We also do not want take coeffs into account (too complex), so

u-> du/dx:

[0,0]->[[2,0],[1,0],[0,0]]

du/dx->d2u/dxdt:

[[2,0],[1,0],[0,0]]->[[2,2],[2,1],[2,0],[1,2],[1,1],[1,0],[0,2],[0,1],[0,0]] 

(damn this long)

Since order is preserved we can compute signs afterwards

'''
# def second_order_f_scheme_build(axes,varn):
#     order=len(axes)
#     finite_diff=[]
#     # we generate [0,0,0,...] for number of variables (varn)
#     for i in range(varn):
#         finite_diff+=[0]
#     # just to make this [[0,0,...]]
#     finite_diff=[finite_diff]
#     # when we increase differential order
#     for i in range(order):
#         diff_list=[]
#         for diff in finite_diff:
#             #we use [0,0]->[[1,0],[-1,0]] rule for the axis
#             f_diff=second_order_f_shift(diff,axes[i])
#             if len(diff_list)==0:
#                 #and put it to the pool of differentials if it is empty
#                 diff_list=f_diff
#             else:
#                 # or add to the existing pool
#                 for diffs in f_diff:
#                     diff_list.append(diffs)
#         # the we go to the next differential if needed
#         finite_diff=diff_list
#     return finite_diff


'''
From transormations above, we always start from +1 (1)

Every +1 changes to ->[1,-4,3] when order of differential rises

[0,0] (+1) ->[0,0]->[[2,0],[1,0],[0,0]] ([-3,4,-1])

Every a changes to [3a,-4a,a]

[[2,0],[1,0],[0,0]] ([3,-4,1])->[[2,2],[2,1],[2,0],[1,2],[1,1],[1,0],[0,2],[0,1],[0,0]]  ([3,-4,1,-12,16,-4,9,-12,3])

further this should be multiplied by (-1/2h)**order

'''

# def second_order_f_sign_order(order):
#     sign_list=[1]
#     for i in range(order):
#         start_list=[]
#         for sign in sign_list:
#             start_list.append([-3*sign,4*sign,-sign])
#         sign_list=flatten_list(start_list)
#     return sign_list
    

'''
backward is the same as forward, but with signs inversed
'''


# def second_order_b_shift(diff,axis):
#     diff_1=copy(diff)
#     diff_2=copy(diff)
#     diff_3=copy(diff)
#     diff_3[axis]=diff_3[axis]-2
#     diff_2[axis]=diff_2[axis]-1
#     return [diff_3,diff_2,diff_1]


# def second_order_b_scheme_build(axes,varn):
#     order=len(axes)
#     finite_diff=[]
#     # we generate [0,0,0,...] for number of variables (varn)
#     for i in range(varn):
#         finite_diff+=[0]
#     # just to make this [[0,0,...]]
#     finite_diff=[finite_diff]
#     # when we increase differential order
#     for i in range(order):
#         diff_list=[]
#         for diff in finite_diff:
#             #we use [0,0]->[[1,0],[-1,0]] rule for the axis
#             f_diff=second_order_b_shift(diff,axes[i])
#             if len(diff_list)==0:
#                 #and put it to the pool of differentials if it is empty
#                 diff_list=f_diff
#             else:
#                 # or add to the existing pool
#                 for diffs in f_diff:
#                     diff_list.append(diffs)
#         # the we go to the next differential if needed
#         finite_diff=diff_list
#     return finite_diff

# def second_order_b_sign_order(order):
#     sign_list=[1]
#     for i in range(order):
#         start_list=[]
#         for sign in sign_list:
#             start_list.append([3*sign,-4*sign,sign])
#         sign_list=flatten_list(start_list)
#     return sign_list

'''
The following functions are forward and backward schemes combined
'''

def second_order_shift(diff,axis,mode):
    diff_1=copy(diff)
    diff_2=copy(diff)
    diff_3=copy(diff)
    if mode=='f':
        diff_3[axis]=diff_3[axis]+2
        diff_2[axis]=diff_2[axis]+1
    elif mode=='b':
        diff_3[axis]=diff_3[axis]-2
        diff_2[axis]=diff_2[axis]-1
    else:
        print('Wrong mode')
    return [diff_3,diff_2,diff_1]    


def second_order_scheme_build(axes,varn,axes_mode):
    order=len(axes)
    finite_diff=[]
    direction_list=[]
    # we generate [0,0,0,...] for number of variables (varn)
    for i in range(varn):
        finite_diff+=[0]
    # just to make this [[0,0,...]]
    finite_diff=[finite_diff]
    # when we increase differential order
    for i in range(order):
        diff_list=[]
        for diff in finite_diff:
            #we use [0,0]->[[1,0],[-1,0]] rule for the axis
            f_diff=second_order_shift(diff,axes[i],axes_mode[axes[i]])
            if len(diff_list)==0:
                #and put it to the pool of differentials if it is empty
                diff_list=f_diff
            else:
                # or add to the existing pool
                for diffs in f_diff:
                    diff_list.append(diffs)
        # the we go to the next differential if needed
        finite_diff=diff_list
        direction_list.append(axes_mode[axes[i]])
    return finite_diff,direction_list

def second_order_sign_order(order,mode,h=1/2):
    sign_list=[1]
    for i in range(order):
        start_list=[]
        for sign in sign_list:
            if mode[i]=='f':
                start_list.append([3*(1/(2*h))*sign,-4*(1/(2*h))*sign,(1/(2*h))*sign])
            elif mode[i]=='b':
                start_list.append([-3*(1/(2*h))*sign,4*(1/(2*h))*sign,-(1/(2*h))*sign])
        sign_list=flatten_list(start_list)
    return sign_list