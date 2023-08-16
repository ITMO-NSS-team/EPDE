#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
import os
import sys
import pandas as pd

# 
sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

# os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))

# from tedeous.device import solver_device

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)




from scipy.optimize import curve_fit



from projects.optics.discovery_utils import epde_discovery
from projects.optics.data_utils import read_data
from projects.optics.interp_utils import compute_derivs




def parametrized_trend(t, a, b, c,d):
    return a+b*np.arctan(c*t+d)


def optics_exp(r0_fix, exp_name='optics',trend_remove=False,custom_derivs=False,derivs_params={'interp_mode':'NN','diff_mode':'FD','diffs_plot':False,'save_derivs':False}):
    
    # solver_device('cpu')
    
    grid,rv=read_data(r0_fix)

    bonudary=0
    
    #rv=rv[80:]

    #m_grid=grid/np.max(grid)

    m_grid=grid*1e-6

    #m_grid=m_grid[80:]

    
    rv=(rv-np.min(rv))/np.max(rv-np.min(rv))


    if trend_remove:
        
        popt, _ = curve_fit(parametrized_trend, m_grid, rv)

        rv=rv-parametrized_trend(m_grid,*popt)



    #plt.plot(m_grid, rv)
    #plt.show()

    #m_grid=m_grid[:-1:10]
    #rv=rv[:-1:10]

    d=None

    if custom_derivs:
        d0,d1,d2=compute_derivs(rv,m_grid,interp_mode=derivs_params['interp_mode'],diff_mode=derivs_params['diff_mode'], plot=derivs_params['diffs_plot'],save_derivs=derivs_params['save_derivs'])
        d=np.transpose(np.stack((d1,d2)))
        rv=d0
        m_grid=m_grid[:len(d0)]

    epde_search_obj = epde_discovery(m_grid, rv,boundary=bonudary,derivs=d,use_ann=False)
        
    res=epde_search_obj.solver_forms() 

    text_eq=epde_search_obj.equations(only_print = False, only_str = True, num = 1)

    # text_eq=text_eq[0]

    # eqs=res[0]

    # results_dir=os.path.join(os.path.dirname( __file__ ), 'results_{}'.format(exp_name))

    # if not (os.path.isdir(results_dir)):
    #     os.mkdir(results_dir)
    
    # if trend_remove:
    #    interp_filename=os.path.join(results_dir,'interp_params_{}.txt'.format(r0_fix))
    #    with open(interp_filename, 'w') as the_file:
    #         the_file.write(str(popt))

    # for i,eq in enumerate(eqs):

    #     model=solver_solution(eq[0][1],rv,m_grid)
        
    #     model=model.to(torch.device('cpu'))

    #     nn_grid=torch.tensor([m_grid]).reshape(-1,1).float().to(torch.device('cpu'))

    #     plt.plot(m_grid, rv, '+', label = 'test data')
    #     plt.plot(m_grid, model(nn_grid).reshape(-1).detach().numpy(), color = 'r', label='solution of the discovered ODE')
    #     plt.grid()
    #     plt.legend(loc='upper right')
    #     img_filename=os.path.join(results_dir,'sln_{}_{}.png'.format(r0_fix,i))
    #     plt.savefig(img_filename)
    #     plt.close()
    #     txt_filename=os.path.join(results_dir,'eqn_{}_{}.txt'.format(r0_fix,i))
    #     with open(txt_filename, 'w') as the_file:
    #         the_file.write(text_eq[i])





if __name__ == "__main__":


    #for r0_fix in [0.1,0.2,0.3,0.4,0.5]:
    #    optics_exp(r0_fix,exp_name='optics')

    #for r0_fix in [0.1,0.2,0.3,0.4,0.5]:
    #    optics_exp(r0_fix,trend_remove=True,exp_name='optics_trend_remove')

    for r0_fix in [0.1,0.2,0.3,0.4,0.5]:
        optics_exp(r0_fix,exp_name='optics_custom_derivs_NN_FD',custom_derivs=True,derivs_params={'interp_mode':'NN','diff_mode':'FD','diffs_plot':False,'save_derivs':False})

    #for r0_fix in [0.1,0.2,0.3,0.4,0.5]:
    #    optics_exp(r0_fix,exp_name='optics_custom_derivs_NN_autograd',custom_derivs=True,derivs_params={'interp_mode':'NN','diff_mode':'autograd','diffs_plot':False,'save_derivs':False})

    #for r0_fix in [0.1,0.2,0.3,0.4,0.5]:
    #    optics_exp(r0_fix,exp_name='optics_custom_derivs_poly',custom_derivs=True,derivs_params={'interp_mode':'poly_3','diff_mode':'poly','diffs_plot':False,'save_derivs':False})




