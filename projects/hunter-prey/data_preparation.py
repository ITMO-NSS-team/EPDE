#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:56:50 2022

@author: maslyaev
"""

import numpy as np
import matplotlib.pyplot as plt

def Lotka_Volterra_by_RK(initial : tuple, timestep : float, steps : int, alpha : float, 
                         beta : float, delta : float, gamma : float):
    res = np.full(shape = (steps, 2), fill_value = initial, dtype=np.float64)
    for step in range(steps-1):
        # print(res[step, :])
        k1 = alpha * res[step, 0] - beta * res[step, 0] * res[step, 1]; x1 = res[step, 0] + timestep/2. * k1
        l1 = delta * res[step, 0] * res[step, 1] - gamma * res[step, 1]; y1 = res[step, 1] + timestep/2. * l1

        k2 = alpha * x1 - beta * x1 * y1; x2 = res[step, 0] + timestep/2. * k2
        l2 = delta * x1 * y1 - gamma * y1; y2 = res[step, 1] + timestep/2. * l2

        k3 = alpha * x2 - beta * x2 * y2
        l3 = delta * x2 * y2 - gamma * y1
        
        x3 = res[step, 0] + timestep * k1 - 2 * timestep * k2 + 2 * timestep * k3
        y3 = res[step, 1] + timestep * l1 - 2 * timestep * l2 + 2 * timestep * l3
        k4 = alpha * x3 - beta * x3 * y3
        l4 = delta * x3 * y3 - gamma * y3
        
        res[step+1, 0] = res[step, 0] + timestep / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        res[step+1, 1] = res[step, 1] + timestep / 6. * (l1 + 2 * l2 + 2 * l3 + l4)
    return res
        
if __name__ == '__main__':
    step = 0.1; steps_num = 1000
    t = np.arange(start = 0, stop = step * steps_num, step = step)
    solution = Lotka_Volterra_by_RK(initial=(1, 1), timestep=step, steps=steps_num, 
                                    alpha=2/5., beta=4./3., delta=1., gamma=1.)
    # plt.plot(solution[:, 0], solution[:, 1])
    
    fig, ax = plt.subplots()    
    # ax.plot(t, solution[:, 0], color= 'k')
    # ax.plot(t, solution[:, 1], color= 'r')
    ax.set_xlabel('Prey')
    ax.set_ylabel('Hunters')
    ax.plot(solution[:, 0], solution[:, 1], color= 'k')
    ax.grid()
    plt.savefig("projects/hunter-prey/pictures/HQ_graph_comb.png", dpi=300)    
    np.save('projects/hunter-prey/data.npy', solution)
    np.save('projects/hunter-prey/t.npy', t)
