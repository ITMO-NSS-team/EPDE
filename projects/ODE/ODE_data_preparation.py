#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:56:50 2022

@author: maslyaev
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

"""
Здесь мы будем готовить данные из решения ОДУ: y'' + sin(2x) y' + 4y = 1.5x 
Метод: рассматриваем его как систему ОДУ первого порядка, потом метод Рунге-Кутты
"""


def second_order_ODE_by_RK(initial : tuple, timestep : float, steps : int, g1 : Callable, 
                           g2 : Callable, g3 : Callable, g4 : Callable):
    res = np.full(shape = (steps, 2), fill_value = initial, dtype=np.float64)
    for step in range(steps-1):
        # print(res[step, :])
        t = step*timestep
        k1 = res[step, 1] ; x1 = res[step, 0] + timestep/2. * k1
        l1 = (g4(t) - g3(t)*res[step, 0] - g2(t)*res[step, 1]) / g1(t); y1 = res[step, 1] + timestep/2. * l1

        k2 = y1; x2 = res[step, 0] + timestep/2. * k2
        l2 = (g4(t) - g3(t)*x1 - g2(t)*y1) / g1(t); y2 = res[step, 1] + timestep/2. * l2

        k3 = y2
        l3 = (g4(t) - g3(t)*x2 - g2(t)*y2) / g1(t);
        
        x3 = res[step, 0] + timestep * k1 - 2 * timestep * k2 + 2 * timestep * k3
        y3 = res[step, 1] + timestep * l1 - 2 * timestep * l2 + 2 * timestep * l3
        k4 = y3
        l4 = (g4(t) - g3(t)*x3 - g2(t)*y3) / g1(t)
        
        res[step+1, 0] = res[step, 0] + timestep / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        res[step+1, 1] = res[step, 1] + timestep / 6. * (l1 + 2 * l2 + 2 * l3 + l4)
    return res
        
if __name__ == '__main__':
    g1 = lambda x: 1.
    g2 = lambda x: np.sin(2*x)#.
    g3 = lambda x: 4.
    g4 = lambda x: 1.5*x #np.cos(3*x)
    
    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    solution = second_order_ODE_by_RK(initial=(0.8, 2.), timestep=step, steps=steps_num, 
                                      g1=g1, g2=g2, g3=g3, g4=g4)
    # plt.plot(solution[:, 0], solution[:, 1])
    
    # fig, ax = plt.subplots()    
    # # ax.plot(t, solution[:, 0], color= 'k')
    # # ax.plot(t, solution[:, 1], color= 'r')
    # ax.set_xlabel('Prey')
    # ax.set_ylabel('Hunters')
    # ax.plot(solution[:, 0], solution[:, 1], color= 'k')
    # ax.grid()
    # plt.savefig("projects/hunter-prey/pictures/HQ_graph_comb.png", dpi=300)    
    np.save('/home/maslyaev/epde/EPDE_main/projects/ODE/data/ODE_sol.npy', solution[:, 0])
    np.save('/home/maslyaev/epde/EPDE_main/projects/ODE/data/ODE_t.npy', t)