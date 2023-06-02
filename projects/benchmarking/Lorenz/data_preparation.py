# -*- coding: utf-8 -*-

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


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
        
def dxdt(x, t):
    sigma = 100.; rho = 280.; beta = 80./3. # Added extra zeros
    res = np.empty(3)
    res[0] = sigma * (x[1] - x[0])
    res[1] = x[0] * (rho - x[2]) - x[1]
    res[2] = x[0] * x[1] - beta * x[2]
    return res

def solve(initial : tuple, timestep : float, steps : int):
    res = np.full(shape = (steps, len(initial)), fill_value = initial, dtype=np.float64)
    for step in range(steps-1):
        k1s = dxdt(res[step, :], timestep * step)
        k2s = dxdt(res[step, :] + 0.5 * timestep * k1s, timestep * (step + 0.5))
        k3s = dxdt(res[step, :] + 0.5 * timestep * k2s, timestep * (step + 0.5))
        k4s = dxdt(res[step, :] + timestep * k3s, timestep * (step + 1.))
        
        res[step+1, :] = res[step, :] + timestep/6. * (k1s + 2*k2s + 2*k3s + k4s)
    return res

def plot3d(solution):
    ax = plt.figure().add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.plot(*solution.T, 'black', lw = 0.3)#[:, 0], solution[:, 1], solution[:, 2], )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(axis='both', which='major', pad=0)
    plt.show()
    

def classical_plot(solution, time, time_max = -1, step = 1):    
    colors = ['k', 'r', 'b']
    for var_idx in range(solution.shape[1]):
        plt.scatter(time[:time_max:step], solution[:time_max:step, var_idx], color = colors[var_idx], 
                    label = f'x{var_idx}')
    plt.grid()
    plt.legend(prop={'size': 13})
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/meas.png', dpi=150,
                bbox_inches='tight')

    plt.show()
    
if __name__ == "__main__":
    step = 0.0001; steps_num = 10000
    t = np.arange(start = 0, stop = step * steps_num, step = step)
    solution = solve(initial=(0, 1, 20), timestep=step, steps=steps_num)    
    
    # for var_idx in range(solution.shape[1]):
    #     plt.scatter(time[:time_max:step], solution[:time_max:step, var_idx], color = colors[var_idx], 
    #                 label = f'x{var_idx}')
    # plt.grid()
    # plt.legend(prop={'size': 13})
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    

    # plt.savefig('/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/fig.png', dpi=150, 
    #             bbox_inches='tight')
    data = np.save(file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/lorenz_1.npy', 
                    arr = solution)
    t = np.save(file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/t_1.npy', 
                arr = t)