# -*- coding: utf-8 -*-

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


def Lotka_Volterra_by_RK(initial : tuple, timestep : float, steps : int, alpha : float, 
                         beta : float, delta : float, gamma : float):
    """
    Calculates the Lotka-Volterra model using the Runge-Kutta method.
    
        This method simulates the Lotka-Volterra equations, a mathematical model
        describing the dynamics of biological systems in which two species interact,
        one as a predator and the other as prey, using the 4th order Runge-Kutta
        method. It is used to generate synthetic data of interacting populations,
        which can then be used to test equation discovery algorithms. By simulating
        known dynamics, the accuracy and efficiency of these algorithms can be
        evaluated.
    
        Args:
            initial: Initial population sizes of prey and predator.
            timestep: The size of the time step used in the Runge-Kutta method.
            steps: The number of time steps to simulate.
            alpha: The natural growth rate of prey.
            beta: The death rate of prey due to predation.
            delta: The growth rate of predators due to eating prey.
            gamma: The natural death rate of predators.
    
        Returns:
            np.ndarray: A 2D array where each row represents a time step and the
                two columns represent the prey and predator population sizes,
                respectively.
    """
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
    """
    Calculates the time derivative of the Lorenz system.
    
        This function is a crucial component for simulating the Lorenz attractor,
        as it defines the system's dynamics. It computes the rate of change of the
        system's state variables (x, y, z) at a given time, which is essential
        for numerical integration and visualization of the chaotic behavior.
    
        Args:
            x (np.ndarray): A 3D vector representing the current state of the system (x, y, z).
            t (float): The current time. While not directly used in the calculation, it's a required argument for ODE solvers.
    
        Returns:
            np.ndarray: A 3D vector representing the derivative of the system's state at the given time (dx/dt, dy/dt, dz/dt).
    """
    sigma = 10.; rho = 28.; beta = 8./3.
    res = np.empty(3)
    res[0] = sigma * (x[1] - x[0])
    res[1] = x[0] * (rho - x[2]) - x[1]
    res[2] = x[0] * x[1] - beta * x[2]
    return res

def solve(initial : tuple, timestep : float, steps : int):
    """
    Solves a system of differential equations using the Runge-Kutta 4th order method to model the system's evolution.
    
        This method is crucial for simulating the behavior of a system over time,
        given its initial state and the differential equations that govern its dynamics.
        It iteratively approximates the system's state at discrete time steps,
        allowing us to observe how the system evolves from its initial conditions.
    
        Args:
            initial (tuple): The initial conditions for the system's variables.
            timestep (float): The size of each time step.
            steps (int): The number of steps to take.
    
        Returns:
            np.ndarray: A 2D array where each row represents the state of the system
                at a given time step. The shape of the array is (steps, len(initial)).
                This array represents the trajectory of the system through time,
                allowing for analysis of its behavior and characteristics.
    """
    res = np.full(shape = (steps, len(initial)), fill_value = initial, dtype=np.float64)
    for step in range(steps-1):
        k1s = dxdt(res[step, :], timestep * step)
        k2s = dxdt(res[step, :] + 0.5 * timestep * k1s, timestep * (step + 0.5))
        k3s = dxdt(res[step, :] + 0.5 * timestep * k2s, timestep * (step + 0.5))
        k4s = dxdt(res[step, :] + timestep * k3s, timestep * (step + 1.))
        
        res[step+1, :] = res[step, :] + timestep/6. * (k1s + 2*k2s + 2*k3s + k4s)
    return res

def plot3d(solution):
    """
    Plots the discovered equation in 3D space to visualize its behavior.
    
        Args:
            solution (np.ndarray): A NumPy array representing the solution to be plotted.
                It is expected to have shape (N, 3), where N is the number of points
                and each point has X, Y, and Z coordinates.
    
        Returns:
            None. Displays the 3D plot using matplotlib.
    
        Why: Visualizing the solution in 3D helps in understanding the behavior of the
        discovered equation and verifying if it accurately represents the underlying dynamics
        of the system.
    """
    ax = plt.figure().add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.plot(*solution.T, 'black', lw = 0.3)#[:, 0], solution[:, 1], solution[:, 2], )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(axis='both', which='major', pad=0)
    plt.show()
    

def classical_plot(solution, time, time_max = -1, step = 1):    
    """
    Generates a classical plot of the solution over time to visualize the identified system dynamics.
        
        This method iterates through each variable in the solution, plotting its
        values against the given time points. It enhances understanding of the system's behavior
        by providing a visual representation of variable interactions over time.
        It then adds a grid, legend, and axis labels to the plot before saving it to a file and displaying it.
        
        Args:
            solution: The solution array, where each column represents a variable
                and each row represents a time point.
            time: The time array corresponding to the rows in the solution array.
            time_max: The maximum time index to plot up to. If -1, plots all time points. Defaults to -1.
            step: The step size for plotting time points. Defaults to 1.
        
        Returns:
            None. The method saves the plot to a file and displays it.
    """
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
    step = 0.001; steps_num = 100000
    t = np.arange(start = 0, stop = step * steps_num, step = step)
    solution = solve(initial=(0, 1, 20), timestep=step, steps=steps_num)    
    

    plt.savefig('/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/fig.png', dpi=150, 
                bbox_inches='tight')
    data = np.save(file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/lorenz.npy', 
                   arr = solution)
    t = np.save(file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/t.npy', 
                arr = t)