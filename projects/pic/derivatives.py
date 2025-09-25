import numpy as np
import os
import matplotlib.pyplot as plt


def kdv_data(filename):
    """
    Loads data, computes time and space steps for differential equation discovery.
    
    This function reads data from a CSV file, transposes it to align with
    the expected format (likely time series data across spatial points),
    and calculates the temporal (dt) and spatial (dx) step sizes. These
    step sizes are crucial for numerical differentiation and integration
    schemes used in discovering the underlying differential equation.
    
    Args:
        filename (str): The name of the CSV file containing the data.
            The data is expected to be organized such that each row
            represents a spatial point and each column represents a time step.
    
    Returns:
        tuple: A tuple containing:
            - data (ndarray): The processed data array.
            - dt (float): The calculated time step size.
            - dx (float): The calculated space step size.
    
    Why:
        The function prepares the data and calculates essential parameters
        (dt, dx) required for subsequent numerical analysis, which is a
        fundamental step in identifying the governing differential equation
        from the provided dataset.
    """
    shape = 80
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    return data, dt, dx

def ac_data(filename: str):
    """
    Loads simulation data from a file and determines the temporal and spatial resolution. This is a preliminary step to analyze the data and identify the underlying differential equations. The increments are needed to properly discretize the domain when comparing discovered equations with the data.
    
        Args:
            filename (str): The name of the file containing the simulation data.
    
        Returns:
            tuple: A tuple containing:
              - data (np.ndarray): The loaded simulation data.
              - dt (float): The time increment between data points.
              - dx (float): The space increment between data points.
    """
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    return data, dt, dx

def darcy_data(filename: str):
    """
    Loads Darcy flow data and grid spacing from a file.
    
    This function is used to load the precomputed solutions of the Darcy flow equation,
    which serve as a testbed for evaluating the equation discovery process. The data
    represents the flow field, and the grid spacing is essential for numerical
    computations and comparisons.
    
    Args:
        filename (str): The name of the file containing the Darcy flow data.
    
    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The Darcy flow data loaded from the file.
            - dx (float): The grid spacing in the x-direction.
            - dy (float): The grid spacing in the y-direction.
    """
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    data = np.load(filename)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return data, dx, dy

def wave_data(filename):
    """
    Loads wave data and prepares it for differential equation discovery.
    
        This method reads wave data from a CSV file, transposes it to the correct format,
        and calculates the time and space increments necessary for subsequent
        differential equation modeling. The data and increments are essential
        for setting up the problem domain and numerical computations within the EPDE framework.
        
        Args:
            filename: The name of the CSV file containing the wave data.
        
        Returns:
            A tuple containing:
              - data: A NumPy array representing the wave data.
              - dt: The time increment.
              - dx: The space increment.
    """
    shape = 80
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    return data, dt, dx


directory = os.path.dirname(os.path.realpath(__file__))
ode_folder_name = os.path.join(directory, 'data\\ode')
vdp_folder_name = os.path.join(directory, 'data\\vdp')
ac_folder_name = os.path.join(directory, 'data\\ac')
wave_folder_name = os.path.join(directory, 'data\\wave')
kdv_folder_name = os.path.join(directory, 'data\\kdv')
darcy_folder_name = os.path.join(directory, 'data\\darcy')

# ODEs
# ode = np.load(os.path.join(ode_folder_name, 'ode_data.npy'))
# ode_dt = 0.05
# vdp = np.load(os.path.join(vdp_folder_name, 'vdp_data.npy'))
# vdp_dt = 0.05
#
# ode_gradient_t = np.gradient(ode, ode_dt, axis=0, edge_order=2)
# ode_gradient_tt = np.gradient(np.gradient(ode, ode_dt, axis=0, edge_order=2), ode_dt, axis=0, edge_order=2)
#
# plt.figure(figsize=(5, 4))
# plt.plot(ode)
# plt.title('data')
# plt.xlabel('t')
# plt.ylabel('x')
#
# plt.figure(figsize=(5, 4))
# plt.plot(ode_gradient_t)
# plt.title('data_t')
# plt.xlabel('t')
# plt.ylabel('x')
#
# plt.figure(figsize=(5, 4))
# plt.plot(ode_gradient_tt)
# plt.title('data_tt')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.show()
#
# vdp_gradient_t = np.gradient(vdp, vdp_dt, axis=0, edge_order=2)
# vdp_gradient_tt = np.gradient(np.gradient(vdp, vdp_dt, axis=0, edge_order=2), vdp_dt, axis=0, edge_order=2)
#
# plt.figure(figsize=(5, 4))
# plt.plot(vdp)
# plt.title('data')
# plt.xlabel('t')
# plt.ylabel('x')
#
# plt.figure(figsize=(5, 4))
# plt.plot(vdp_gradient_t)
# plt.title('data_t')
# plt.xlabel('t')
# plt.ylabel('x')
#
# plt.figure(figsize=(5, 4))
# plt.plot(vdp_gradient_tt)
# plt.title('data_tt')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.show()
#
# # 1D PDEs
# kdv, kdv_dt, kdv_dx = kdv_data(os.path.join(kdv_folder_name, 'data.csv'))
# ac, ac_dt, ac_dx = ac_data(os.path.join(ac_folder_name, 'ac_data.npy'))
# wave, wave_dt, wave_dx = wave_data(os.path.join(wave_folder_name, 'wave_sln_80.csv'))
# print(kdv.shape)
#
# kdv_gradient_t = np.gradient(kdv, kdv_dt, axis=0, edge_order=2)
# kdv_gradient_tt = np.gradient(np.gradient(kdv, kdv_dt, axis=0, edge_order=2), kdv_dt, axis=0, edge_order=2)
#
# kdv_gradient_x = np.gradient(kdv, kdv_dx, axis=1, edge_order=2)
# kdv_gradient_xx = np.gradient(np.gradient(kdv, kdv_dx, axis=1, edge_order=2), kdv_dx, axis=1, edge_order=2)
# kdv_gradient_xxx = np.gradient((np.gradient(np.gradient(kdv, kdv_dx, axis=1, edge_order=2), kdv_dx, axis=1, edge_order=2)), kdv_dx, axis=1, edge_order=2)
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv_gradient_t, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_t')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv_gradient_tt, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_tt')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv_gradient_x, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_x')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv_gradient_xx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xx')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(kdv_gradient_xxx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xxx')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.show()
#
#
# ac_gradient_t = np.gradient(ac, ac_dt, axis=0, edge_order=2)
# ac_gradient_tt = np.gradient(np.gradient(ac, ac_dt, axis=0, edge_order=2), ac_dt, axis=0, edge_order=2)
#
# ac_gradient_x = np.gradient(ac, ac_dx, axis=1, edge_order=2)
# ac_gradient_xx = np.gradient(np.gradient(ac, ac_dx, axis=1, edge_order=2), ac_dx, axis=1, edge_order=2)
# ac_gradient_xxx = np.gradient((np.gradient(np.gradient(ac, ac_dx, axis=1, edge_order=2), ac_dx, axis=1, edge_order=2)), ac_dx, axis=1, edge_order=2)
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac_gradient_t, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_t')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac_gradient_tt, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_tt')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac_gradient_x, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_x')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac_gradient_xx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xx')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(ac_gradient_xxx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xxx')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.show()
#
#
# wave_gradient_t = np.gradient(wave, wave_dt, axis=0, edge_order=2)
# wave_gradient_tt = np.gradient(np.gradient(wave, wave_dt, axis=0, edge_order=2), wave_dt, axis=0, edge_order=2)
#
# wave_gradient_x = np.gradient(wave, wave_dx, axis=1, edge_order=2)
# wave_gradient_xx = np.gradient(np.gradient(wave, wave_dx, axis=1, edge_order=2), wave_dx, axis=1, edge_order=2)
# wave_gradient_xxx = np.gradient((np.gradient(np.gradient(wave, wave_dx, axis=1, edge_order=2), wave_dx, axis=1, edge_order=2)), wave_dx, axis=1, edge_order=2)
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave_gradient_t, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_t')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave_gradient_tt, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_tt')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave_gradient_x, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_x')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave_gradient_xx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xx')
# plt.xlabel('x')
# plt.ylabel('t')
#
# plt.figure(figsize=(5, 4))
# plt.imshow(wave_gradient_xxx, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('data_xxx')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.show()

# 2D PDEs
darcy, darcy_dx, darcy_dy = darcy_data(os.path.join(darcy_folder_name, 'darcy.npy'))
nu = np.load(os.path.join(darcy_folder_name, 'darcy_nu.npy'))
print(nu.shape)

darcy_gradient_nux = np.gradient(nu[0], darcy_dx, axis=0, edge_order=2)
darcy_gradient_nuy = np.gradient(nu[0], darcy_dy, axis=1, edge_order=2)

darcy_gradient_x = np.gradient(darcy[0], darcy_dx, axis=0, edge_order=2)
darcy_gradient_xx = np.gradient(np.gradient(darcy[0], darcy_dx, axis=0, edge_order=2), darcy_dx, axis=0, edge_order=2)
darcy_gradient_xxx = np.gradient((np.gradient(np.gradient(darcy[0], darcy_dx, axis=0, edge_order=2), darcy_dx, axis=0, edge_order=2)), darcy_dx, axis=0, edge_order=2)

plt.figure(figsize=(5, 4))
plt.imshow(darcy[0, :, :], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(figsize=(5, 4))
plt.imshow(darcy_gradient_nux, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data_nux')
plt.xlabel('x')
plt.ylabel('t')

plt.figure(figsize=(5, 4))
plt.imshow(darcy_gradient_nuy, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data_nuy')
plt.xlabel('x')
plt.ylabel('t')

plt.figure(figsize=(5, 4))
plt.imshow(darcy_gradient_x, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data_x')
plt.xlabel('x')
plt.ylabel('t')

plt.figure(figsize=(5, 4))
plt.imshow(darcy_gradient_xx, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data_xx')
plt.xlabel('x')
plt.ylabel('t')

plt.figure(figsize=(5, 4))
plt.imshow(darcy_gradient_xxx, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('data_xxx')
plt.xlabel('x')
plt.ylabel('t')
plt.show()

