#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from dataclasses import dataclass
import warnings
from typing import List, Union

import numpy as np
import torch
# device = torch.device('cpu') # TODO: make system-agnostic approach

from epde.cache.cache import Cache
from epde.cache.ctrl_cache import ControlNNContainer
from epde.supplementary import create_solution_net, AutogradDeriv
from epde.preprocessing.smoothers import NN


def init_caches(set_grids: bool = False, device = 'cpu'):
    """
    Initializes global caches for storing tensors, initial data, and grid information.
    
    These caches are used to store and manage intermediate calculations and data
    during the equation discovery process, improving efficiency by avoiding redundant computations.
    The grid cache is optional and is used when working with data defined on a grid.
    
    Args:
        set_grids (bool, optional): A flag indicating whether to initialize the grid cache.
            Defaults to False.
        device (str, optional): Device to store tensors. Defaults to 'cpu'.
    
    Returns:
        None
    """
    global tensor_cache, grid_cache, initial_data_cache
    tensor_cache = Cache(device = device)
    initial_data_cache = Cache(device = device)
    if set_grids:
        grid_cache = Cache(device = device)
    else:
        grid_cache = None


def set_time_axis(axis: int):
    """
    Sets the global time axis for data processing.
    
    This ensures that all subsequent operations within the framework 
    correctly interpret the temporal dimension of the input data.
    
    Args:
        axis (int): The index of the axis representing time.
    
    Returns:
        None
    
    Why:
        The time axis is a crucial parameter for various data processing 
        steps, such as calculating derivatives or integrating solutions. 
        Setting it globally ensures consistency across different modules 
        and functions within the framework.
    """
    global time_axis
    time_axis = axis


def init_eq_search_operator(operator):
    """
    Initializes the global equality search operator.
    
    This method sets the global variable `eq_search_operator` to the provided operator. This operator will be used to evaluate the equality of equation structures during the evolutionary search process.
    
    Args:
        operator: The operator to be used for equality searches.
    
    Returns:
        None.
    """
    global eq_search_operator
    eq_search_operator = operator


def init_sys_search_operator(operator):
    """
    Initializes the global system search operator.
    
    This method sets the global variable `sys_search_operator` to the
    provided operator. This global variable will be used by the evolutionary algorithm
    to explore the search space of possible differential equations. By initializing this operator,
    the system ensures that all search processes utilize a consistent and pre-defined search strategy
    for discovering equation structures.
    
    Args:
        operator: The search operator to be used by the system.
    
    Returns:
        None.
    """
    global sys_search_operator
    sys_search_operator = operator


def delete_cache():
    """
    Deletes the global tensor and grid caches used for storing intermediate calculations.
    
        This method removes the global variables `tensor_cache` and `grid_cache` to
        free up memory and ensure a clean state for subsequent equation discoveries.
        If either cache does not exist, a message is printed, indicating that it was
        not previously initialized. This is done to manage memory effectively during
        the equation discovery process, especially when dealing with large datasets
        or complex equation structures.
    
        Args:
            None
    
        Returns:
            None
    """
    global tensor_cache, grid_cache
    try:
        del tensor_cache
    except NameError:
        print('Failed to delete tensor cache due to its inexistance')
    try:
        del grid_cache
    except NameError:
        print('Failed to delete grid cache due to its inexistance')


class TrainHistory(object):
    """
    A class to store the history of training metrics.
    
        Attributes:
            history (list): A list to store historical data.
            _idx (int): An index to track the current position.
    """

    def __init__(self):
        """
        Initializes a new TrainHistory instance.
        
        The TrainHistory class stores the equation structures, losses, and other relevant information during the training process.
        This initialization prepares the object to record the evolutionary search for differential equations.
        
        Args:
            None
        
        Returns:
            None
        """
        self.reset()
        
    def reset(self):
        """
        Resets the training history and index to their initial states.
        
                This method clears the stored training history and sets the current index to zero,
                effectively starting a new training cycle. This is useful when restarting or
                re-initializing the training process to ensure that previous training data does
                not influence the new run.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self.history = []
        self._idx = 0

    def add(self, element):
        """
        Adds a new element to the training history, associating it with a unique index. This is crucial for tracking the evolution of solutions during the equation discovery process, allowing to analyze how different equation structures perform over time and to identify promising candidates for further refinement.
        
                Args:
                    element: The element to add to the history (e.g., an equation, a set of parameters, or a performance metric).
        
                Returns:
                    None
        """
        self.history.append((self._idx, element))
        self._idx += 1

def reset_hist():
    """
    Resets the global training history.
    
    This method reinitializes the global 'history' variable with a new
    instance of the TrainHistory class, effectively clearing the
    previous training history. This is crucial for starting a new equation discovery run
    without being influenced by the results of previous runs.
    
    Args:
        None
    
    Returns:
        None
    """
    global history
    history = TrainHistory()

@dataclass
class VerboseManager:
    """
    Manager for output in text form
    """

    plot_DE_solutions : bool
    show_iter_idx : bool
    iter_fitness : bool
    iter_stats : bool
    show_ann_loss : bool    
    show_warnings : bool
    
def init_verbose(plot_DE_solutions : bool = False, show_iter_idx : bool = True, 
                 show_iter_fitness : bool = False, show_iter_stats : bool = False, 
                 show_ann_loss : bool = False, show_warnings : bool = False):
    """
    Initializes the verbosity settings for the evolutionary algorithm's output.
    
    This configuration controls what information is displayed during the equation discovery process, 
    allowing users to monitor progress and diagnose potential issues. By adjusting these flags, 
    users can tailor the output to focus on specific aspects of the search, such as iteration number, 
    fitness, or statistical properties of the population. This helps in understanding how the algorithm 
    is exploring the search space and identifying promising equation structures.
    
    Args:
        plot_DE_solutions (`bool`): optional
            Display solutions of a differential equation, default - False. Useful for visualizing the behavior of discovered equations.
        show_iter_idx (`bool`): optional
            Display the index of each iteration of the evolutionary algorithm, default - True. Helps track the progress of the search.
        show_iter_fitness (`bool`): optional
            Display the fitness of each iteration of the evolutionary algorithm, default - False.  Provides insight into the quality of solutions found over time.
        show_iter_stats (`bool`): optional
            Display statistical properties of the population in each iteration of the evolutionary algorithm, default - False.  Allows monitoring of population diversity and convergence.
        show_ann_loss (`bool`): optional
            Display loss of approximation by artificial neural network, default - False.
        show_warnings (`bool`): optional
            Display warnings arising during the operation of the algorithm, default - False.  Can help identify potential issues or unexpected behavior.
    
    Returns:
        `None`
    """
    global verbose
    if not show_warnings:
        warnings.filterwarnings("ignore")
    verbose = VerboseManager(plot_DE_solutions, show_iter_idx, show_iter_fitness, 
                             show_iter_stats, show_ann_loss, show_warnings)

def reset_control_nn(n_control: int = 1, ann: torch.nn.Sequential = None, 
                     ctrl_args: list = [(0, [None,]),], device: str = 'cpu'):
    """
    Resets the global control neural network (`control_nn`) with a new configuration. This is used to adjust the parameters and architecture of the neural network responsible for generating control signals within the EPDE framework. By resetting the control network, the search process can explore different control strategies and adapt to the evolving equation structures.
    
        Args:
            n_control (int, optional): The number of control signals to generate. Defaults to 1.
            ann (torch.nn.Sequential, optional): A custom neural network to use as the control network. If None, a default network is created. Defaults to None.
            ctrl_args (list, optional): Arguments for the control network. Defaults to `[(0, [None,])]`.
            device (str, optional): The device (e.g., 'cpu', 'cuda') on which to run the control network. Defaults to 'cpu'.
    
        Returns:
            None. The function modifies the global `control_nn` variable.
    """

    global control_nn
    control_nn = ControlNNContainer(output_num = n_control, args = ctrl_args,
                                    net = ann, device = device)


def reset_data_repr_nn(data: List[np.ndarray], grids: List[np.ndarray], train: bool = True,
                       derivs: List[Union[int, List, Union[np.ndarray]]] = None, 
                       penalised_derivs: List[Union[int, List]] = None,
                       epochs_max=1e3, predefined_ann: torch.nn.Sequential = None,
                       batch_frac=0.5, learining_rate=1e-6, device = 'cpu',
                       use_fourier: bool = True, fourier_params: dict = {'L' : [4,], 'M' : [3,]}): 
    """
    Represent the data with an artificial neural network (ANN) to provide an informed initial guess for candidate equation solutions during the equation search. This is achieved by training the ANN to approximate the provided data, enabling a more efficient and accurate exploration of the solution space.
    
        The ANN learns a continuous representation of the data, which can then be used to evaluate the fitness of candidate equations using solver-based methods.
    
        Args:
            data (List[np.ndarray]): A list of numpy arrays representing the data to be approximated.
            grids (List[np.ndarray]): A list of numpy arrays representing the grid coordinates corresponding to the data.
            train (bool, optional): Whether to train the ANN. If False, a predefined ANN must be provided. Defaults to True.
            derivs (List[Union[int, List, Union[np.ndarray]]], optional): List of derivatives to be included in the loss function. Defaults to None.
            penalised_derivs (List[Union[int, List]], optional): List of derivatives to be penalised in the loss function. Defaults to None.
            epochs_max (float, optional): The maximum number of training epochs. Defaults to 1e3.
            predefined_ann (torch.nn.Sequential, optional): A pre-defined ANN to use instead of creating a new one. Defaults to None.
            batch_frac (float, optional): The fraction of data to use in each batch during training. Defaults to 0.5.
            learining_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-6.
            device (str, optional): The device to use for training (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            use_fourier (bool, optional): Whether to use Fourier features in the input layer of the ANN. Defaults to True.
            fourier_params (dict, optional): Parameters for the Fourier features. Defaults to {'L' : [4,], 'M' : [3,]}.
    
        Returns:
            None. The trained ANN is stored in the global variable `solution_guess_nn`.
    """

    global solution_guess_nn

    if predefined_ann is None:
        model = create_solution_net(equations_num=len(data), domain_dim=len(grids), device = device,
                                    use_fourier=use_fourier, fourier_params=fourier_params)
        # model = NN(Num_Hidden_Layers=5, Neurons_Per_Layer=50, Input_Dim=len(grids), Activation_Function='Tanh')

    else:
        model = predefined_ann

    if train:
        model.to(device)

        grids_tr = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T
        data_tr = torch.from_numpy(np.array([data_var.reshape(-1) for data_var in data])).float().T
        grids_tr = grids_tr.to(device)
        data_tr = data_tr.to(device)

        batch_size = int(data[0].size * batch_frac)
        optimizer = torch.optim.Adam(model.parameters(), lr = learining_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.5)
        deriv_calc = AutogradDeriv()

        t = 0
        min_loss = np.inf
        loss_mean = np.inf
        print(f'Training NN to represent data for {epochs_max} epochs')
        while loss_mean > 1e-6 and t < epochs_max:

            permutation = torch.randperm(grids_tr.size()[0])

            loss_list = []

            for i in range(0, grids_tr.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = grids_tr[indices], data_tr[indices]

                # print(f'batch_y {batch_y.get_device()}, batch_x {batch_x.get_device()},, {next(model.parameters()).is_cuda}')
                # print(f'model(batch_x) {model(batch_x)}') 
                loss = torch.mean(torch.abs(batch_y - model(batch_x)))
                if derivs is not None:
                    for var_idx, deriv_axes, deriv_tensor in derivs:
                        deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes = deriv_axes, component = var_idx)
                        batch_derivs = torch.from_numpy(deriv_tensor)[torch.unravel_index(indices, 
                                                                                          deriv_tensor.shape)].reshape_as(deriv_autograd).to(device)
                        
                        loss_add = 1e2 * torch.mean(torch.abs(batch_derivs - deriv_autograd))
                        # print(loss, loss_add)
                        loss += loss_add

                if penalised_derivs is not None:
                    for var_idx, deriv_axes in derivs:
                        deriv_autograd = deriv_calc.take_derivative(model, batch_x, axes = deriv_axes, component = var_idx)
                        batch_derivs = torch.from_numpy(deriv_tensor)[torch.unravel_index(indices, 
                                                                                          deriv_tensor.shape)].reshape_as(deriv_autograd).to(device)
                        higher_ord_penalty = 1e3 * torch.mean(torch.abs(deriv_autograd))

                        loss += higher_ord_penalty

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            scheduler.step()
            loss_mean = np.mean(loss_list)
            if loss_mean < min_loss:
                best_model = model
                min_loss = loss_mean
            t += 1
        model = best_model
        print(f'min loss is {min_loss}, in last epoch: {loss_list}, ')
        solution_guess_nn = best_model
    else:
        solution_guess_nn = model
