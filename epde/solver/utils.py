"""this one contain some stuff for computing different auxiliary things."""

from typing import Tuple, List, Union, Any
from torch.nn import Module
import datetime
import os
import shutil
import numpy as np
import torch
from epde.solver.device import check_device

def create_random_fn(eps: float) -> callable:
    """
    Creates a function to randomly perturb the weights and biases of linear and convolutional layers in a neural network.
    
    This randomization helps explore the model space during the evolutionary search for differential equations,
    introducing diversity and preventing premature convergence.
    
    Args:
        eps (float): The magnitude of the random perturbation applied to the weights and biases.
    
    Returns:
        callable: A function that, when applied to a PyTorch module, adds random noise to the weights and biases
                  of its linear and convolutional layers.
    """
    def randomize_params(m):
        if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)) and m.bias is not None:
            m.weight.data = m.weight.data + \
                            (2 * torch.randn(m.weight.size()) - 1) * eps
            m.bias.data = m.bias.data + (2 * torch.randn(m.bias.size()) - 1) * eps

    return randomize_params


def samples_count(second_order_interactions: bool,
                  sampling_N: int,
                  op_length: list,
                  bval_length: list) -> Tuple[int, int]:
    """
    Estimate the number of samples required for sensitivity analysis within the equation discovery process.
    
    This function calculates the sampling requirements based on the complexity of the search space,
    specifically the number of operators and boundary values, and whether second-order interactions
    are considered. This is crucial for efficiently exploring the space of possible differential
    equations and identifying those that best fit the data.
    
    Args:
        second_order_interactions (bool): Flag indicating whether to calculate second-order sensitivities,
            which increases the sampling requirements.
        sampling_N (int): A scaling factor that determines the base number of samples.  It influences
            how thoroughly the search space is explored.
        op_length (list): List containing the lengths of different operator sets used in the equation
            discovery process.
        bval_length (list): List containing the lengths of different boundary value sets.
    
    Returns:
        Tuple[int, int]: A tuple containing:
            - sampling_amount (int): The total number of samples required for the sensitivity analysis.
            - sampling_D (int): The sum of the lengths of the operator and boundary value sets,
              representing the dimensionality of the search space.
    """

    grid_len = sum(op_length)
    bval_len = sum(bval_length)

    sampling_D = grid_len + bval_len

    if second_order_interactions:
        sampling_amount = sampling_N * (2 * sampling_D + 2)
    else:
        sampling_amount = sampling_N * (sampling_D + 2)
    return sampling_amount, sampling_D


def lambda_print(lam: torch.Tensor, keys: List) -> None:
    """
    Prints the values of identified equation coefficients.
    
    This function displays the optimized coefficients (lambdas) 
    along with their corresponding equation terms, providing a 
    readable output of the discovered equation.
    
    Args:
        lam (torch.Tensor): A tensor containing the optimized coefficient values.
        keys (List): A list of strings, where each string represents the 
            corresponding term in the identified equation.
    
    Returns:
        None
    """

    lam = lam.reshape(-1)
    for val, key in zip(lam, keys):
        print('lambda_{}: {}'.format(key, val.item()))


def bcs_reshape(
    bval: torch.Tensor,
    true_bval: torch.Tensor,
    bval_length: List) -> Tuple[dict, dict, dict, dict]:
    """
    Reshapes the predicted and true boundary values into a format suitable for calculating boundary condition losses. This function prepares the boundary conditions for use in the equation discovery process.
    
        Args:
            bval (torch.Tensor): Predicted boundary values, where each column represents a different boundary type.
            true_bval (torch.Tensor): True (observed) boundary values, structured similarly to `bval`.
            bval_length (List): A list containing the length of each boundary type column in `bval` and `true_bval`.
    
        Returns:
            torch.Tensor: A concatenated vector representing the difference between predicted and true boundary values for all boundary types. This flattened vector is used to compute the loss associated with boundary condition satisfaction.
    """

    bval_diff = bval - true_bval

    bcs = torch.cat([bval_diff[0:bval_length[i], i].reshape(-1)
                                        for i in range(bval_diff.shape[-1])])

    return bcs


def remove_all_files(folder: str) -> None:
    """
    Remove all files and subdirectories from the specified folder.
    
    This function is used to clean up working directories before or after 
    equation discovery runs, ensuring a clean environment for each experiment.
    
    Args:
        folder (str): The path to the folder to be cleaned.
    
    Returns:
        None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mat_op_coeff(equation: Any) -> Any:
    """
    Reshapes coefficient tensors within the equation to ensure compatibility with downstream numerical methods and automatic differentiation.
    
    This function iterates through the terms of the equation and reshapes coefficient tensors to a column vector format (shape (-1, 1)).
    This reshaping is crucial for correct matrix operations and gradient calculations within the numerical solvers and automatic differentiation routines used by the EPDE framework.
    Callable coefficients are flagged with a warning, as they may interfere with caching mechanisms.
    
    Args:
        equation (Any): The equation object containing a list of operators and their terms.
    
    Returns:
        Any: The modified equation object with reshaped coefficient tensors.
    """

    for op in equation.equation_lst:
        for label in list(op.keys()):
            term = op[label]
            if isinstance(term['coeff'], torch.Tensor):
                term['coeff'] = term['coeff'].reshape(-1, 1)
            elif callable(term['coeff']):
                print("Warning: coefficient is callable,\
                                it may lead to wrong cache item choice")
    return equation


def model_mat(model: torch.Tensor,
                    domain: Any,
                    cache_model: torch.nn.Module=None) -> Tuple[torch.Tensor, torch.nn.Module]:
    """
    Creates a neural network model that approximates a pre-computed solution on a grid.
    
    This function takes a solution (computed using a matrix-based method) and a corresponding grid, and trains a neural network to approximate this solution. This allows for efficient evaluation of the solution at arbitrary points within the domain, as opposed to being limited to the grid points.
    
    Args:
        model (torch.Tensor): The pre-computed solution (e.g., from a matrix method).
        domain (Domain): The domain on which the solution is defined. Used to build the grid.
        cache_model (torch.nn.Module, optional): An existing neural network to refine. If None, a new network is created. Defaults to None.
    
    Returns:
        torch.nn.Module: A neural network that approximates the provided solution.
    """
    grid = domain.build('mat')
    input_model = grid.shape[0]
    output_model = model.shape[0]

    if cache_model is None:
        cache_model = torch.nn.Sequential(
            torch.nn.Linear(input_model, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, output_model)
        )

    return cache_model


def save_model_nn(
    cache_dir: str,
    model: torch.nn.Module,
    name: Union[str, None] = None) -> None:
    """
    Saves a trained neural network model to the specified cache directory. This allows for later reuse of the trained model, avoiding the need to retrain it from scratch.
    
        Args:
            cache_dir (str): The path to the directory where the model will be saved.
            model (torch.nn.Module): The trained neural network model to be saved.
            name (str, optional): A custom name for the saved model file. If None, a timestamp-based name will be generated. Defaults to None.
    
        Returns:
            None
    """

    if name is None:
        name = str(datetime.datetime.now().timestamp())
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    parameters_dict = {'model': model.to('cpu'),
                        'model_state_dict': model.state_dict()}

    try:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar')
        print(f'model is saved in cache dir: {cache_dir}')
    except RuntimeError:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar',
                    _use_new_zipfile_serialization=False)  # cyrillic in path
        print(f'model is saved in cache: {cache_dir}')
    except:
        print(f'Cannot save model in cache: {cache_dir}')


def save_model_mat(cache_dir: str,
                    model: torch.Tensor,
                    domain: Any,
                    cache_model: Union[torch.nn.Module, None] = None,
                    name: Union[str, None] = None) -> None:
    """
    Refines a coarse-grained solution by training a neural network to approximate it, and saves the trained network.
    
    This allows for efficient evaluation of the solution at arbitrary points within the domain,
    overcoming the limitations of the original coarse-grained representation.
    
    Args:
        cache_dir (str): Path to the directory where the trained neural network will be saved.
        model (torch.Tensor): The coarse-grained solution (e.g., from a numerical solver).
        domain (Any): The domain over which the solution is defined.
        cache_model (Union[torch.nn.Module, None], optional): An existing neural network to initialize the training from. Defaults to None.
        name (Union[str, None], optional): A name to use when saving the trained neural network. Defaults to None.
    
    Returns:
        None
    """

    net_autograd = model_mat(model, domain, cache_model)
    nn_grid = domain.build('autograd')
    optimizer = torch.optim.Adam(net_autograd.parameters(), lr=0.001)
    model_res = model.reshape(-1, model.shape[0])

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((net_autograd(check_device(nn_grid)) - model_res) ** 2)
        loss.backward()
        return loss

    loss = np.inf
    t = 0
    while loss > 1e-5 and t < 1e5:
        loss = optimizer.step(closure)
        t += 1
        print('Interpolate from trained model t={}, loss={}'.format(
                t, loss))

    save_model_nn(cache_dir, net_autograd, name=name)

def replace_none_by_zero(tuple_data: tuple) -> torch.Tensor:
    """
    Convert a tuple of data, replacing any None elements with zero tensors.
    
    This ensures that the data structure is compatible with downstream processing
    steps that require numerical tensors, particularly when constructing
    equation structures.
    
    Args:
        tuple_data (tuple | torch.Tensor | None): A tuple, tensor, or None representing the data.
    
    Returns:
        torch.Tensor | tuple: A tensor or tuple with None elements replaced by zero tensors.
    """
    if isinstance(tuple_data, torch.Tensor):
        tuple_data[tuple_data == None] = 0
    elif tuple_data is None:
        tuple_data = torch.tensor([0.])
    elif isinstance(tuple_data, tuple):
        new_tuple = tuple(replace_none_by_zero(item) for item in tuple_data)
        return new_tuple
    return tuple_data

class PadTransform(Module):
    """
    Pad tensor to a fixed length with given padding value.
    
        src: https://pytorch.org/text/stable/transforms.html#torchtext.transforms.PadTransform
    
        Done to avoid torchtext dependency (we need only this function).
    """


    def __init__(self, max_length: int, pad_value: int) -> None:
        """
        Pads sequences to a specified maximum length.
        
                This ensures consistent input sizes for subsequent processing steps,
                which is crucial for algorithms that require fixed-size inputs.
        
                Args:
                    max_length (int): Maximum length to pad to.
                    pad_value (int): Value to pad the tensor with.
        
                Returns:
                    None
        """
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads a tensor to a specified maximum length.
        
                This ensures that all input tensors have the same length, which is crucial
                for consistent processing, especially when dealing with variable-length
                sequences or signals.
        
                Args:
                    x (torch.Tensor): The input tensor to pad. The padding is applied along the last dimension.
        
                Returns:
                    torch.Tensor: The padded tensor. If the input tensor's length along the last dimension
                        is already equal to or greater than the maximum length, the original tensor is returned.
                        Otherwise, the tensor is padded with the specified `pad_value` until it reaches the
                        maximum length.
        """

        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x
