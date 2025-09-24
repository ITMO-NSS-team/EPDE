"""Module for working with device mode"""

from typing import Any
import torch

verbose = False

def solver_device(device: str):
    """
    Sets the default device for subsequent tensor operations.
    
    This function configures the global default device (CPU or CUDA)
    for all newly created tensors. It checks for CUDA availability
    if a CUDA device is requested and falls back to CPU if CUDA is
    unavailable.
    
    Args:
        device (str): The desired device ('cuda', 'gpu', or 'cpu').
    
    Returns:
        None: This function modifies the global PyTorch state.
    
    Why:
        This ensures consistency in device placement for all tensors
        created during the equation discovery process, which is
        crucial for efficient computation and compatibility with
        available hardware.
    """
    if device in ['cuda','gpu'] and torch.cuda.is_available():
        if verbose:
            print('CUDA is available and used.')
        return torch.set_default_device('cuda')
    elif device in ['cuda','gpu'] and not torch.cuda.is_available():
        if verbose:
            print('CUDA is not available, cpu is used!')
        return torch.set_default_device('cpu')
    else:
        if verbose:        
            print('Default cpu processor is used.')
        return torch.set_default_device('cpu')

def check_device(data: Any):
    """
    Ensures that the input data (model or tensor) resides on the correct device.
    
    This function checks if the device of the input data matches the currently
    configured default device. If they differ, the data is moved to the
    default device to ensure compatibility and proper execution within the
    framework. This is crucial for maintaining consistency across different
    hardware configurations (CPU/GPU) and preventing device-related errors
    during computations.
    
    Args:
        data (Any): The input data, which can be a model or a PyTorch tensor.
    
    Returns:
        Any: The input data, moved to the default device if necessary.
    """
    device = torch.tensor([0.]).device.type
    if data.device.type != device:
        return data.to(device)
    else:
        return data

def device_type():
    """
    Return the default device type used by PyTorch.
    
    This is useful for ensuring that operations are performed on the correct device,
    especially when working with hardware accelerators like GPUs.
    
    Args:
        None
    
    Returns:
        str: A string representing the device type (e.g., 'cpu' or 'cuda').
    """
    return torch.tensor([0.]).device.type