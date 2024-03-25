"""Module for working with device mode"""

from typing import Any
import torch


def solver_device(device: str):
    """ Corresponding to chosen device, all futher
        created tensors will be with the same device

    Args:
        device (str): device mode, **cuda, gpu, cpu*.

    """
    if device in ['cuda','gpu'] and torch.cuda.is_available():
        print('CUDA is available and used.')
        return torch.set_default_device('cuda')
    elif device in ['cuda','gpu'] and not torch.cuda.is_available():
        print('CUDA is not available, cpu is used!')
        return torch.set_default_device('cpu')
    else:
        print('Default cpu processor is used.')
        return torch.set_default_device('cpu')

def check_device(data: Any):
    """ checking the device of the data.
        If the data.device is not same with torch.set_default_device,
        change one.
    Args:
        data (Any): it could be model or torch.Tensors

    Returns:
        data (Any): data with correct device
    """
    device = torch.tensor([0.]).device.type
    if data.device.type != device:
        return data.to(device)
    else:
        return data

def device_type():
    """ Return the default device.
    """
    return torch.tensor([0.]).device.type