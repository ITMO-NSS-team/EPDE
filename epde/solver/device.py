import torch

def solver_device(device):
    if device in ['cuda','gpu'] and torch.cuda.is_available():
        print('CUDA is available and used.')
        return torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif device in ['cuda','gpu'] and not torch.cuda.is_available():
        print('CUDA is not available, cpu is used!')
        return torch.set_default_tensor_type('torch.FloatTensor')
    else:
        print('Default cpu processor is used.')
        return torch.set_default_tensor_type('torch.FloatTensor')

def check_device(data):
    device = torch.tensor([0.]).device.type
    if data.device.type != device:
        return data.to(device)
    else:
        return data

def device_type():
    return torch.tensor([0.]).device.type