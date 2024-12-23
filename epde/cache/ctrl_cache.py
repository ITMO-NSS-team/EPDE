from typing import Union, List, Tuple

import torch

class ControlNNContainer():
    def __init__(self, output_num: int = 1, args: List[Tuple[Union[int, List]]] = [(0, [None,]),], 
                 net: torch.nn.Sequential = None, device: str = 'cpu'):
        self.net_args = args
        self.net = net if isinstance(net, torch.nn.Sequential) else self.create_shallow_nn(len(self.net_args), 
                                                                                           output_num, device)

    @staticmethod
    def create_shallow_nn(arg_num: int = 1, output_num: int = 1, 
                          device: str = 'cpu') -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
        hidden_neurons = 256
        layers = [torch.nn.Linear(arg_num, hidden_neurons, device=device),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_neurons, output_num, device=device)]
        control_nn = torch.nn.Sequential(*layers)
        control_nn.to(device)
        print('control_nn', control_nn, next(control_nn.parameters()).device, 'should be ', device)
        return control_nn