from typing import Union, List, Tuple

import torch

class ControlNNContainer():
    def __init__(self, output_num: int = 1, args: List[Tuple[Union[int, List]]] = [(0, [None,]),], 
                 net: torch.nn.Sequential = None):
        self.net_args = args
        self.net = net if isinstance(net, torch.nn.Sequential) else self.create_shallow_nn(len(self.net_args), 
                                                                                           output_num)

    @staticmethod
    def create_shallow_nn(arg_num: int = 1, output_num: int = 1) -> torch.nn.Sequential: # net: torch.nn.Sequential = None, 
        hidden_neurons = 256
        layers = [torch.nn.Linear(arg_num, hidden_neurons),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_neurons, output_num)]
        return torch.nn.Sequential(*layers)