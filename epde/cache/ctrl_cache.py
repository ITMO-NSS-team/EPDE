from typing import Union, List, Tuple

import torch

class ControlNNContainer():
    """
    A container class for managing a neural network, its arguments, and the device it runs on.
    
        Class Methods:
        - __init__: Initializes the ControlNNContainer with network arguments, a neural network, and a device.
        - create_shallow_nn: Creates a shallow neural network.
    
        Attributes:
            net_args: Stores the network arguments provided during initialization.
            net: Stores the neural network, either the provided one or a newly created shallow network.
    """

    def __init__(self, output_num: int = 1, args: List[Tuple[Union[int, List]]] = [(0, [None,]),], 
                 net: torch.nn.Sequential = None, device: str = 'cpu'):
        """
        Initializes the ControlNNContainer with network configuration, a neural network, and the computation device.
        
                This setup is crucial for managing and executing neural networks within the equation discovery process. The neural network architecture and its computational environment are configured to efficiently evaluate candidate equation structures.
        
                Args:
                    output_num: The number of output neurons in the network. Defaults to 1.
                    args: A list of tuples, where each tuple defines the input and any associated parameters for a network layer. Defaults to `[(0, [None,])]`.
                    net: A `torch.nn.Sequential` neural network. If `None`, a default shallow network is created based on the provided `args`. Defaults to `None`.
                    device: The device ('cpu' or 'cuda') on which the neural network will be executed. Defaults to 'cpu'.
        
                Returns:
                    None
        
                Initializes:
                    net_args (List[Tuple[Union[int, List]]]): Stores the network arguments for later use.
                    net (torch.nn.Sequential): Stores the initialized neural network, either the provided one or a newly created shallow network.
        """
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