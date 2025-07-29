import numpy as np
from typing import Union
import torch
import datetime
from epde.solver.callbacks.callback import Callback


class InverseTask(Callback):
    """Class for printing the parameters during inverse task solution.
    """
    def __init__(self,
                 parameters: dict,
                 info_string_every: Union[int, None] = None):
        """
        Args:
            parameters (dict): dictioanry with initial guess of parameters.
            info_string_every (Union[int, None], optional): print parameters after every *int* step. Defaults to None.
        """
        super().__init__()
        self.parameters = parameters
        self.info_string_every = info_string_every
    
    def str_param(self):
        """ printing the inverse parameters.
        """
        if self.info_string_every is not None and self.model.t % self.info_string_every == 0:
            param = list(self.parameters.keys())
            for name, p in self.model.net.named_parameters():
                if name in param:
                    try:
                        param_str += name + '=' + str(p.item()) + ' '
                    except:
                        param_str = name + '=' + str(p.item()) + ' '
            print(param_str)
    
    def on_epoch_end(self, logs=None):
        self.str_param()