import numpy as np
from typing import Union
import torch
import datetime
from epde.solver.callbacks.callback import Callback


class InverseTask(Callback):
    """
    Class for printing the parameters during inverse task solution.
    """

    def __init__(self,
                 parameters: dict,
                 info_string_every: Union[int, None] = None):
        """
        Initializes the inverse problem with parameter guesses and logging frequency.
        
                Args:
                    parameters (dict): Dictionary containing the initial guesses for the parameters of the differential equation.
                    info_string_every (Union[int, None], optional): Frequency (in iterations) at which to print the current parameter values during the optimization process. If None, parameter values are not printed. Defaults to None.
        
                Returns:
                    None
        
                Why:
                    This initialization sets up the inverse task by storing the initial parameter guesses and the frequency at which to log parameter values during the optimization process. This allows the evolutionary algorithm to start with reasonable parameter values and provides a way to monitor the progress of the optimization.
        """
        super().__init__()
        self.parameters = parameters
        self.info_string_every = info_string_every
    
    def str_param(self):
        """
        Prints the values of specified network parameters to monitor their evolution during the training process. This helps in understanding how the parameters are being adjusted as the inverse problem is solved.
        
                Args:
                    self: The instance of the InverseTask class.
        
                Returns:
                    None. The method prints the parameter values to the console.
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
        """
        Called at the end of each epoch to perform string parameter updates.
        
        This ensures that the evolutionary process adapts based on the model's performance during training.
        
        Args:
            logs: The logs returned by the Keras model.
        
        Returns:
            None.
        """
        self.str_param()