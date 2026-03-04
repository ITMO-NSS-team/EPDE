import torch
from epde.solver.callbacks.callback import Callback
import os



class SaveModel(Callback):
    """Class for saving model during train
    """
    def __init__(self,
                 path_to_folder: str,
                 every_step : int = 1):
        """
        Args:
            path (str): path_to_folder to save model 
            every_step (int): save model every n steps. Defaults 1.
        """
        super().__init__()
        self.path_to_folder = path_to_folder
        self.every_step = every_step
    
    def save_model(self):
        model_name = "model-{}.pt".format(self.model.t-1)
        save_path = os.path.join(self.path_to_folder, model_name)
        torch.save(self.model.net, save_path)

    def on_epoch_end(self, logs=None):
        if (self.model.t-1) % self.every_step == 0:
            self.save_model()
