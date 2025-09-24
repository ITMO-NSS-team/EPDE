import os
import datetime
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from epde.solver.callbacks.callback import Callback
from mpl_toolkits.mplot3d import Axes3D


class Plots(Callback):
    """
    Class for ploting solutions.
    """


    def __init__(self,
                 print_every: Union[int, None] = 500,
                 save_every: Union[int, None] = 500,
                 title: str = None,
                 img_dir: str = None):
        """
        Initializes the plotting utility.
        
                The plotting utility provides functionality to visualize the identified equations during the search process.
                It allows to track the progress of the algorithm and assess the quality of the discovered equations.
                The frequency of printing and saving plots can be controlled to balance between real-time monitoring
                and storage overhead.
        
                Args:
                    print_every (Union[int, None], optional): Print plots after every *print_every* steps. Defaults to 500.
                    save_every (Union[int, None], optional): Save plots after every *save_every* steps. Defaults to 500.
                    title (str, optional): Title for the plots. Defaults to None.
                    img_dir (str, optional): Directory to save the plots. Defaults to None.
        """
        super().__init__()
        self.print_every = print_every if print_every is not None else 0.1
        self.save_every = save_every if save_every is not None else 0.1
        self.title = title
        self.img_dir = img_dir

    def _print_nn(self):
        """
        Generates a solution plot for neural network models, visualizing the learned function across the input grid. This is particularly useful in the context of equation discovery, as it allows to examine the behavior of the neural network solution and compare it with the underlying dynamics of the target equation.
        
                Args:
                    self: The Plots class instance, containing the trained neural network (self.net) and the input grid (self.grid).
        
                Returns:
                    None. Displays the generated plot using matplotlib.
        """

        attributes = {'model': ['out_features', 'output_dim', 'width_out'],
                      'layers': ['out_features', 'output_dim', 'width_out']}

        nvars_model = None

        for key, values in attributes.items():
            for value in values:
                try:
                    nvars_model = getattr(getattr(self.net, key)[-1], value)
                    break
                except AttributeError:
                    pass

        if nvars_model is None:
            try:
                nvars_model = self.net[-1].out_features
            except:
                nvars_model = self.net.width_out[-1]

        nparams = self.grid.shape[1]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net(self.grid)[:, i].detach().cpu().numpy())

            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))

                ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(),
                                 self.grid[:, 1].detach().cpu().numpy(),
                                 self.net(self.grid)[:, i].detach().cpu().numpy(),
                                 cmap=cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")

    def _print_mat(self):
        """
        Plots the solution surface or scatter plot for each variable of the identified differential equation, visualizing the relationship between the input grid and the model's output. This helps in understanding the behavior of the discovered equation across the input space.
        
                Args:
                    self (Plots): The Plots object containing the grid and network data.
        
                Returns:
                    None: Displays the plot using matplotlib.
        """

        nparams = self.grid.shape[0]
        nvars_model = self.net.shape[0]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')

                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                                 self.grid[1].detach().cpu().numpy().reshape(-1),
                                 self.net[i].detach().cpu().numpy().reshape(-1),
                                 cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def _dir_path(self, save_dir: str) -> str:
        """
        Generates a unique file path for saving a plot.
        
        If a `save_dir` is provided, the plot will be saved in that directory with a timestamped filename.
        Otherwise, the plot will be saved in a default 'img' directory within the project, also with a timestamped filename.
        This ensures that each generated plot has a unique and easily identifiable file path.
        
        Args:
            save_dir (str, optional): The directory where the plot should be saved.
                If None, the plot is saved to the default 'img' directory.
        
        Returns:
            str: The absolute path to the generated plot file.
        """

        if save_dir is None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                                     str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + '.png')
        return directory

    def solution_print(
            self):
        """
        Generates and displays or saves visualizations of the solution at specified intervals.
        
                The method checks if the current time step aligns with the printing or saving frequency.
                If so, it generates a plot of the solution (either from the matrix or neural network representation)
                and either saves it to a file or displays it on the screen, or both.
                This functionality allows to track the solution's evolution during the modeling process.
        
                Args:
                    self:  A reference to the Plots class instance.
        
                Returns:
                    None
        """
        print_flag = self.model.t % self.print_every == 0
        save_flag = self.model.t % self.save_every == 0

        if print_flag or save_flag:
            self.net = self.model.net
            self.grid = self.model.solution_cls.grid
            if self.model.mode == 'mat':
                self._print_mat()
            else:
                self._print_nn()
            if save_flag:
                directory = self._dir_path(self.img_dir)
                plt.savefig(directory)
            if print_flag:
                plt.show()
            plt.close()

    def on_epoch_end(self, logs=None):
        """
        Called at the end of an epoch to trigger the printing of the current best solution.
        
        This allows for monitoring the progress of the evolutionary search for the differential equation during training.
        
        Args:
            logs: The logs returned by the Keras model.
        
        Returns:
            None.
        """
        self.solution_print()
