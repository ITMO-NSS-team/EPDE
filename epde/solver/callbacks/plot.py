import os
import datetime
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from epde.solver.callbacks.callback import Callback


class Plots(Callback):
    """Class for ploting solutions."""
    def __init__(self,
                 print_every: Union[int, None] = 500,
                 save_every: Union[int, None] = 500,
                 title: str = None,
                 img_dir: str = None):
        """
        Args:
            print_every (Union[int, None], optional): print plots after every *print_every* steps. Defaults to 500.
            save_every (Union[int, None], optional): save plots after every *print_every* steps. Defaults to 500.
            title (str, optional): plots title. Defaults to None.
            img_dir (str, optional): directory title where plots are being saved. Defaults to None.
        """
        super().__init__()
        self.print_every = print_every if print_every is not None else 0.1
        self.save_every =  save_every if save_every is not None else 0.1
        self.title = title
        self.img_dir = img_dir

    def _print_nn(self):
        """
        Solution plot for *NN, autograd* mode.

        """

        try:
            nvars_model = self.net[-1].out_features
        except:
            nvars_model = self.net.model[-1].out_features

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
        Solution plot for mat mode.
        """

        nparams = self.grid.shape[0]
        nvars_model = self.net.shape[0]
        fig = plt.figure(figsize=(15,8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i+1)
                if self.title is not None:
                    ax1.set_title(self.title+' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')

                if self.title is not None:
                    ax1.set_title(self.title+' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                            self.grid[1].detach().cpu().numpy().reshape(-1),
                            self.net[i].detach().cpu().numpy().reshape(-1),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def _dir_path(self, save_dir: str) -> str:
        """ Path for save figures.

        Args:
            save_dir (str): directory where saves in

        Returns:
            str: directory where saves in
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
        """ printing or saving figures.
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
        self.solution_print()
