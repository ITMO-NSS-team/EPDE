import numpy as np
from typing import Union
import torch
import datetime
from epde.solver.callbacks.callback import Callback
from epde.solver.utils import create_random_fn


class EarlyStopping(Callback):
    """
    Class for using adaptive stop criterias at training process.
    """

    def __init__(self,
                 eps: float = 1e-5,
                 loss_window: int = 100,
                 no_improvement_patience: int = 1000,
                 patience: int = 5,
                 abs_loss: Union[float, None] = None,
                 normalized_loss: bool = False,
                 randomize_parameter: float = 1e-5,
                 info_string_every: Union[int, None] = None,
                 verbose: bool = True,
                 save_best: bool = False
                 ):
        """
        Initializes the EarlyStopping callback.
        
                This callback monitors the loss function during training and stops the training process when the loss stops improving,
                potentially saving the best model encountered during training. This prevents overfitting and saves computational resources
                by avoiding unnecessary iterations.
        
                Args:
                    eps (float, optional): A small threshold to consider an improvement in loss. Defaults to 1e-5.
                    loss_window (int, optional): The number of recent losses to average for trend estimation. Defaults to 100.
                    no_improvement_patience (int, optional): How many iterations to wait after the best loss before stopping. Defaults to 1000.
                    patience (int, optional): How many times the `no_improvement_patience` can be reached before stopping. Defaults to 5.
                    abs_loss (Union[float, None], optional): An absolute target loss value. Training stops if this loss is reached. Defaults to None.
                    normalized_loss (bool, optional): Whether to use a normalized loss function (all lambdas=1) for early stopping. Defaults to False.
                    randomize_parameter (float, optional): A small value to randomize model weights to escape local optima. Defaults to 1e-5.
                    info_string_every (Union[int, None], optional): How often (in iterations) to print the current loss and early stopping status. Defaults to None.
                    verbose (bool, optional): Whether to print information about the loss and early stopping status. Defaults to True.
                    save_best (bool, optional): Whether to save the model with the best loss encountered during training. Defaults to False.
        """
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.patience = patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self._stop_dings = 0
        self._t_imp_start = 0
        self._r = create_random_fn(randomize_parameter)
        self.info_string_every = info_string_every if info_string_every is not None else np.inf
        self.verbose = verbose
        self.save_best=save_best
        self.best_model=None



    def _line_create(self):
        """
        Approximates the trend of recent loss values using linear regression.
        
        This helps to identify whether the loss is consistently decreasing, increasing, or oscillating,
        which is used to determine when to stop training to prevent overfitting.
        
        Args:
            None
        
        Returns:
            None
        """
        self._line = np.polyfit(range(self.loss_window), self.last_loss, 1)

    def _window_check(self):
        """
        Checks for early stopping based on the trend of the loss function within a window.
        
        This method assesses whether the rate of change of the loss, normalized by the current loss value,
        falls below a specified threshold (*eps*). If this condition is met, it indicates that the training
        is no longer significantly improving the model's performance and may be converging.
        
        Args:
            None
        
        Returns:
            None
        """
        if self.t % self.loss_window == 0 and self._check is None:
            self._line_create()
            if abs(self._line[0] / self.model.cur_loss) < self.eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.net.apply(self._r)
                self._check = 'window_check'

    def _patience_check(self):
        """
        Checks if the training should be stopped based on the patience criterion.
        
        The patience mechanism monitors the training loss and stops the training process if the loss
        does not improve for a specified number of epochs. This helps to prevent overfitting and
        improve the generalization ability of the discovered equations. The method checks if the
        number of epochs since the last improvement exceeds the `no_improvement_patience` threshold.
        If it does, and a check hasn't already been triggered, it increments the stop counter, resets
        the improvement start time, and potentially restores the best model found so far.
        
        Args:
            None
        
        Returns:
            None
        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._stop_dings += 1
            self._t_imp_start = self.t
            if self.mode in ('NN', 'autograd'):
                if self.save_best:
                    self.model.net=self.best_model
                self.model.net.apply(self._r)
            self._check = 'patience_check'

    def _absloss_check(self):
        """
        Check if the absolute loss is below the specified threshold.
        
        This check is part of the early stopping mechanism, which aims to prevent overfitting by monitoring the loss function and stopping the training process when the loss reaches a satisfactory level.
        The training stops to avoid unnecessary computations when the model performance, measured by the loss function, is already good enough.
        
        Args:
            self (EarlyStopping): The EarlyStopping instance.
        
        Returns:
            None
        """
        if self.abs_loss is not None and self.model.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def verbose_print(self):
        """
        Prints information about the current loss, stopping criteria, and normalized loss line to monitor the training process and understand the behavior of the equation discovery.
        
                Args:
                    None
        
                Returns:
                    None
        """

        if self._check == 'window_check':
            print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
        elif self._check == 'patience_check':
            print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), self.no_improvement_patience))
        elif self._check == 'absloss_check':
            print('[{}] Absolute value of loss is lower than threshold'.format(
                                                        datetime.datetime.now()))

        if self._check is not None or self.t % self.info_string_every == 0:
            try:
                self._line
            except:
                self._line_create()
            loss = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else self.mdoel.cur_loss
            info = '[{}] Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    datetime.datetime.now(), self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
            print(info)

    def on_epoch_end(self, logs=None):
        """
        Handles end-of-epoch actions, focusing on early stopping to optimize equation discovery.
        
                This method orchestrates checks for early stopping based on loss trends,
                patience criteria, and absolute loss thresholds. It saves the best model
                encountered so far and provides verbose output for monitoring progress.
                The goal is to efficiently identify the most promising equation structures
                during the evolutionary search process.
        
                Args:
                    logs: Metric results for the current epoch, used to evaluate model performance.
        
                Returns:
                    None
        """
        self._window_check()
        self._patience_check()
        self._absloss_check()

        if self.model.cur_loss < self.model.min_loss:
            self.model.min_loss = self.model.cur_loss
            if self.save_best:
                self.best_model=self.model.net
            self._t_imp_start = self.t

        if self.verbose:
            self.verbose_print()
        if self._stop_dings >= self.patience:
            self.model.stop_training = True
            if self.save_best:
                self.model.net=self.best_model
        self._check = None

    def on_epoch_begin(self, logs=None):
        """
        Updates the callback's internal state at the start of each epoch.
        
                This method retrieves the current training step, mode, and check flag from the model to track the training progress.
                It also maintains a history of recent loss values, which is used to determine when early stopping criteria are met.
                By tracking these values, the callback can make informed decisions about when to terminate training,
                preventing overfitting and saving computational resources.
        
                Args:
                    logs: Optional dictionary of logs. Will be passed to all callbacks.
        
                Returns:
                    None
        """
        self.t = self.model.t
        self.mode = self.model.mode
        self._check = self.model._check
        try:
            self.last_loss[(self.t - 3) % self.loss_window] = self.model.cur_loss
        except:
            self.last_loss = np.zeros(self.loss_window) + float(self.model.min_loss)
