# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""

import glob
from typing import Union
import torch
import numpy as np
from copy import deepcopy
import tempfile
import os

from epde.solver.device import device_type
from epde.solver.callbacks.callback import Callback
from epde.solver.utils import create_random_fn, mat_op_coeff, model_mat, remove_all_files
from epde.solver.model import Model


def count_output(model: torch.Tensor) -> int:
    """ Determine the out features of the model.

    Args:
        model (torch.Tensor): torch neural network.

    Returns:
        int: number of out features.
    """
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


class CachePreprocessing:
    """class for preprocessing cache files.
    """
    def __init__(self,
                 model: Model
                 ):
        """
        Args:
            model (Model): object of Model class
        """
        self.solution_cls = model.solution_cls

    @staticmethod
    def _cache_files(files: list, nmodels: Union[int, None]=None) -> np.ndarray:
        """ At some point we may want to reduce the number of models that are
            checked for the best in the cache.

        Args:
            files (list): list with all model names in cache.
            nmodels (Union[int, None], optional): models quantity for checking. Defaults to None.

        Returns:
            cache_n (np.ndarray): array with random cache files names.
        """

        if nmodels is None:
            # here we take all files that are in cache
            cache_n = np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n = np.random.choice(len(files), nmodels, replace=False)

        return cache_n

    @staticmethod
    def _model_reform(init_model: Union[torch.nn.Sequential, torch.nn.ModuleList],
                     model: Union[torch.nn.Sequential, torch.nn.ModuleList]):
        """
        As some models are nn.Sequential class objects,
        but another models are nn.Module class objects.
        This method does checking the solver model (init_model)
        and the cache model (model).
        Args:
            init_model (nn.Sequential or nn.ModuleList): solver model.
            model (nn.Sequential or nn.ModuleList): cache model.
        Returns:
            init_model (nn.Sequential or nn.ModuleList): checked init_model.
            model (nn.Sequential or nn.ModuleList): checked model.
        """
        try:
            model[0]
        except:
            model = model.model

        try:
            init_model[0]
        except:
            init_model = init_model.model

        return init_model, model

    def cache_lookup(self,
                     cache_dir: str,
                     nmodels: Union[int, None] = None,
                     save_graph: bool = False,
                     cache_verbose: bool = False) -> Union[None, dict, torch.nn.Module]:
        """Looking for the best model (min loss) model from the cache files.

        Args:
            cache_dir (str): folder where system looks for cached models.
            nmodels (Union[int, None], optional): maximal number of models that are taken from cache dir. Defaults to None.
            save_graph (bool, optional): responsible for saving the computational graph. Defaults to False.
            cache_verbose (bool, optional): verbose cache operations. Defaults to False.

        Returns:
            Union[None, dict, torch.Tensor]: best model with optimizator state.
        """

        files = glob.glob(cache_dir + '\*.tar')

        if cache_verbose:
            print(f"The CACHE will be searched among the models in the folder {cache_dir}.")

        if len(files) == 0:
            best_checkpoint = None
            return best_checkpoint

        cache_n = self._cache_files(files, nmodels)
        
        min_loss = np.inf
        best_checkpoint = {}

        device = device_type()

        initial_model = self.solution_cls.model

        for i in cache_n:
            file = files[i]
            
            try:
                checkpoint = torch.load(file)
            except Exception:
                if cache_verbose:
                    print('Error loading file {}'.format(file))
                continue

            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])

            # this one for the input shape fix if needed

            try:
                solver_model, cache_model = self._model_reform(self.solution_cls.model, model)
            except Exception:
                if cache_verbose:
                    print('Error reforming file {}'.format(file))
                continue

            if cache_model[0].in_features != solver_model[0].in_features:
                continue
            try:
                if count_output(model) != count_output(self.solution_cls.model):
                    continue
            except Exception:
                continue

            model = model.to(device)
            self.solution_cls._model_change(model)
            loss, _ = self.solution_cls.evaluate(save_graph=save_graph)

            if loss < min_loss:
                min_loss = loss
                best_checkpoint['model'] = model
                best_checkpoint['model_state_dict'] = model.state_dict()
                if cache_verbose:
                    print('best_model_num={} , loss={}'.format(i, min_loss.item()))

            self.solution_cls._model_change(initial_model)

        if best_checkpoint == {}:
            best_checkpoint = None

        return best_checkpoint

    def scheme_interp(self,
                      trained_model: torch.nn.Module,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ If the cache model has another arcitechure to user's model,
            we will not be able to use it. So we train user's model on the
            outputs of cache model.

        Args:
            trained_model (torch.nn.Module): the best model (min loss) from cache.
            cache_verbose (bool, optional): verbose on/off of cache operations. Defaults to False.

        """

        grid = self.solution_cls.grid

        model = self.solution_cls.model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(
            trained_model(grid) - model(grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((trained_model(grid) - model(grid)) ** 2)
            loss.backward()
            return loss

        t = 0
        while loss > 1e-5 and t < 1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(
                trained_model(grid) - model(grid)))
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))
        
        self.solution_cls._model_change(model)

    def cache_retrain(self,
                      cache_checkpoint: dict,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ The comparison of the user's model and cache model architecture.
            If they are same, we will use model from cache. In the other case
            we use interpolation (scheme_interp method)

        Args:
            cache_checkpoint (dict): checkpoint of the cache model
            cache_verbose (bool, optional): on/off printing cache operations. Defaults to False.

        """

        model = self.solution_cls.model

        # do nothing if cache is empty
        if cache_checkpoint is None:
            return None
        # if models have the same structure use the cache model state,
        # and the cache model has ordinary structure
        if str(cache_checkpoint['model']) == str(model) and \
                isinstance(model, torch.nn.Sequential) and \
                isinstance(model[0], torch.nn.Linear):
            model = cache_checkpoint['model']
            model.load_state_dict(cache_checkpoint['model_state_dict'])
            model.train()
            self.solution_cls._model_change(model)
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model
        else:
            cache_model = cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.scheme_interp(
                cache_model, cache_verbose=cache_verbose)


class Cache(Callback):
    """
    Prepares user's model. Serves for computing acceleration.\n
    Saves the trained model to the cache, and subsequently it is possible to use pre-trained model
    (if it saved and if the new model is structurally similar) to sped up computing.\n
    If there isn't pre-trained model in cache, the training process will start from the beginning.
    """

    def __init__(self,
                 nmodels: Union[int, None] = None,
                 cache_dir: str = 'tedeous_cache',
                 cache_verbose: bool = False,
                 cache_model: Union[torch.nn.Sequential, None] = None,
                 model_randomize_parameter: Union[int, float] = 0,
                 clear_cache: bool = False
                ):
        """
        Args:
            nmodels (Union[int, None], optional): maximal number of models that are taken from cache dir. Defaults to None. Defaults to None.
            cache_dir (str, optional): directory with cached models. Defaults to '../tedeous_cache/' in temporary directoy of user system.
                If cache_dir is custom, then file will be searched in *torch_de_solver* directory.
            cache_verbose (bool, optional): printing cache operations. Defaults to False.
            cache_model (Union[torch.nn.Sequential, None], optional): model for mat method, which will be saved in cache. Defaults to None.
            model_randomize_parameter (Union[int, float], optional): creates a random model parameters (weights, biases)
                multiplied with a given randomize parameter.. Defaults to 0.
            clear_cache (bool, optional): clear cache directory. Defaults to False.
        """

        self.nmodels = nmodels
        self.cache_verbose = cache_verbose
        self.cache_model = cache_model
        self.model_randomize_parameter = model_randomize_parameter
        if cache_dir == 'tedeous_cache':
            temp_dir = tempfile.gettempdir()
            folder_path = os.path.join(temp_dir, 'tedeous_cache/')
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                pass
            else:
                os.makedirs(folder_path)
            self.cache_dir = folder_path
        else:
            try:
                file = __file__
            except:
                file = os.getcwd()
            self.cache_dir = os.path.normpath((os.path.join(os.path.dirname(file), '..', '..', cache_dir)))
        if clear_cache:
            remove_all_files(self.cache_dir)

    def _cache_nn(self):
        """  take model from cache as initial guess for *NN, autograd* modes.
        """

        cache_preproc = CachePreprocessing(self.model)

        r = create_random_fn(self.model_randomize_parameter)

        cache_checkpoint = cache_preproc.cache_lookup(cache_dir=self.cache_dir,
                                                      nmodels=self.nmodels,
                                                      cache_verbose=self.cache_verbose)

        cache_preproc.cache_retrain(cache_checkpoint,
                                               cache_verbose=self.cache_verbose)
        self.model.solution_cls.model.apply(r)

    def _cache_mat(self) -> torch.Tensor:
        """  take model from cache as initial guess for *mat* mode.
        """

        net = self.model.net
        domain = self.model.domain
        equation = mat_op_coeff(deepcopy(self.model.equation))
        conditions = self.model.conditions
        lambda_operator = self.model.lambda_operator
        lambda_bound = self.model.lambda_bound
        weak_form = self.model.weak_form

        net_autograd = model_mat(net, domain)

        autograd_model = Model(net_autograd, domain, equation, conditions)

        autograd_model.compile('autograd', lambda_operator, lambda_bound, weak_form=weak_form)

        r = create_random_fn(self.model_randomize_parameter)

        cache_preproc = CachePreprocessing(autograd_model)

        cache_checkpoint = cache_preproc.cache_lookup(
            cache_dir=self.cache_dir,
            nmodels=self.nmodels,
            cache_verbose=self.cache_verbose)

        if cache_checkpoint is not None:
            cache_preproc.cache_retrain(
                cache_checkpoint,
                cache_verbose=self.cache_verbose)

            autograd_model.solution_cls.model.apply(r)

            model = autograd_model.solution_cls.model(
                autograd_model.solution_cls.grid).reshape(
                self.model.solution_cls.model.shape).detach()
            
            self.model.solution_cls._model_change(model.requires_grad_())

    def cache(self):
        """ Wrap for cache_mat and cache_nn methods.
        """

        if self.model.mode != 'mat':
            return self._cache_nn()
        elif self.model.mode == 'mat':
            return self._cache_mat()

    def on_train_begin(self, logs=None):
        self.cache()
        self.model._save_dir = self.cache_dir
