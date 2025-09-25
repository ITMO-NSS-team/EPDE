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
    """
    Determine the number of output features of the model's final layer.
    
    This is crucial for adapting the discovered equation to the specific data being modeled.
    By identifying the output features, the framework ensures that the equation's structure aligns
    with the dimensionality of the target variables.
    
    Args:
        model (torch.Tensor): The neural network model.
    
    Returns:
        int: The number of output features in the final layer.
    """
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


class CachePreprocessing:
    """
    class for preprocessing cache files.
    """

    def __init__(self,
                 model: Model
                 ):
        """
        Initializes the CachePreprocessing object with a model.
        
                This ensures that the preprocessing steps are aligned with the specific solution class
                defined within the provided model, facilitating consistent data handling throughout the equation discovery process.
        
                Args:
                    model (Model): The model containing the solution class to be used for preprocessing.
        
                Returns:
                    None
        """
        self.solution_cls = model.solution_cls

    @staticmethod
    def _cache_files(files: list, nmodels: Union[int, None]=None) -> np.ndarray:
        """
        Reduces the number of cached models to be evaluated, potentially improving the efficiency of the equation discovery process. By limiting the number of models considered, the search for the best equation can be accelerated.
        
                Args:
                    files (list): A list of all model names available in the cache.
                    nmodels (Union[int, None], optional): The desired number of models to select for evaluation. If None, all models are selected. Defaults to None.
        
                Returns:
                    np.ndarray: An array containing the indices of the selected cache files. These indices can be used to access the corresponding model data.
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
        Checks and adjusts the structure of the initial and cached models to ensure compatibility.
        
                This function verifies if the provided models are directly indexable (e.g., `nn.Sequential`) or if they are nested within a `model` attribute. If a model is nested, it extracts the underlying model. This ensures that subsequent operations can correctly access the model's layers, regardless of its initial structure. This adjustment is crucial for consistent handling of different model types within the caching mechanism.
        
                Args:
                    init_model (nn.Sequential or nn.ModuleList): The initial model used by the solver.
                    model (nn.Sequential or nn.ModuleList): The cached model to be checked.
        
                Returns:
                    init_model (nn.Sequential or nn.ModuleList): The adjusted initial model.
                    model (nn.Sequential or nn.ModuleList): The adjusted cached model.
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
        """
        Looks for the best performing model within the cached models based on validation loss.
        
                This method iterates through the cached models, evaluates their performance, and selects the one with the lowest validation loss.
                It ensures compatibility between the cached model and the current solver's model before evaluation.
                This is done to efficiently reuse previously trained models and accelerate the equation discovery process.
        
                Args:
                    cache_dir (str): Folder where the system looks for cached models.
                    nmodels (Union[int, None], optional): Maximal number of models to consider from the cache directory. Defaults to None, meaning all models are considered.
                    save_graph (bool, optional): Whether to save the computational graph during evaluation. Defaults to False.
                    cache_verbose (bool, optional): Enables verbose output for cache operations. Defaults to False.
        
                Returns:
                    Union[None, dict, torch.nn.Module]: The best model found in the cache, along with its optimizer state, or None if no suitable model is found.
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
        """
        Trains the user's model to mimic the behavior of a pre-trained model from the cache, effectively transferring knowledge and adapting to potential architectural differences. This ensures compatibility and leverages prior learning when the cache model's structure differs from the user's specified model.
        
                Args:
                    trained_model (torch.nn.Module): The pre-trained model from the cache whose behavior will be mimicked.
                    cache_verbose (bool, optional): Enables verbose output during the training process. Defaults to False.
        
                Returns:
                    torch.nn.Module: The user's model, fine-tuned to approximate the behavior of the `trained_model`.
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
        """
        Compares the user-provided model architecture with the cached model architecture to determine if the cached model can be directly used or if interpolation is required to transfer knowledge.
        
                Args:
                    cache_checkpoint (dict): Checkpoint containing the cached model's architecture and state.
                    cache_verbose (bool, optional): Enables verbose logging of cache operations. Defaults to False.
        
                Returns:
                    torch.nn.Module: Returns the model. It returns None if the cache is empty, returns cache model if architectures are the same, and returns the user defined model after interpolation in other cases.
        
                Why:
                    This method aims to accelerate the model training process by reusing previously learned knowledge from similar models. If the architectures match, the cached model's state is directly loaded. Otherwise, interpolation is used to adapt the cached knowledge to the current model, leveraging prior learning to guide the training process.
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
        Initializes the Cache object for managing pre-trained models.
        
                The cache facilitates the reuse of previously trained models, potentially reducing the computational cost of solving similar problems.
        
                Args:
                    nmodels (Union[int, None], optional): The maximum number of models to keep in the cache directory. If None, all models are kept. Defaults to None.
                    cache_dir (str, optional): The directory where cached models are stored. Defaults to 'tedeous_cache', which resolves to a subdirectory within the system's temporary directory. If a custom directory is specified, it is resolved relative to the *torch_de_solver* directory.
                    cache_verbose (bool, optional): Enables verbose output for cache operations, providing more detailed information about loading and saving models. Defaults to False.
                    cache_model (Union[torch.nn.Sequential, None], optional): A PyTorch model to be immediately saved to the cache. Defaults to None.
                    model_randomize_parameter (Union[int, float], optional): A factor to randomize the model's initial parameters (weights and biases) before saving it to the cache. This can help explore different regions of the solution space. Defaults to 0.
                    clear_cache (bool, optional): If True, the cache directory is emptied upon initialization. Defaults to False.
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
        """
        Utilizes a model from the cache as an initial starting point for neural network-based equation discovery. This approach leverages previously learned information to accelerate the search process and potentially improve the quality of the discovered equations by initializing the model with a promising configuration.
        
                Args:
                    self: Instance of the Cache class.
        
                Returns:
                    None
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
        """
        Refines the initial guess for the *mat* mode solution by leveraging a cache of previously trained models. It retrieves a suitable model from the cache, retrains it to better align with the current problem, and uses its output as a starting point for the *mat* mode solution. This approach accelerates the solution process by providing a pre-trained model that captures general solution characteristics.
        
                Args:
                    self: The Cache instance.
        
                Returns:
                    torch.Tensor: A refined initial guess for the *mat* mode solution, obtained from the cache and adapted to the current problem.
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
        """
        Caches the model's computations based on its mode.
        
        This method acts as a dispatcher, selecting the appropriate caching strategy
        based on whether the model is operating in 'mat' mode or another mode (presumably 'nn').
        It ensures that the computationally intensive parts of the model are cached for later use,
        potentially speeding up subsequent calculations or analyses.
        
        Args:
            self: The Cache instance.
        
        Returns:
            The result of either `self._cache_mat()` if the model is in 'mat' mode,
            or `self._cache_nn()` otherwise.
        """

        if self.model.mode != 'mat':
            return self._cache_nn()
        elif self.model.mode == 'mat':
            return self._cache_mat()

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.
        
        Caches the model and sets the save directory. This ensures that the discovered equation and its associated parameters can be efficiently stored and retrieved during the evolutionary search process, facilitating faster exploration of the solution space.
        
        Args:
            logs: Contains the logs.
        
        Returns:
            None
        """
        self.cache()
        self.model._save_dir = self.cache_dir
