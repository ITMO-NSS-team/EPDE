# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""
import pickle
import datetime
import torch
import os
import glob
import numpy as np
import shutil
from copy import deepcopy
from typing import Union, Tuple, Any

from epde.solver.solution import Solution
from epde.solver.input_preprocessing import Equation, EquationMixin
from epde.solver.device import device_type


def count_output(model):
    """
    Counts the output features of the last relevant layer in a given model.
    
        This method iterates through the layers of a model in reverse order, searching for the
        'out_features' attribute. It returns the 'out_features' of the first layer found with this attribute.
        This is useful for determining the dimensionality of the solution space when constructing
        equation candidates.
    
        Args:
            model: The model to analyze.
    
        Returns:
            int: The output features of the last layer with 'out_features' attribute, or None if no such layer is found.
    """
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


def create_random_fn(eps):
    """
    Creates a function that adds random noise to the parameters of linear and convolutional layers.
    This helps to explore the model space by slightly perturbing the weights and biases, 
    allowing the evolutionary algorithm to discover new and potentially better-performing equation structures.
    
    Args:
        eps (float): The magnitude of the random noise to add to the parameters.
    
    Returns:
        function: A function that takes a PyTorch module as input and adds random noise to the weights and biases of its linear and convolutional layers.
    """
    def randomize_params(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            m.weight.data = m.weight.data + \
                            (2 * torch.randn(m.weight.size()) - 1) * eps
            m.bias.data = m.bias.data + (2 * torch.randn(m.bias.size()) - 1) * eps

    return randomize_params

def remove_all_files(folder):
    """
    Removes all files and subdirectories within a specified folder to ensure a clean state before or after equation discovery processes. This is crucial for managing temporary files and directories created during the search for the best equation structure.
    
        Args:
            folder (str): The path to the folder whose contents should be removed.
    
        Returns:
            None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class Model_prepare():
    """
    Prepares initial model. Serves for computing acceleration.\n
        Saves the trained model to the cache, and subsequently it is possible to use pre-trained model (if \\\
        it saved and if the new model is structurally similar) to sped up computing.\n
        If there isn't pre-trained model in cache, the training process will start from the beginning.
    """

    def __init__(self, grid, equal_cls, model, mode, weak_form):
        """
        Initializes the Model_prepare instance, setting up the computational environment for equation discovery.
        
                This involves associating the provided grid, equation class, model, mode of operation, and weak form setting with the instance.
                The cache directory is initialized to store intermediate results, which speeds up the equation discovery process by avoiding redundant computations.
        
                Args:
                    grid (object): The spatial or temporal grid on which the data is defined.
                    equal_cls (object): The class defining the equation structure and its parameters.
                    model (object): The model object containing data and any prior assumptions.
                    mode (str): The mode of operation (e.g., training, validation).
                    weak_form (bool): A flag indicating whether to use the weak form of the equation.
        
                Returns:
                    None
        """
        self.grid = grid
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form
        try:
             file = __file__
        except:
            file = os.getcwd()
        self.cache_dir = os.path.normpath((os.path.join(os.path.dirname(file), '..','cache')))

    def change_cache_dir(self, string):
        """
        Changes the directory where intermediate results and processed data are stored.
        
        This allows to control where the framework saves its working files, 
        ensuring reproducibility and efficient management of storage space 
        during the equation discovery process.
        
        Args:
            string (str): The new path to the cache directory.
        
        Returns:
            None
        """
        self.cache_dir=string
        return None

    def clear_cache_dir(self, directory=None):
        """
        Clears the specified cache directory, removing all files within it. This ensures that outdated or irrelevant cached data does not interfere with subsequent equation discovery runs, promoting accurate and efficient model generation.
        
                Args:
                    directory (str, optional): The path to the cache directory to clear. If None, the default cache directory associated with the `Model_prepare` instance is cleared. Defaults to None.
        
                Returns:
                    None
        """
        if directory==None:
            remove_all_files(self.cache_dir)
        else:
            remove_all_files(directory)
        return None

    @staticmethod
    def cache_files(files, nmodels):
        """
        Caches a subset of files to optimize the search for the best equation structure.
        
                When exploring a large space of possible differential equations,
                it can be beneficial to work with a subset of the available models
                to reduce computational cost. This method selects a subset of files
                representing these models.
        
                Args:
                    files: A list of files to consider for caching.
                    nmodels: The number of models to randomly select from the cache.
                        If None, all files are used.
        
                Returns:
                    A NumPy array of indices representing the cached files.
        """

        # at some point we may want to reduce the number of models that are
        # checked for the best in the cache
        if nmodels == None:
            # here we take all files that are in cache
            cache_n = np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n = np.random.choice(len(files), nmodels, replace=False)

        return cache_n

    def grid_model_mat(self, cache_model):
        """
        Generates a grid representing the input space and prepares a neural network model for approximating the solution.
        
                This method constructs a grid from the object's grid attribute, which represents the domain where the differential equation is defined. It also initializes or reuses a neural network model that will learn to approximate the solution of the differential equation over this grid.
        
                Args:
                    cache_model: An optional pre-existing neural network model. If provided, this model will be used; otherwise, a new model will be created.
        
                Returns:
                    tuple: A tuple containing the grid matrix (NN_grid) and the neural network model (cache_model). If cache_model was None, a new model is created and returned; otherwise, the original cache_model is returned.
        
                Why:
                    The grid is needed to discretize the domain of the differential equation, allowing the neural network to learn the solution at specific points. The neural network model serves as a function approximator, learning to map the grid points to the corresponding solution values.
        """
        NN_grid = torch.vstack([self.grid[i].reshape(-1) for i in \
                                range(self.grid.shape[0])]).T.float()
        out = self.model.shape[0]

        if cache_model == None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(self.grid.shape[0], 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, out)
            )
        return NN_grid, cache_model

    @staticmethod
    def mat_op_coeff(operator):
        """
        Applies necessary transformations to the coefficients of operators.
        
                This method iterates through the provided operator(s) and reshapes the
                coefficient of each term within the operator(s) if it's a PyTorch tensor to ensure compatibility
                with subsequent mathematical operations. It also issues a warning if a coefficient is callable,
                as this might interfere with the caching mechanism and lead to unexpected behavior during
                equation discovery. This ensures that coefficients are in the correct format for equation processing
                and alerts the user to potential issues with dynamically defined coefficients.
        
                Args:
                  operator: The operator to process. It can be a single operator (dict) or a list of operators.
        
                Returns:
                  The processed operator(s) with reshaped coefficients (if applicable).
        """
        if type(operator) is not list:
            operator = [operator]
        for op in operator:
            for label in list(op.keys()):
                term = op[label]
                if type(term['coeff']) == torch.Tensor:
                    term['coeff'] = term['coeff'].reshape(-1, 1)
                elif callable(term['coeff']):
                    print("Warning: coefficient is callable,\
                                it may lead to wrong cache item choice")
        return operator

    @staticmethod
    def model_reform(init_model, model):
        """
        Reformats the initial and evolved models to ensure compatibility with subsequent operations.
        
                This method checks if the provided models (`init_model` and `model`) are directly indexable (e.g., `nn.Sequential`) or if they are encapsulated within a `model` attribute (e.g., a custom `nn.Module`). If the models are encapsulated, it extracts the underlying model. This ensures that both models can be consistently accessed as indexable structures, facilitating operations like layer-wise comparisons or modifications during the evolutionary process.
        
                Args:
                    init_model: The initial model (either `nn.Sequential` or a custom `nn.Module` with a `model` attribute).
                    model: The evolved model (either `nn.Sequential` or a custom `nn.Module` with a `model` attribute).
        
                Returns:
                    * **init_model** -- The reformed initial model (either `nn.Sequential` or `nn.ModuleList`).
                    * **model** -- The reformed evolved model (either `nn.Sequential` or `nn.ModuleList`).
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


    def cache_lookup(self, lambda_operator: float = 1., lambda_bound: float = 0.001,
                nmodels: Union[int, None] = None, save_graph: bool = False,
                cache_verbose: bool = False, return_normalized_loss: bool = False) -> Tuple[dict, torch.Tensor]:
        """
        Looks for a pre-trained model in the cache to initialize the optimization process.
        
                This function searches for previously saved models in the specified cache directory and evaluates their performance on the target problem. The best-performing model, based on the loss function, is then selected to provide a warm start for the optimization, potentially accelerating convergence and improving the final solution.
        
                Args:
                    lambda_operator (float, optional): Weight for the operator loss term. Defaults to 1.0.
                    lambda_bound (float, optional): Weight for the boundary loss term. Defaults to 0.001.
                    nmodels (Union[int, None], optional): Maximum number of models to consider from the cache. If None, all models are considered. Defaults to None.
                    save_graph (bool, optional): Whether to save the computational graph during evaluation. Defaults to False.
                    cache_verbose (bool, optional): Whether to print detailed information about the models in the cache. Defaults to False.
                    return_normalized_loss (bool, optional): Whether to return the normalized loss instead of the raw loss. Defaults to False.
        
                Returns:
                    Tuple[dict, torch.Tensor]: A tuple containing:
                        * **best_checkpoint** (dict or None): A dictionary containing the best model's state, including the model itself, its state dictionary, and the optimizer's state dictionary. Returns None if no suitable model is found in the cache.
                        * **min_loss** (torch.Tensor): The minimum loss achieved by the best model in the cache.
        """
        files = glob.glob(self.cache_dir + '\*.tar')
        if len(files) == 0:
            best_checkpoint = None
            min_loss = torch.tensor([float('inf')])
            return best_checkpoint, min_loss

        cache_n = self.cache_files(files, nmodels)

        min_loss = np.inf
        min_norm_loss =np.inf
        best_checkpoint = {}

        device = device_type()

        for i in cache_n:
            file = files[i]
            checkpoint = torch.load(file)
            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])
            # this one for the input shape fix if needed

            solver_model, cache_model = self.model_reform(self.model, model)

            if cache_model[0].in_features != solver_model[0].in_features:
                continue
            try:
                if count_output(model) != count_output(self.model):
                    continue
            except Exception:
                continue

            model = model.to(device)
            loss, loss_normalized = Solution(self.grid, self.equal_cls,
                                      model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol=0,
                                      derivative_points=2).evaluate(save_graph=save_graph)

            if loss < min_loss:
                min_loss = loss
                min_norm_loss=loss_normalized
                best_checkpoint['model'] = model
                best_checkpoint['model_state_dict'] = model.state_dict()
                best_checkpoint['optimizer_state_dict'] = \
                    checkpoint['optimizer_state_dict']
                if cache_verbose:
                    print('best_model_num={} , normalized_loss={}'.format(i, min_norm_loss.item()))
        if best_checkpoint == {}:
            best_checkpoint = None
            min_loss = np.inf
        if return_normalized_loss:
            min_loss=min_norm_loss
        return best_checkpoint, min_loss

    def save_model(self, prep_model: Any, state: dict, optimizer_state: dict, name: Union[str, None] = None):
        """
        Saves the trained model, its state, and the optimizer's state to a cache file. This allows for later reuse of the trained model without retraining, which is useful for comparing different equation structures and optimization strategies.
        
                Args:
                    prep_model: The trained model to be saved.
                    state: A dictionary containing the model's state (layer-to-parameter tensor mapping).
                    optimizer_state: A dictionary containing the optimizer's state (values, hyperparameters).
                    name: An optional name for the saved model file. If None, a timestamp is used.
        
                Returns:
                    None. The model is saved to a file in the cache directory.
        """
        if name == None:
            name = str(datetime.datetime.now().timestamp())
        if not(os.path.isdir(self.cache_dir)):
            os.mkdir(self.cache_dir)

        try:
            torch.save({'model': prep_model.to('cpu'), 'model_state_dict': state,
                        'optimizer_state_dict': optimizer_state}, self.cache_dir+'\\' + name + '.tar')
            print('model is saved in cache')
        except RuntimeError:
            torch.save({'model': prep_model.to('cpu'), 'model_state_dict': state,
                        'optimizer_state_dict': optimizer_state}, self.cache_dir+'\\' + name + '.tar', _use_new_zipfile_serialization=False) #cyrrilic in path
            print('model is saved in cache')
        except:
            print('Cannot save model in cache')


    def save_model_mat(self, name: None = None, cache_model: None = None,
                       cache_verbose: bool=False):
        """
        Fine-tunes a simplified model to mimic the behavior of a more complex, pre-trained model.
        
        This process involves training a smaller, more efficient model to approximate the output of a larger, pre-trained model on a specific grid.
        This is done to create a computationally cheaper representation of the original model, suitable for tasks where speed and memory efficiency are critical.
        
        Args:
            name (str, optional):  A name to assign to the saved model. Defaults to None.
            cache_model (torch.nn.Module, optional): The simplified model to be trained. Defaults to None.
            cache_verbose (bool, optional): If True, prints the training loss at each iteration. Defaults to False.
        
        Returns:
            None
        """

        NN_grid, cache_model = self.grid_model_mat(cache_model)
        optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.001)
        model_res = self.model.reshape(-1, self.model.shape[0])

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(NN_grid) - model_res) ** 2)
            loss.backward()
            return loss

        loss = np.inf
        t = 0
        while loss > 1e-5 and t < 1e5:
            loss = optimizer.step(closure)
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        self.save_model(cache_model, cache_model.state_dict(),
                        optimizer.state_dict(), name=name)

    def scheme_interp(self, trained_model: Any, cache_verbose: bool = False) -> Tuple[Any, dict]:
        """
        Interpolates the model's parameters to match the behavior of a pre-trained model.
        
                This method refines the current model by minimizing the difference between its output and the output of a `trained_model` on a given grid. It uses an optimization loop to adjust the model's parameters until the loss (mean squared error) falls below a threshold or a maximum number of iterations is reached. This is useful for adapting a general model to a specific solution learned by another model.
        
                Args:
                    trained_model: A pre-trained model whose behavior the current model should mimic.
                    cache_verbose: If True, prints the loss at each iteration of the optimization loop.
        
                Returns:
                    * **model**: The refined model, with parameters adjusted to approximate the `trained_model`.
                    * **optimizer_state**: A dictionary containing the state of the optimizer after the interpolation process.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(
            trained_model(self.grid) - self.model(self.grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((
                trained_model(self.grid) - self.model(self.grid)) ** 2)
            loss.backward()
            return loss

        t = 0
        while loss > 1e-5 and t < 1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(
                trained_model(self.grid) - self.model(self.grid)))
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        return self.model, optimizer.state_dict()

    def cache_retrain(self, cache_checkpoint, cache_verbose: bool = False) -> Union[
        Tuple[Any, None], Tuple[Any, Union[dict, Any]]]:
        """
        Refines the model using a cached model if available, otherwise performs a retraining.
        
                This method attempts to leverage a previously cached model to accelerate the training process.
                If a compatible cached model exists (same structure), it's loaded and used directly.
                Otherwise, the input model is retrained, potentially using the cached model as a starting point.
                This is done to avoid training models from scratch every time, saving computational resources and time.
        
                Args:
                    cache_checkpoint: A dictionary containing the cached model's state, optimizer state, and model architecture. If None, caching is skipped.
                    cache_verbose: If True, prints a message indicating whether the model was loaded from the cache.
        
                Returns:
                    * **model**: The refined model, either loaded from the cache or retrained.
                    * **optimizer_state**: The optimizer state, loaded from the cache if a compatible model was found, otherwise None.
        """

        # do nothing if cache is empty
        if cache_checkpoint == None:
            optimizer_state = None
            return self.model, optimizer_state
        # if models have the same structure use the cache model state,
        # and the cache model has ordinary structure
        if str(cache_checkpoint['model']) == str(self.model) and \
                 isinstance(self.model, torch.nn.Sequential) and \
                 isinstance(self.model[0], torch.nn.Linear):
            self.model = cache_checkpoint['model']
            self.model.load_state_dict(cache_checkpoint['model_state_dict'])
            self.model.train()
            optimizer_state = cache_checkpoint['optimizer_state_dict']
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model
        else:
            optimizer_state = None
            model_state = None
            cache_model = cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.model, optimizer_state = self.scheme_interp(
                cache_model, cache_verbose=cache_verbose)
        return self.model, optimizer_state

    def cache_nn(self, nmodels: Union[int, None], lambda_operator: float, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, return_normalized_loss: bool = False):
        """
        Restores a pre-trained model from the cache and refines it to better fit the specific problem. This leverages prior knowledge encoded in the cache to accelerate the model fitting process.
        
               Args:
                   nmodels: The number of models to consider from the cache.
                   lambda_operator: Regularization strength for operators in the equation.
                   lambda_bound: Bound for the regularization strength.
                   cache_verbose:  Whether to print detailed information about the models loaded from the cache.
                   model_randomize_parameter:  If provided, randomizes the model parameters (weights, biases) by multiplying them with this value.
                   cache_model: The cached model to restore.
                   return_normalized_loss: Whether to return the normalized loss.
        
               Returns:
                   * **model** -- The refined neural network model.
                   * **min_loss** -- The minimum loss achieved by the restored model.
        """
        r = create_random_fn(model_randomize_parameter)
        cache_checkpoint, min_loss = self.cache_lookup(nmodels=nmodels,
                                                       cache_verbose=cache_verbose,
                                                       lambda_operator= lambda_operator,
                                                       lambda_bound=lambda_bound, 
                                                       return_normalized_loss = return_normalized_loss)
        
        self.model, optimizer_state = self.cache_retrain(cache_checkpoint,
                                                         cache_verbose=cache_verbose)

        self.model.apply(r)

        return self.model, min_loss

    def cache_mat(self, nmodels: Union[int, None],lambda_operator: float, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, return_normalized_loss: bool = False):
        """
        Restores a model from a pre-computed cache, potentially refining it to better fit the specific problem instance. This leverages prior computations to accelerate the solution process.
        
                Args:
                    nmodels (Union[int, None]): Number of models to consider from the cache. If None, all cached models are used.
                    lambda_operator (float): Weighting factor for the operator loss term.
                    lambda_bound (float): Weighting factor for the boundary condition loss term. Influences convergence speed.
                    cache_verbose (bool): Enables more detailed logging of cache operations.
                    model_randomize_parameter (Union[float, None]): If provided, randomizes the model parameters (weights, biases) by multiplying them with a random value.
                    cache_model (torch.nn.Sequential): The cached model to be used as a starting point.
                    return_normalized_loss (bool, optional): If True, returns the normalized loss. Defaults to False.
        
                Returns:
                    torch.Tensor: The refined model, adapted to the current problem.
                    float: The minimum loss achieved by the refined model.
        
                Why:
                    This method aims to improve the efficiency of solving differential equations by reusing previously computed solutions. By starting from a cached model and fine-tuning it to the specific problem instance, the method can converge faster and achieve better accuracy than training a model from scratch.
        """

        NN_grid, cache_model = self.grid_model_mat(cache_model)
        operator = deepcopy(self.equal_cls.operator)
        bconds = deepcopy(self.equal_cls.bconds)
        operator = self.mat_op_coeff(operator)
        r = create_random_fn(model_randomize_parameter)
        eq = Equation(NN_grid, operator, bconds).set_mode('autograd')
        model_cls = Model_prepare(NN_grid, eq, cache_model, 'autograd', self.weak_form)
        cache_checkpoint, min_loss = model_cls.cache_lookup(
            nmodels=nmodels,
            cache_verbose=cache_verbose,
            lambda_bound=lambda_bound,
            lambda_operator=lambda_operator,
            return_normalized_loss=return_normalized_loss)
        if cache_checkpoint is not None:
            prepared_model, optimizer_state = model_cls.cache_retrain(
                                                        cache_checkpoint,
                                                        cache_verbose=cache_verbose)

            prepared_model.apply(r)
        
            self.model = prepared_model(NN_grid).reshape(
                            self.model.shape).detach()

        min_loss, _ = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol=0,
                                      derivative_points=2).evaluate()

        return self.model, min_loss

    def cache(self, nmodels: Union[int, None],lambda_operator, lambda_bound: float,
              cache_verbose: bool,model_randomize_parameter: Union[float, None],
              cache_model: torch.nn.Sequential, 
              return_normalized_loss: bool = False):
        """
        Restores a pre-trained model from the cache to refine its parameters based on new data. This leverages prior knowledge to accelerate the model fitting process and potentially improve generalization.
        
                Args:
                    nmodels: The number of cached models to consider.
                    lambda_operator: regularization parameter for operators
                    lambda_bound: A constant influencing the convergence speed of the retraining process.
                    cache_verbose: Enables more detailed logging of the cached models.
                    model_randomize_parameter: Introduces slight variations to the model's initial parameters (weights, biases) by multiplying them with a random factor.
                    cache_model: The pre-trained model loaded from the cache.
                    return_normalized_loss: if true returns normalized loss
        
                Returns:
                    The refined model, either a neural network (`cache.cache_nn`) or a matrix-based model (`cache.cache_mat`), depending on the configuration.
        """

        if self.mode != 'mat':
            return self.cache_nn(nmodels,lambda_operator, lambda_bound,
                                 cache_verbose, model_randomize_parameter,
                                 cache_model,return_normalized_loss=return_normalized_loss)
        elif self.mode == 'mat':
            return self.cache_mat(nmodels, lambda_operator, lambda_bound,
                                  cache_verbose, model_randomize_parameter,
                                  cache_model,return_normalized_loss=return_normalized_loss)

