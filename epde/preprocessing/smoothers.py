#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:15:12 2022

@author: maslyaev
"""
import math
from warnings import warn
from abc import ABC
from scipy.ndimage import gaussian_filter
import numpy as np

import torch


import epde.globals as global_var

class AbstractSmoother(ABC):
    """
    Abstract base class for smoothers.
    
        This class defines the interface for all smoother classes.
        It provides a common structure for implementing smoothing algorithms.
    
        Class Methods:
        - __call__:
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the smoother object.
        
        This method serves as a placeholder in the abstract class.
        Subclasses should override this method to implement specific
        initialization procedures required for different smoothing techniques.
        
        Args:
            *args: Variable length argument list.  These arguments will be passed to the underlying smoother.
            **kwargs: Arbitrary keyword arguments. These keyword arguments will be passed to the underlying smoother.
        
        Returns:
            None.
        
        Why: This initialization ensures that all concrete smoother classes have a consistent interface,
        allowing them to be used interchangeably within the equation discovery process.
        """
        pass

    def __call__(self, data, *args, **kwargs):
        """
        Applies a smoothing operation to the input data.
        
        This method is part of the abstract base class for all smoothing techniques
        within the EPDE framework. It ensures that all concrete smoother
        implementations provide a consistent interface for applying smoothing
        operations.
        
        Raises:
            NotImplementedError: Always raised, as this is an abstract method and
                must be implemented by subclasses.
        
        Args:
            data (array-like): The input data to be smoothed. This can be a
                NumPy array or any other array-like object that can be processed
                by the smoothing algorithm.
            *args: Variable length argument list.  These arguments are passed
                to the underlying smoothing implementation.
            **kwargs: Arbitrary keyword arguments. These keyword arguments are
                passed to the underlying smoothing implementation.
        
        Returns:
            None: This method always raises an error, as it is an abstract method.
        """
        raise NotImplementedError('Calling abstract smoothing object')


class PlaceholderSmoother(AbstractSmoother):
    """
    A placeholder smoother that does nothing.
    
        This smoother simply returns the input data without modification.
        It is useful as a default or when no smoothing is desired.
    """

    def __init__(self):
        """
        Initializes the PlaceholderSmoother.
        
                This class serves as a placeholder for more sophisticated smoothing techniques.
                Currently, it performs no smoothing, acting as an identity operation.
                It is used as a base or default option within the EPDE framework when smoothing is not required.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        pass

    def __call__(self, data, *args, **kwargs):
        """
        Applies a placeholder transformation to the input data.
        
        This transformation serves as a stand-in when no actual smoothing or data modification is required. It ensures that the data pipeline remains consistent and functional even when a smoothing step is not necessary.
        
        Args:
            data: The input data, which can be of any type.
        
        Returns:
            The input data, returned without any modifications. This maintains data integrity when smoothing is not needed.
        """
        return data


def baseline_ann(dim):
    """
    Creates a baseline artificial neural network (ANN) model for approximating differential equation solutions.
    
        This method constructs a simple feedforward neural network using PyTorch.
        The network consists of several linear layers with ReLU activation functions.
        The input dimension is determined by the 'dim' parameter, and the output is a single value,
        representing the approximated solution at a given point. This baseline model serves as a
        fundamental building block for more complex equation discovery and solution approximation tasks
        within the EPDE framework.
    
        Args:
            dim: The input dimension of the first linear layer, corresponding to the number of independent
                 variables in the differential equation.
    
        Returns:
            torch.nn.Sequential: A PyTorch Sequential model representing the ANN.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        # torch.nn.Linear(64, 64),
        # torch.nn.Tanh(),
        torch.nn.Linear(64, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1)
    )
    return model

class Rational(torch.nn.Module):
    """
    Represents a rational number with a numerator and denominator.
    
        Class Methods:
        - __init__: Initializes a Rational object.
        - __str__: Returns a string representation of the Rational object.
        - __repr__: Returns a string representation of the Rational object for debugging.
        - __eq__: Checks if two Rational objects are equal.
        - __ne__: Checks if two Rational objects are not equal.
        - __lt__: Checks if one Rational object is less than another.
        - __gt__: Checks if one Rational object is greater than another.
        - __le__: Checks if one Rational object is less than or equal to another.
        - __ge__: Checks if one Rational object is greater than or equal to another.
        - __add__: Adds two Rational objects.
        - __sub__: Subtracts two Rational objects.
        - __mul__: Multiplies two Rational objects.
        - __truediv__: Divides two Rational objects.
        - __float__: Converts the Rational object to a float.
    
        Attributes:
          numerator (int): The numerator of the rational number.
          denominator (int): The denominator of the rational number.
    """

    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
        """
        Initializes the Rational activation function with trainable numerator and denominator coefficients.
        
                This initialization sets up the parameters that define the rational function,
                allowing the network to learn an appropriate non-linear activation. The
                coefficients are initialized to values that provide a good starting point
                for approximating ReLU, enabling effective gradient-based optimization
                during training.
        
                Args:
                    Data_Type (torch.dtype): The data type for the coefficients (e.g., torch.float32).
                    Device (torch.device): The device to store the coefficients on (e.g., 'cpu' or 'cuda').
        
                Returns:
                    None
        
                Class Fields:
                    a (torch.nn.parameter.Parameter): Numerator coefficients, initialized to (1.1915, 1.5957, 0.5, .0218). Requires gradient tracking.
                    b (torch.nn.parameter.Parameter): Denominator coefficients, initialized to (2.3830, 0.0, 1.0). Requires gradient tracking.
        
                Why:
                    Initializing the Rational activation function with trainable coefficients allows the model to learn complex, non-linear relationships within the data, which is crucial for accurately representing the dynamics described by differential equations. The initial values are chosen to provide a good starting point for optimization, facilitating the discovery of governing equations from data.
        """
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__()

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype = Data_Type,
                                     device = Device))
        self.a.requires_grad_(True)

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype = Data_Type,
                                     device = Device))
        self.b.requires_grad_(True)

    def forward(self, X : torch.tensor):
        """
        Applies a rational function, defined by learnable parameters, element-wise to the input tensor.
        
                The rational function is a ratio of two polynomials, allowing the model to approximate complex functions and relationships within the data. This transformation enhances the model's ability to capture non-linear dynamics present in the data.
        
                Args:
                    X (torch.tensor): The input tensor to which the rational function will be applied.
        
                Returns:
                    torch.tensor: A tensor of the same shape as X, with each element transformed by the rational function.
        """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a
        b = self.b

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X))
        D_X = b[0] + X*(b[1] + b[2]*X)

        # Return R = N_X/D_X. This is also applied element-wise.
        return N_X/D_X

class Sin(torch.nn.Module):
    """
    Applies the sine function to the input tensor.
    
        Class Methods:
        - forward:
    """

    def __init__(self):
        """
        Initializes a new instance of the Sin class.
        
                This constructor prepares the `Sin` object for symbolic manipulation and equation discovery. It ensures that the object is properly set up to represent a sine function within the broader equation search space.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None
        """
        super(Sin, self).__init__()

    def forward(self, x):
        """
        Applies the sine function element-wise to the input tensor.
        
                This operation is a fundamental building block for constructing more complex equation structures within the evolutionary search process. By applying sine, the framework can explore non-linear relationships and periodic behaviors in the data, which are common in many physical and biological systems modeled by differential equations.
        
                Args:
                    x (torch.Tensor): The input tensor.
        
                Returns:
                    torch.Tensor: A tensor with the same shape as input, containing the sine of each element in `x`.
        """
        x = torch.sin(x)
        return x

class NN(torch.nn.Module):
    """
    A feedforward neural network class.
    
        Class Methods:
        - __init__:
    """

    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int          = 1,    # Dimension of the input
                 Output_Dim          : int          = 1,    # Dimension of the output
                 Data_Type           : torch.dtype  = torch.float32,
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Tanh",
                 Batch_Norm          : bool         = False):
        """
        Initializes the neural network architecture.
        
                This method sets up the layers, activation functions, and normalization layers (if specified) of the neural network. The architecture is designed to provide a flexible foundation for representing functions, allowing the framework to approximate solutions to differential equations. The dimensions of the input and output layers, as well as the number of hidden layers and neurons per layer, are configurable to suit the complexity of the target function.
        
                Args:
                    Num_Hidden_Layers (int): The number of hidden layers in the network.
                    Neurons_Per_Layer (int): The number of neurons in each hidden layer.
                    Input_Dim (int): The dimension of the input.
                    Output_Dim (int): The dimension of the output.
                    Data_Type (torch.dtype): The data type to use for the network's parameters (e.g., torch.float32).
                    Device (torch.device): The device to use for the network's computations (e.g., torch.device('cpu') or torch.device('cuda')).
                    Activation_Function (str): The activation function to use for the hidden layers (e.g., "Tanh", "Sin", "Rational").
                    Batch_Norm (bool): Whether to use batch normalization.
        
                Returns:
                    None
        """
        # For the code below to work, Num_Hidden_Layers, Neurons_Per_Layer,
        # Input_Dim, and Output_Dim must be positive integers.
        assert(Num_Hidden_Layers   > 0), "Num_Hidden_Layers must be positive. Got %du" % Num_Hidden_Layers;
        assert(Neurons_Per_Layer   > 0), "Neurons_Per_Layer must be positive. Got %u" % Neurons_Per_Layer;
        assert(Input_Dim           > 0), "Input_Dim must be positive. Got %u"  % Input_Dim;
        assert(Output_Dim          > 0), "Output_Dim must be positive. Got %u" % Output_Dim;

        super(NN, self).__init__()

        # Define object attributes.
        self.Input_Dim          : int  = Input_Dim
        self.Output_Dim         : int  = Output_Dim
        self.Num_Hidden_Layers  : int  = Num_Hidden_Layers
        self.Batch_Norm         : bool = Batch_Norm

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.Layers = torch.nn.ModuleList()

        # Initialize Batch Normalization, if we're doing that.
        if(Batch_Norm == True):
            self.Norm_Layer = torch.nn.BatchNorm1d(
                                    num_features = Input_Dim,
                                    dtype        = Data_Type,
                                    device       = Device)

        # Append the first hidden layer. The domain of this layer is
        # R^{Input_Dim}. Thus, in_features = Input_Dim. Since this is a hidden
        # layer, its co-domain is R^{Neurons_Per_Layer}. Thus, out_features =
        # Neurons_Per_Layer.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Now append the rest of the hidden layers. Each maps from
        # R^{Neurons_Per_Layer} to itself. Thus, in_features = out_features =
        # Neurons_Per_Layer. We start at i = 1 because we already created the
        # 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(
                                    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias         = True ).to(dtype = Data_Type, device = Device))

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Initialize the weight matrices, bias vectors in the network.
        if(Activation_Function == "Tanh" or Activation_Function == "Rational"):
            Gain : float = 0
            if  (Activation_Function == "Tanh"):
                Gain = 5./3.
            elif(Activation_Function == "Rational"):
                Gain = 1.41

            for i in range(self.Num_Hidden_Layers + 1):
                torch.nn.init.xavier_normal_(self.Layers[i].weight, gain = Gain)
                torch.nn.init.zeros_(self.Layers[i].bias)

        elif(Activation_Function == "Sin"):
            # The SIREN paper suggests initializing the elements of every weight
            # matrix (except for the first one) by sampling a uniform
            # distribution over [-c/root(n), c/root(n)], where c > root(6),
            # and n is the number of neurons in the layer. I use c = 3 > root(6).
            #
            # Further, for simplicity, I initialize each bias vector to be zero.
            a : float = 3./math.sqrt(Neurons_Per_Layer)
            for i in range(0, self.Num_Hidden_Layers + 1):
                torch.nn.init.uniform_( self.Layers[i].weight, -a, a)
                torch.nn.init.zeros_(   self.Layers[i].bias)

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList()
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh())
        elif(Activation_Function == "Sin"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Sin())
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational(Data_Type = Data_Type, Device = Device))
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function)
            print("Thrown by Neural_Network.__init__. Aborting.")
            exit();

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.  This method propagates the input tensor `X` through the network's layers, applying batch normalization (if enabled) and activation functions to each hidden layer. The final layer's output is then returned. Note that the user should NOT call this function directly. Rather, they should call it through the __call__ method (using the NN object like a function), which is part of the module class and calls forward.
        
        The forward pass is a crucial step in both training and inference, as it determines the network's output for a given input. By structuring the network in this way, we can approximate complex functions and discover underlying relationships within the data, which is essential for identifying governing differential equations.
        
        Args:
            X: A batch of inputs. This should be a B by Input_Dim tensor, where B
                is the batch size. The ith row of X should hold the ith input.
        
        Returns:
            A B by Output_Dim tensor, whose ith row holds the value of the network
            applied to the ith row of X.
        """

        # If we are using batch normalization, then normalize the inputs.
        if (self.Batch_Norm == True):
            X = self.Norm_Layer(X);

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (with no activation function) and return.
        return self.Layers[self.Num_Hidden_Layers](X);

class ANNSmoother(AbstractSmoother):
    """
    Applies smoothing to data using an Artificial Neural Network (ANN).
    
        Class Methods:
        - __init__:
    """

    def __init__(self):
        """
        Initializes the ANNSmoother object.
        
        This method initializes the base class and sets the internal model to None. 
        The model will be later populated with a suitable approximation method 
        to represent the discovered equation.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            model (object): The approximation model used for smoothing. Initialized to None.
        """
        super().__init__()  # Optional depending on AbstractSmoother
        self.model = None

    def __call__(self, data, grid, epochs_max=1000, loss_mean=1000, loss_threshold=1e-8,
                 batch_frac=0.5, val_frac=0.1, learning_rate=1e-3, return_ann=False, device='cpu'):
        """
        Applies a trained neural network to smooth the input data and approximate the underlying function.
        
                This method leverages a neural network to learn the relationship between the input grid and the corresponding data values. By training the network on flattened data, it captures the essential features and dependencies, effectively smoothing out noise and irregularities. The trained network is then used to predict values on the original grid, providing a smoothed representation of the data. Early stopping is employed to prevent overfitting and optimize the network's generalization ability.
        
                Args:
                    data (np.ndarray): The input data to be smoothed.
                    grid (tuple[np.ndarray]): The grid corresponding to the input data. Each element of the tuple represents a dimension of the grid.
                    epochs_max (int, optional): The maximum number of training epochs. Defaults to 1000.
                    loss_mean (float, optional): Unused parameter. Defaults to 1000.
                    loss_threshold (float, optional): The loss threshold for early stopping. Defaults to 1e-8.
                    batch_frac (float, optional): Unused parameter. Defaults to 0.5.
                    val_frac (float, optional): The fraction of data to use for validation. Defaults to 0.1.
                    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
                    return_ann (bool, optional): Whether to return the trained ANN model. Defaults to False.
                    device (str, optional): The device to use for training (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        
                Returns:
                    np.ndarray: The smoothed data as a NumPy array with the same shape as the input data.
        
                WHY: This smoothing operation is crucial for preparing data for equation discovery. By reducing noise and highlighting underlying patterns, it enables the evolutionary algorithm to identify more accurate and meaningful differential equation models.
        """
        if torch.cuda.is_available():
            device = "cuda"
        # Convert to int if passed as float
        epochs_max = int(epochs_max)

        # Infer input dimension
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim

        # Initialize model
        # model = baseline_ann(dim).to(device)
        model = NN(Num_Hidden_Layers=5, Neurons_Per_Layer=50, Input_Dim=dim, Activation_Function='Sin').to(device)
        self.model = model

        # Flatten grid and reshape field
        grid_flattened = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grid])).float().T.to(device)
        field_ = torch.from_numpy(data.reshape(-1, 1)).float().to(device)
        original_shape = data.shape

        # Train/val split
        N = grid_flattened.size(0)
        val_size = int(N * val_frac)
        train_size = N - val_size
        indices = torch.randperm(N)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_x, train_y = grid_flattened[train_idx], field_[train_idx]
        val_x, val_y = grid_flattened[val_idx], field_[val_idx]

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_max // 10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max // 10)

        # Loss function
        loss_fn = torch.nn.MSELoss()

        # Batch size
        batch_size = max(1, int(train_size))

        # Training loop
        min_val_loss = np.inf
        best_model_state = None

        model.train()
        for epoch in range(epochs_max):
            permutation = torch.randperm(train_x.size(0))
            train_loss_list = []

            for i in range(0, train_x.size(0)-1, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = train_x[indices]
                batch_y = train_y[indices]

                optimizer.zero_grad()
                pred = model(batch_x)
                # loss = loss_fn(pred, batch_y)
                loss = torch.mean(torch.abs(batch_y - pred))
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())

            train_loss = np.mean(train_loss_list)
            scheduler.step(train_loss)

            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = torch.mean(torch.abs(val_y - val_pred))
                # val_loss = loss_fn(val_pred, val_y).item()

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {val_loss:.6e}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_state = model.state_dict()

            if val_loss <= loss_threshold:
                print(f"Early stopping at epoch {epoch}, loss = {val_loss:.4e}")
                break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()
        self.model = model

        with torch.no_grad():
            prediction = model(grid_flattened).cpu().numpy().reshape(original_shape)

        if return_ann:
            warn('Returning ANN from smoother. This should only happen in selected experiments.')
            return prediction, model
        else:
            return prediction


class GaussianSmoother(AbstractSmoother):
    """
    Applies a Gaussian smoothing filter to data.
    
        Attributes:
            sigma: The standard deviation for the Gaussian kernel.
            truncate: Truncate the filter at this many standard deviations.
    """

    def __init__(self):
        """
        Initializes a new instance of the GaussianSmoother class.
        
        This constructor prepares the Gaussian Smoother for subsequent operations.
        Currently, it performs no specific initialization, deferring setup to later
        stages when data and smoothing parameters are available.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Why:
            The GaussianSmoother is initialized without any specific parameters.
            This allows for a flexible setup where data and smoothing parameters
            can be provided later, adapting the smoother to different datasets
            and noise levels, which is essential for discovering underlying
            differential equations from potentially noisy data.
        """
        pass

    def __call__(self, data, kernel_fun='gaussian', **kwargs):
        """
        Applies a Gaussian smoothing kernel to the input data.
        
                This method smooths the input data using a Gaussian filter, effectively reducing noise and highlighting underlying trends.
                The smoothing is applied either across the entire dataset or independently for each time point, depending on the input parameters.
        
                Args:
                    data (np.ndarray): The data to be smoothed.
                    kernel_fun (str, optional): The type of smoothing kernel to apply. Currently, only 'gaussian' is supported. Defaults to 'gaussian'.
                    **kwargs: Keyword arguments to pass to the Gaussian filter, such as 'sigma' (the standard deviation for Gaussian kernel) and 'include_time' (boolean flag to indicate whether to smooth across time).
        
                Returns:
                    np.ndarray: The smoothed data.
        """
        smoothed_data = np.empty_like(data)
        if kernel_fun == 'gaussian':
            if not kwargs['include_time'] and np.ndim(data) > 1:
                for time_idx in np.arange(data.shape[0]):
                    smoothed_data[time_idx, ...] = gaussian_filter(data[time_idx, ...],
                                                                   sigma=kwargs['sigma'])
            else:
                smoothed_data = gaussian_filter(data, sigma=kwargs['sigma'])
        else:
            raise NotImplementedError(
                'Wrong kernel passed into function. Current version supports only Gaussian smoothing.')

        return smoothed_data
