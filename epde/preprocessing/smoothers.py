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
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data, *args, **kwargs):
        raise NotImplementedError('Calling abstract smoothing object')


class PlaceholderSmoother(AbstractSmoother):
    def __init__(self):
        pass

    def __call__(self, data, *args, **kwargs):
        return data


def baseline_ann(dim):
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
    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
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
        """ This function applies a rational function to each element of X.
        ------------------------------------------------------------------------
        Arguments:
        X: A tensor. We apply the rational function to every element of X.
        ------------------------------------------------------------------------
        Returns:
        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

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
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        x = torch.sin(x)
        return x

class NN(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int          = 1,    # Dimension of the input
                 Output_Dim          : int          = 1,    # Dimension of the output
                 Data_Type           : torch.dtype  = torch.float32,
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Tanh",
                 Batch_Norm          : bool         = False):
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
        """ Forward method for the NN class. Note that the user should NOT call
        this function directly. Rather, they should call it through the __call__
        method (using the NN object like a function), which is part of the
        module class and calls forward.

        ------------------------------------------------------------------------
        Arguments:

        X: A batch of inputs. This should be a B by Input_Dim tensor, where B
        is the batch size. The ith row of X should hold the ith input.

        ------------------------------------------------------------------------
        Returns:

        If X is a B by Input_Dim tensor, then the output of this function is a
        B by Output_Dim tensor, whose ith row holds the value of the network
        applied to the ith row of X. """

        # If we are using batch normalization, then normalize the inputs.
        if (self.Batch_Norm == True):
            X = self.Norm_Layer(X);

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (with no activation function) and return.
        return self.Layers[self.Num_Hidden_Layers](X);

# class ANNSmoother(AbstractSmoother):
#     def __init__(self):
#         pass
#
#     def __call__(self, data, grid, epochs_max=1e3, loss_mean=1000, batch_frac=0.5, learining_rate=1e-2, return_ann: bool = False, device = 'cpu'):
#         dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim
#         model = baseline_ann(dim)
#         # model = NN(Num_Hidden_Layers=1, Neurons_Per_Layer=1024, Input_Dim=dim, Activation_Function='Rational')
#         # model = NN(Num_Hidden_Layers=4, Neurons_Per_Layer=256, Input_Dim=dim, Activation_Function='Sin')
#         grid_flattened = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grid])).float().T
#
#         original_shape = data.shape
#
#         field_ = torch.from_numpy(data.reshape(-1, 1)).float()
#
#         # device = torch.device(device)
#         grid_flattened.to(device)
#         field_.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=learining_rate)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs_max//10, gamma=0.5)
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
#         batch_size = int(data.size * batch_frac)
#
#         t = 0
#
#         min_loss = np.inf
#         while loss_mean > 1e-5 and t < epochs_max:
#
#             permutation = torch.randperm(grid_flattened.size()[0])
#
#             loss_list = []
#
#             for i in range(0, grid_flattened.size()[0]-1, batch_size):
#                 optimizer.zero_grad()
#
#                 indices = permutation[i:i+batch_size]
#                 batch_x, batch_y = grid_flattened[indices], field_[indices]
#
#                 loss = torch.mean(torch.abs(batch_y-model(batch_x)))
#
#                 loss.backward()
#                 optimizer.step()
#                 loss_list.append(loss.item())
#             scheduler.step()
#             loss_mean = np.mean(loss_list)
#             if loss_mean < min_loss:
#                 best_model = model
#                 min_loss = loss_mean
#
#         model.load_state_dict(best_model)
#         model.eval()
#
#         with torch.no_grad():
#             prediction = model(grid_flattened).cpu().numpy().reshape(original_shape)
#
#         if return_ann:
#             warn('Returning ANN from smoother. This should only happen in selected experiments.')
#             return prediction, model
#         else:
#             return prediction

class ANNSmoother(AbstractSmoother):
    def __init__(self):
        super().__init__()  # Optional depending on AbstractSmoother
        self.model = None

    def __call__(self, data, grid, epochs_max=1000, loss_mean=1000, loss_threshold=1e-8,
                 batch_frac=0.5, val_frac=0.2, learning_rate=1e-3, return_ann=False, device='cpu'):

        # Convert to int if passed as float
        epochs_max = int(epochs_max)

        # Infer input dimension
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim

        # Initialize model
        # model = baseline_ann(dim).to(device)
        model = NN(Num_Hidden_Layers=5, Neurons_Per_Layer=50, Input_Dim=dim, Activation_Function='Rational').to(device)
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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_max // 10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max // 10)

        # Loss function
        loss_fn = torch.nn.MSELoss()

        # Batch size
        batch_size = max(1, int(data.size * batch_frac))

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
                # pred = model(batch_x)
                # loss = loss_fn(pred, batch_y)
                loss = torch.mean(torch.abs(batch_y - model(batch_x)))
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())

            train_loss = np.mean(train_loss_list)
            scheduler.step(train_loss)

            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = loss_fn(val_pred, val_y).item()

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

        with torch.no_grad():
            prediction = model(grid_flattened).cpu().numpy().reshape(original_shape)

        if return_ann:
            warn('Returning ANN from smoother. This should only happen in selected experiments.')
            return prediction, model
        else:
            return prediction


class GaussianSmoother(AbstractSmoother):
    def __init__(self):
        pass

    def __call__(self, data, kernel_fun='gaussian', **kwargs):
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
