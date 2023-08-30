#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:15:12 2022

@author: maslyaev
"""

from warnings import warn
from abc import ABC
from scipy.ndimage import gaussian_filter
import numpy as np

import torch
device = torch.device('cpu')


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
        torch.nn.Tanh(),
        torch.nn.Linear(256, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1024),
        torch.nn.Tanh(),
        torch.nn.Linear(1024, 1)
    )
    return model


class ANNSmoother(AbstractSmoother):
    def __init__(self):
        pass

    def __call__(self, data, grid, epochs_max=1e3, loss_mean=1000, batch_frac=0.5,
                 learining_rate=1e-4, return_ann: bool = False):
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim
        model = baseline_ann(dim)
        grid_flattened = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grid])).float().T

        original_shape = data.shape

        field_ = torch.from_numpy(data.reshape(-1, 1)).float()
        grid_flattened.to(device)
        field_.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learining_rate)

        batch_size = int(data.size * batch_frac)

        t = 0

        min_loss = np.inf
        while loss_mean > 1e-5 and t < epochs_max:

            permutation = torch.randperm(grid_flattened.size()[0])

            loss_list = []

            for i in range(0, grid_flattened.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = grid_flattened[indices], field_[indices]

                loss = torch.mean(torch.abs(batch_y-model(batch_x)))

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)
            if loss_mean < min_loss:
                best_model = model
                min_loss = loss_mean
            print('Surface training t={}, loss={}'.format(t, loss_mean))
            t += 1

        data_approx = best_model(grid_flattened).detach().numpy().reshape(original_shape)
        if return_ann:
            warn('Returning ANN from smoother. This should not occur anywhere, except selected experiments.')
            return data_approx, best_model
        else:
            return data_approx


class GaussianSmoother(AbstractSmoother):
    def __init__(self):
        pass

    def __call__(self, data, kernel_fun='gaussian', **kwargs):
        # print('kwargs', kwargs)
        smoothed_data = np.empty_like(data)
        if kernel_fun == 'gaussian':
            if kwargs['include_time']:
                print('full smoothing')
                smoothed_data = gaussian_filter(data, sigma=kwargs['sigma'])
            elif np.ndim(data) > 1:
                for time_idx in np.arange(data.shape[0]):
                    smoothed_data[time_idx, ...] = gaussian_filter(data[time_idx, ...],
                                                                   sigma=kwargs['sigma'])
            else:
                smoothed_data = gaussian_filter(data, sigma=kwargs['sigma'])
        else:
            raise NotImplementedError(
                'Wrong kernel passed into function. Current version supports only Gaussian smoothing.')

        return smoothed_data
