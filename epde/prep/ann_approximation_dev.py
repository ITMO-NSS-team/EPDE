#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:53:24 2020

@author: mike_ubuntu
"""
import time
from collections import Counter

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import copy
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')


def approximator_shallow(*args):
    data = tf.stack(args, axis = 1, name = 'data_stacked')

    with tf.name_scope("UA_fit"):   
        w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
        b1 = tf.get_default_graph().get_tensor_by_name("b1:0")
        w_out = tf.get_default_graph().get_tensor_by_name("w_out:0")        
        
        ws1 = tf.matmul(data, w1) + b1
        a = tf.nn.sigmoid(ws1)
        ws_out = tf.matmul(a, w_out)
    return ws_out

def approximator_deep(*args):
    data = tf.stack(args, axis = 1, name = 'data_stacked')
    data = tf.squeeze(data)
    with tf.name_scope("UA_fit"): 
        tf.disable_eager_execution()
        w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
        b1 = tf.get_default_graph().get_tensor_by_name("b1:0")
        w2 = tf.get_default_graph().get_tensor_by_name("w2:0")
        b2 = tf.get_default_graph().get_tensor_by_name("b2:0")
        w3 = tf.get_default_graph().get_tensor_by_name("w3:0")
        b3 = tf.get_default_graph().get_tensor_by_name("b3:0")
        w4 = tf.get_default_graph().get_tensor_by_name("w4:0")
        b4 = tf.get_default_graph().get_tensor_by_name("b4:0")
        w5 = tf.get_default_graph().get_tensor_by_name("w5:0")
        b5 = tf.get_default_graph().get_tensor_by_name("b5:0")        
        w_out = tf.get_default_graph().get_tensor_by_name("w_out:0")
        print(data.shape, w1.shape)
        ws1 = tf.matmul(data, w1) + b1
        a1 = tf.nn.sigmoid(ws1) 

        ws2 = tf.matmul(a1, w2) + b2
        a2 = tf.nn.sigmoid(ws2)

        ws3 = tf.matmul(a2, w3) + b3
        a3 = tf.nn.sigmoid(ws3)


        ws4 = tf.matmul(a3, w4) + b4
        a4 = tf.nn.sigmoid(ws4)

        ws5 = tf.matmul(a4, w5) + b5
        a5 = tf.nn.sigmoid(ws5)

        ws_out = tf.matmul(a5, w_out)
    return ws_out            
        
def approximate_ann_shallow(data):
    hidden_layers = 50
    input_layers = data.shape[-1]
    output_dim = 1
    
    with tf.name_scope("UA_fit"):
        w1 = tf.get_variable(name = "w1", dtype = tf.float32, shape=[input_layers, hidden_layers], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b1 = tf.get_variable(name = "b1", dtype = tf.float32, shape=[hidden_layers], 
                            initializer=tf.constant_initializer(0.))
        
        ws1 = tf.matmul(data, w1) + b1
        a = tf.nn.sigmoid(ws1)
        
        w_out = tf.get_variable(name = "w_out", dtype = tf.float32, shape=[hidden_layers, output_dim], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        ws_out = tf.matmul(a, w_out)
#    print(w1.name)        
    return ws_out    

        
def approximate_ann_deep(data):
    hidden_layers_1 = 256
    hidden_layers_2 = 512
    hidden_layers_3 = 256
    hidden_layers_4 = 128
    hidden_layers_5 = 64
    
    input_layers = data.shape[-1]
    output_dim = 1
    
    with tf.name_scope("UA_fit"):
        tf.disable_eager_execution()
        w1 = tf.get_variable(name = "w1", dtype = tf.float32, shape=[input_layers, hidden_layers_1], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b1 = tf.get_variable(name = "b1", dtype = tf.float32, shape=[hidden_layers_1], 
                            initializer=tf.constant_initializer(0.))
        
        w2 = tf.get_variable(name = "w2", dtype = tf.float32, shape=[hidden_layers_1, hidden_layers_2], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b2 = tf.get_variable(name = "b2", dtype = tf.float32, shape=[hidden_layers_2], 
                            initializer=tf.constant_initializer(0.))

        w3 = tf.get_variable(name = "w3", dtype = tf.float32, shape=[hidden_layers_2, hidden_layers_3], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b3 = tf.get_variable(name = "b3", dtype = tf.float32, shape=[hidden_layers_3], 
                            initializer=tf.constant_initializer(0.))

        w4 = tf.get_variable(name = "w4", dtype = tf.float32, shape=[hidden_layers_3, hidden_layers_4], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b4 = tf.get_variable(name = "b4", dtype = tf.float32, shape=[hidden_layers_4], 
                            initializer=tf.constant_initializer(0.))

        w5 = tf.get_variable(name = "w5", dtype = tf.float32, shape=[hidden_layers_4, hidden_layers_5], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b5 = tf.get_variable(name = "b5", dtype = tf.float32, shape=[hidden_layers_5], 
                            initializer=tf.constant_initializer(0.))
        
        
        ws1 = tf.matmul(data, w1) + b1
        a1 = tf.nn.sigmoid(ws1)
        
        ws2 = tf.matmul(a1, w2) + b2
        a2 = tf.nn.sigmoid(ws2)

        ws3 = tf.matmul(a2, w3) + b3
        a3 = tf.nn.sigmoid(ws3)
        
        ws4 = tf.matmul(a3, w4) + b4
        a4 = tf.nn.sigmoid(ws4)

        ws5 = tf.matmul(a4, w5) + b5
        a5 = tf.nn.sigmoid(ws5)        
        
        w_out = tf.get_variable(name = "w_out", dtype = tf.float32, shape=[hidden_layers_5, output_dim], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        ws_out = tf.matmul(a5, w_out)
    print(w1.name, w1.shape)        
    return ws_out    

class Differentiable_Function(object):
    def __init__(self, function, history = []):
        self.function = function
        self.deriv_history = history
        
    def differentiate(self, data, axes_names, orders = 1):
        gradient = tf.gradients(ys=self.function, xs=data)
        if orders == 1:
            deriv_history = [self.deriv_history + [axis_name] for axis_name in axes_names] 
            return gradient, deriv_history
        else:
            derivatives = []; deriv_history = []
            for idx in np.arange(len(data)):
                hist = self.deriv_history + [axes_names[idx]]
                temp_fun = Differentiable_Function(gradient[idx], history = hist)
                der_fun, history = temp_fun.differentiate(data, axes_names, orders = orders - 1)
                derivatives.extend(der_fun); deriv_history.extend(history)
            return derivatives, deriv_history

class Approximator(object):
    def __init__(self, data, steps, **kwargs):
        self.training_session = tf.Session()
        self.data = data; 
        self.steps = steps
        
        self.data_grid = np.indices(self.data.shape)
        self.dgr = np.transpose(self.data_grid.reshape((self.data_grid.shape[0], np.prod(self.data_grid.shape[1:]))))
        
        self.coords = [self.data_grid[idx]*self.steps[idx] for idx in np.arange(len(self.steps))]
        self.coords = np.stack(self.coords, axis=-1)
        self.approximation = None
        
        self.set_interp_function(self.coords, self.data)

    def grid_function(self, coords, coord_eps = 1e-10):
        for idx, vals in np.ndenumerate(self.coords[..., 0]): 
            if np.all(np.abs(self.coords[idx] - np.array(coords)) < coord_eps):
                return self.data[idx]        
        return -np.inf        
        
    def train(self, batch_proportion = 0.9, epochs = None, loss_bnd = None):       
        x = tf.placeholder(tf.float32, shape = [None, self.data.ndim], name="x")
        y_appr = approximate_ann_deep(x)
        y_true = tf.placeholder(tf.float32, shape = [None, 1], name="y_true")
        
        with tf.variable_scope('Loss'):
            loss = tf.losses.mean_squared_error(y_true, y_appr)

        adam = tf.train.AdamOptimizer(learning_rate=1e-2)
        train_optimizer = adam.minimize(loss)

        self.training_session.run(tf.global_variables_initializer())
        if epochs is not None and loss_bnd is None:        
            for epoch in np.arange(epochs):
                dgr = copy.deepcopy(self.dgr)
                np.random.shuffle(dgr)
                idx_train, idx_val = dgr[:int(dgr.shape[0]*batch_proportion), :], dgr[int(dgr.shape[0]*batch_proportion):, :]            
                x_train = np.matrix([self.coords[tuple(idx)] for idx in idx_train])
                x_val = np.matrix([self.coords[tuple(idx)] for idx in idx_val])
                y_train = [self.interp_function(self.coords[tuple(idx)]) for idx in idx_train]
                y_train = np.asarray(y_train).reshape((len(y_train), 1))
                y_val = [self.interp_function(self.coords[tuple(idx)]) for idx in idx_val]
                y_val = np.asarray(y_val).reshape((len(y_val), 1))
                print(y_train.shape)
    
                feed_t = {x: x_train, y_true: y_train}; feed_val = {x: x_val, y_true: y_val}
                current_loss, _, x_cur, y_cur, y_approx = self.training_session.run([loss, train_optimizer,x, y_true, y_appr], feed_dict = feed_t)
                val_loss = self.training_session.run([loss], feed_dict = feed_val)
                w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
                print('w1 shape', w1.shape)
                print(epoch, ' loss on train: ', current_loss, ',', ' loss on val (after weight adjustment): ', val_loss[0])
        elif epochs is None and loss_bnd is not None:
            val_loss = np.inf
            while val_loss > loss_bnd:
                dgr = copy.deepcopy(self.dgr)
                np.random.shuffle(dgr)
                idx_train, idx_val = dgr[:int(dgr.shape[0]*batch_proportion), :], dgr[int(dgr.shape[0]*batch_proportion):, :]            
                x_train = np.matrix([self.coords[tuple(idx)] for idx in idx_train])
                x_val = np.matrix([self.coords[tuple(idx)] for idx in idx_val])
                y_train = [self.interp_function(self.coords[tuple(idx)]) for idx in idx_train]
                y_train = np.asarray(y_train).reshape((len(y_train), 1))
                y_val = [self.interp_function(self.coords[tuple(idx)]) for idx in idx_val]
                y_val = np.asarray(y_val).reshape((len(y_val), 1))
                print(y_train.shape)
    
                feed_t = {x: x_train, y_true: y_train}; feed_val = {x: x_val, y_true: y_val}
                current_loss, _, x_cur, y_cur, y_approx = self.training_session.run([loss, train_optimizer,x, y_true, y_appr], feed_dict = feed_t)
                val_loss = self.training_session.run([loss], feed_dict = feed_val)
                w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
                print('w1 shape', w1.shape)

                print(epoch, ' loss on train: ', current_loss, ',', ' loss on val (after weight adjustment): ', val_loss[0])
            
        else:
            raise NotImplementedError('Epochs number or loss function value must be set')
        w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
        print(w1.shape)

    def apply(self, dims = ['t', 'x', 'y', 'z']):
        # self.dims = dims        
        self.inputs = self.coords.reshape((np.prod(self.coords.shape[:-1]), self.coords.shape[-1]))
        self.coords_tensors = []
        for dim_idx in range(self.dgr.shape[-1]):
            var_name = dims[dim_idx] + '_coords'
            self.coords_tensors.append(tf.placeholder(dtype = tf.float32, shape = (self.dgr.shape[0],), name = var_name))  
        
        feed = {}
        for dim_idx in range(self.dgr.shape[-1]):
            feed[self.coords_tensors[dim_idx]] = self.inputs[:, dim_idx]

        self.approximation = approximator_deep(*self.coords_tensors)  
        u = self.training_session.run([self.approximation], feed_dict = feed)
        return u[0].reshape(self.data.shape)
    
    def get_derivatives(self, order, axes = ['t', 'x', 'y' ,'z']):
        if type(self.approximation) == type(None):
            self.inputs = self.coords.reshape((np.prod(self.coords.shape[:-1]), self.coords.shape[-1]))
            self.coords_tensors = []
            for dim_idx in range(self.dgr.shape[-1]):
                var_name = axes[dim_idx] + '_coords'
                self.coords_tensors.append(tf.placeholder(dtype = tf.float32, shape = (self.dgr.shape[0],), name = var_name))  
            self.approximation = approximator_deep(*self.coords_tensors)
            
        feed = {}
        for dim_idx in range(self.dgr.shape[-1]):
            feed[self.coords_tensors[dim_idx]] = self.inputs[:, dim_idx]
        
        ax = axes[:self.data.ndim]
        diff = Differentiable_Function(test_approx.approximation)
        
        compare_ders = lambda ds1, ds2: Counter(ds1) == Counter(ds2)
        
        deriv_combined = [[], []]
        deriv_unfiltered = diff.differentiate(self.coords_tensors, ax, order)
        print('obtained', deriv_unfiltered)
        for deriv_idx in np.arange(len(deriv_unfiltered[0])):
            repeating = False
            for prev_symbolic in deriv_combined[1]:
                if compare_ders(deriv_unfiltered[1][deriv_idx], prev_symbolic):
                    repeating = True
            if not repeating: 
                deriv_evaluated = self.training_session.run([deriv_unfiltered[0][deriv_idx]], feed_dict = feed)
                deriv_combined[0].append(deriv_evaluated); deriv_combined[1].append(deriv_unfiltered[1][deriv_idx])
        return deriv_combined
    
    
    def interp_function(self, coords):
        try:
            return self._interp(coords)[0]
        except IndexError:
            return np.float(self._interp(coords))
        

    def set_interp_function(self, data_coords, data, eps = 1e-9):
        data_coords = [np.linspace(0, self.steps[idx]*(self.data.shape[idx] - 1), self.data.shape[idx]) for idx in np.arange(self.data.ndim)]
        for dim in np.arange(len(data_coords)):
            data_coords[dim][0] -= eps; data_coords[dim][-1] += eps
                
        print(data_coords)
        self._interp = RegularGridInterpolator(data_coords, data, method = 'nearest')
        print('interpolator set')
    
    
if __name__ == '__main__':

    def ic_1(x):
        x_max = 5.
        return np.sin(x/x_max*np.pi)*np.sin((5-x)/x_max*np.pi)
    
    
    def ic_2(x):
        x_max = 5.; coeff = -1
        return coeff * np.sin(x/x_max*np.pi)*np.sin((5-x)/x_max*np.pi)
    
    
    
    x_shape = 301; t_shape = 301
    x_max = 5; t_max = 1
    x_vals = np.linspace(0, x_max, x_shape)
    t_vals = np.linspace(0, t_max, t_shape)
    delta_x = x_vals[1] - x_vals[0]; delta_t = t_vals[1] - t_vals[0]
    k = 0.5
    
    solution = np.empty((t_shape, x_shape))
    solution[:, 0] = solution[:, -1] = 0
    solution[0, :] = ic_1(x_vals)
    solution[1, :] = solution[0, :] + delta_t * ic_2(x_vals)
    
    for t in np.arange(2, t_shape):
        for x in np.arange(1, x_shape - 1):
            solution[t, x] = k*delta_t**2/delta_x**2 * (solution[t-1, x+1] - 2*solution[t-1, x] + solution[t-1, x-1]) + 2*solution[t-1, x] - solution[t-2, x]
    
    
    tf.reset_default_graph()
    test_approx = Approximator(solution, (delta_t, delta_x))
    test_approx.train(batch_proportion = 0.8, epochs=4)
    sol_approximation = test_approx.apply()
    derivatives = test_approx.get_derivatives(order = 2)
    
#    
#def plot_comparison()
#    inputs = test_approx.coords.reshape((np.prod(test_approx.coords.shape[:-1]), test_approx.coords.shape[-1]))
#    
#    dims = ['x', 'y', 'z', 't']
#    coords = []
#    for dim_idx in range(test_approx.dgr.shape[-1]):
#        var_name = dims[dim_idx] + '_coords'
#        coords.append(tf.placeholder(dtype = tf.float32, shape = (test_approx.dgr.shape[0],), name = var_name))
#    z_approx = approximator_deep(*coords)    
#
#    feed = {}
#    for dim_idx in range(test_approx.dgr.shape[-1]):
#        feed[coords[dim_idx]] = inputs[:, dim_idx]
#    u = test_approx.training_session.run([z_approx], feed_dict = feed)        
#
        