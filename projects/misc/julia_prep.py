#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:40:09 2023

@author: maslyaev
"""

import numpy as np
from epde.interface.interface import EpdeSearch

from epde.supplementary import define_derivatives
from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

if __name__ == "__main__":
    t_max = 200
    # x = np.load('convection_x.npy')
    x = np.linspace(0, 4*np.pi, 100)
    
    # t = np.load('convection_t.npy')[:t_max]
    u = -np.sin(x) + np.sqrt(np.power(np.sin(x), 2) + 2./3.)
    du_analytical = -np.cos(x) + np.cos(x) * np.sin(x) * np.power(np.power(np.sin(x), 2) + 2./3., -0.5)
    
    u_n = u + np.random.normal(scale = 0.1 * u, size = u.size)
    
    default_preprocessor_type = 'poly'
    preprocessor_kwargs = {'use_smoothing' : False,
                            'include_time' : True}
    # preprocessor_kwargs = {'epochs_max' : 10000}
    
    setup = PreprocessorSetup()
    builder = ConcretePrepBuilder()
    setup.builder = builder
    
    if default_preprocessor_type == 'ANN':
        setup.build_ANN_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'poly':
        setup.build_poly_diff_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'spectral':
        setup.build_spectral_preprocessing(**preprocessor_kwargs)
    else:
        raise NotImplementedError('Incorrect default preprocessor type. Only ANN or poly are allowed.')
    preprocessor_pipeline = setup.builder.prep_pipeline

    if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
        preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None
        
    max_order = (2,)
    deriv_names, deriv_orders = define_derivatives('u', dimensionality=u.ndim,
                                                   max_order=max_order)

    data_tensor, derivatives = preprocessor_pipeline.run(u, grid=[x,],
                                                         max_order=max_order)
    data_tensor_n, derivatives_n = preprocessor_pipeline.run(u_n, grid=[x,],
                                                             max_order=max_order)   
    
# np.linalg.norm(u - data_tensor_n)/np.linalg.norm(u)
# Out[40]: 0.04486079554885309

# np.linalg.norm(derivatives[:, 1] - derivatives_n[:, 1])/np.linalg.norm(derivatives[:, 1])
# Out[41]: 1.1130077

# np.linalg.norm(derivatives[:, 0] - derivatives_n[:, 0])/np.linalg.norm(derivatives[:, 0])
# Out[42]: 0.26487035    