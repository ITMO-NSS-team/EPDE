#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:07:12 2021

@author: maslyaev
"""

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import Trigonometric_tokens

if __name__ == '__main__':
    t_max = 400

    file = np.loadtxt('Data_32_points_.dat', 
                      delimiter=' ', usecols=range(33))

    x = np.linspace(0.5, 16, 32)
    t = file[:t_max, 0]
    grids = np.meshgrid(t, x, indexing = 'ij')
    u = file[:t_max, 1:]

    boundary = 2
    dimensionality = u.ndim    
    
    epde_search_obj_derivs = epde_alg.epde_search(dimensionality=dimensionality)
    epde_search_obj_derivs.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary,
                                additional_tokens=[], method='ANN', 
                                method_kwargs = {'epochs_max':200}, 
                                coordinate_tensors = grids)
    pool = epde_search_obj_derivs.pool # В пулле токенов только производные
    
    _, example_token = pool.create()
    # Метод epde.interface.token_family.TokenFamily.create() возвращает 2 объекта: лист названий 
    # токенов, которые "блокируются" сгенерированным токеном (на основе его свойств: уникальности/уникальности семейства). 
    # Вторым возвращаемым объектом является токен, представленный через объект epde.factor.Factor.
    # Описание класса множителя (соответствующего токену. Тут небольшие проблемы с терминологией)
    # смотри в гайде

    trig_tokens = Trigonometric_tokens()
    epde_search_obj = epde_alg.epde_search(dimensionality=dimensionality)
    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary,
                                additional_tokens=[trig_tokens], method='ANN', 
                                method_kwargs = {'epochs_max':200}, 
                                coordinate_tensors = grids)
    pool = epde_search_obj.pool # В пулле токенов производные и тригонометрические ф-ции
    _, example_token_sin = pool.create('sin')
        