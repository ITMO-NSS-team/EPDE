#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:55:34 2021

@author: mike_ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt

class pareto_viusalizer(object):
    def __init__(self, eq_pareto_levels):
        '''
            Проще всего получить pareto_levels из атрибута optimizer.pareto_levels
        '''
        self.pareto_frontier = eq_pareto_levels
        
    def plot_pareto(self, dimensions : tuple = (0, 1), annotate_best = True):
        assert len(dimensions) == 2, 'The pareto levels are projected on the 2D plane, therefore only 2 coordinates are processible'
        coords = [[(solution.obj_fun[dimensions[0]], solution.obj_fun[dimensions[1]]) for solution in self.pareto_frontier.levels[front_idx]]
                    for front_idx in np.arange(len(self.pareto_frontier.levels))]
        annotations = [[solution.latex_form for solution in self.pareto_frontier.levels[front_idx]] for front_idx in np.arange(len(self.pareto_frontier.levels))]
        coords_arrays = []
        for coord_set in coords:
            coords_arrays.append(np.array(coord_set))    
        
        colors = ['r', 'k', 'b', 'y', 'g'] + ['m' for idx in np.arange(len(coords_arrays) - 5)]
        fig, ax = plt.subplots()
        for front_idx in np.arange(len(coords_arrays)):
            ax.scatter(coords_arrays[front_idx][:, 0], coords_arrays[front_idx][:, 1], color = colors[front_idx])
            for front_elem_idx in np.arange(coords_arrays[front_idx].shape[0]):
                if front_idx == 0 or not annotate_best:
                    annotation = annotations[front_idx][front_elem_idx]
                    if annotation[0] != '$':
                        annotation = '$' + annotation + '$'
                    ax.annotate(annotations[front_idx][front_elem_idx], 
                                (coords_arrays[front_idx][front_elem_idx, 0], coords_arrays[front_idx][front_elem_idx, 1]),
                                fontsize = 'xx-large')

        fig.show()                
        