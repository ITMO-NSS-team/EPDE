#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:55:34 2021

@author: mike_ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt

class ParetoVisualizer(object):
    def __init__(self, eq_pareto_levels):
        '''
            Проще всего получить pareto_levels из атрибута optimizer.pareto_levels
        '''
        self.pareto_frontier = eq_pareto_levels

    def plot_pareto(self, dimensions: tuple = (0, 1), annotate_best=True, plot_level = 1, 
                    filename = None, save_format = 'eps'):
        assert len(
            dimensions) == 2, 'The pareto levels are projected on the 2D plane, therefore only 2 coordinates are processible'
        coords = [[(solution.obj_fun[dimensions[0]], solution.obj_fun[dimensions[1]]) for solution in self.pareto_frontier.levels[front_idx]]
                  for front_idx in np.arange(len(self.pareto_frontier.levels))]
        if annotate_best:
            annotations = [[solution.latex_form for solution in self.pareto_frontier.levels[front_idx]]
                       for front_idx in np.arange(len(self.pareto_frontier.levels))]
        coords_arrays = []
        for coord_set in coords:
            coords_arrays.append(np.array(coord_set))

        colors = ['r', 'k', 'b', 'y', 'g'] + \
            ['m' for idx in np.arange(len(coords_arrays) - 5)]
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.grid()
        for front_idx in np.arange(min(len(coords_arrays), plot_level)):
            ax.scatter(coords_arrays[front_idx][:, 0],
                       coords_arrays[front_idx][:, 1], color=colors[front_idx])
            for front_elem_idx in np.arange(coords_arrays[front_idx].shape[0]):
                if front_idx == 0 or not annotate_best:
                    if annotate_best:
                        annotation = annotations[front_idx][front_elem_idx]
                        if annotation[0] != '$':
                            annotation = '$' + annotation + '$'
                            ax.annotate(annotations[front_idx][front_elem_idx], 
                                        (coords_arrays[front_idx][front_elem_idx, 0], coords_arrays[front_idx][front_elem_idx, 1]),
                                        fontsize = 'xx-large')


        if filename is not None:
            plt.savefig(filename + '.' + save_format, dpi = 300, quality = 94, format=save_format)
        fig.show()
