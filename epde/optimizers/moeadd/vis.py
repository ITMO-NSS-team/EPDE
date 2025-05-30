#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class ParetoVisualizer(object):
    def __init__(self, eq_pareto_levels):
        """
        Проще всего получить pareto_levels из атрибута optimizer.pareto_levels
        """
        self.pareto_frontier = eq_pareto_levels

    def plot_pareto(self, dimensions: tuple = (0, 1), annotate_best=True, plot_level=1,
                    filename=None, save_format='eps'):

        #TODO replace latex with Mathtext
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = True

        SMALL_SIZE = 12
        mpl.rc('font', size=SMALL_SIZE)
        mpl.rc('axes', titlesize=SMALL_SIZE)

        assert len(
            dimensions) == 2, 'The pareto levels are projected on the 2D plane, therefore only 2 coordinates are processible'
        coords = [[(solution.obj_fun[dimensions[0]], solution.obj_fun[dimensions[1]])
                   for solution in self.pareto_frontier.levels[front_idx]]
                  for front_idx in np.arange(plot_level)]  # len(self.pareto_frontier.levels))]
        if annotate_best:
            try:
                annotations = [[solution.latex_form for solution in self.pareto_frontier.levels[front_idx]]
                               for front_idx in np.arange(len(self.pareto_frontier.levels))]
            except AttributeError:
                annotations = [[str(solution.obj_fun) for solution in self.pareto_frontier.levels[front_idx]]
                               for front_idx in np.arange(len(self.pareto_frontier.levels))]

        coords_arrays = []
        for coord_set in coords:
            coords_arrays.append(np.array(coord_set))

        colors = ['r', 'k', 'b', 'y', 'g'] + \
                 ['m' for idx in np.arange(len(coords_arrays) - 5)]

        if len(coords_arrays) > 1:
            x_min = min(*[np.min(coord_arr[:, 0]) for coord_arr in coords_arrays])
            x_max = max(*[np.max(coord_arr[:, 0]) for coord_arr in coords_arrays])
            y_min = min(*[np.min(coord_arr[:, 1]) for coord_arr in coords_arrays])
            y_max = max(*[np.max(coord_arr[:, 1]) for coord_arr in coords_arrays])
        else:
            x_min = np.min(coords_arrays[0][:, 0]);
            x_max = np.max(coords_arrays[0][:, 0])
            y_min = np.min(coords_arrays[0][:, 1]);
            y_max = np.max(coords_arrays[0][:, 1])

        x_interval = max(x_max - x_min, 5)
        y_interval = max(y_max - y_min, 2)

        plt.grid()
        plt.xlim(x_min - 0.1 * x_interval, x_max + 0.8 * x_interval)  # ax.set_
        plt.ylim(y_min - 0.1 * y_interval, y_max + 0.3 * y_interval)  # ax.set_

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')

        for front_idx in np.arange(min(len(coords_arrays), plot_level)):

            plt.scatter(coords_arrays[front_idx][:, 0],
                        coords_arrays[front_idx][:, 1], color=colors[front_idx])
            locs_used = []
            for front_elem_idx in np.arange(coords_arrays[front_idx].shape[0]):
                if any([all(np.isclose(np.array((coords_arrays[front_idx][front_elem_idx, 0],
                                                 coords_arrays[front_idx][front_elem_idx, 1])), entry))
                        for entry in locs_used]):
                    continue
                else:
                    locs_used.append(np.array((coords_arrays[front_idx][front_elem_idx, 0],
                                               coords_arrays[front_idx][front_elem_idx, 1])))
                if front_idx == 0 or not annotate_best:
                    if annotate_best:
                        annotation = annotations[front_idx][front_elem_idx]
                        if annotation[0] != r'$':
                            annotation = r'$' + annotation + r'$'
                            print(annotation)
                            plt.annotate(annotations[front_idx][front_elem_idx],
                                         (coords_arrays[front_idx][front_elem_idx, 0] + 0.4,
                                          coords_arrays[front_idx][front_elem_idx, 1] + 0.2),
                                         bbox=dict(boxstyle="Square,pad=0.3",
                                                   fc="white", lw=0.5))  # ,
                            # fontsize = 'xx-large')

    def plot_pareto_mt(self, dimensions: tuple = (0, 1), annotate_best=True, plot_level=1,
                    filename=None, save_format='eps'):
        # Reset to default matplotlib settings
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering

        SMALL_SIZE = 12
        mpl.rc('font', size=SMALL_SIZE)
        mpl.rc('axes', titlesize=SMALL_SIZE)

        assert len(
            dimensions) == 2, 'The pareto levels are projected on the 2D plane, therefore only 2 coordinates are processible'

        # Prepare coordinates
        coords = [[(solution.obj_fun[dimensions[0]], solution.obj_fun[dimensions[1]])
                   for solution in self.pareto_frontier.levels[front_idx]]
                  for front_idx in np.arange(plot_level)]

        if annotate_best:
            try:
                annotations = [[solution.latex_form for solution in self.pareto_frontier.levels[front_idx]]
                               for front_idx in np.arange(len(self.pareto_frontier.levels))]
            except AttributeError:
                annotations = [[str(solution.obj_fun) for solution in self.pareto_frontier.levels[front_idx]]
                               for front_idx in np.arange(len(self.pareto_frontier.levels))]

        coords_arrays = []
        for coord_set in coords:
            coords_arrays.append(np.array(coord_set))

        colors = ['r', 'k', 'b', 'y', 'g'] + \
                 ['m' for idx in np.arange(len(coords_arrays) - 5)]

        # Calculate plot boundaries
        if len(coords_arrays) > 1:
            x_min = min(*[np.min(coord_arr[:, 0]) for coord_arr in coords_arrays])
            x_max = max(*[np.max(coord_arr[:, 0]) for coord_arr in coords_arrays])
            y_min = min(*[np.min(coord_arr[:, 1]) for coord_arr in coords_arrays])
            y_max = max(*[np.max(coord_arr[:, 1]) for coord_arr in coords_arrays])
        else:
            x_min = np.min(coords_arrays[0][:, 0])
            x_max = np.max(coords_arrays[0][:, 0])
            y_min = np.min(coords_arrays[0][:, 1])
            y_max = np.max(coords_arrays[0][:, 1])

        x_interval = max(x_max - x_min, 5)
        y_interval = max(y_max - y_min, 2)

        plt.grid()
        plt.xlim(x_min - 0.1 * x_interval, x_max + 0.8 * x_interval)
        plt.ylim(y_min - 0.1 * y_interval, y_max + 0.3 * y_interval)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')

        for front_idx in np.arange(min(len(coords_arrays), plot_level)):
            plt.scatter(coords_arrays[front_idx][:, 0],
                        coords_arrays[front_idx][:, 1], color=colors[front_idx])

            locs_used = []
            for front_elem_idx in np.arange(coords_arrays[front_idx].shape[0]):
                if any([all(np.isclose(np.array((coords_arrays[front_idx][front_elem_idx, 0],
                                                 coords_arrays[front_idx][front_elem_idx, 1])), entry))
                        for entry in locs_used]):
                    continue
                else:
                    locs_used.append(np.array((coords_arrays[front_idx][front_elem_idx, 0],
                                               coords_arrays[front_idx][front_elem_idx, 1])))

                if front_idx == 0 or not annotate_best:
                    if annotate_best:
                        x, y = coords_arrays[front_idx][front_elem_idx, 0], coords_arrays[front_idx][front_elem_idx, 1]
                        annotation = annotations[front_idx][front_elem_idx].replace('\\begin{eqnarray*} ', '$')
                        annotation = annotation.replace('\\end{eqnarray*}', '$')
                        annotation = annotation.replace('{power: ', '^{')
                        # plt.annotate(annotation,
                        #              (coords_arrays[front_idx][front_elem_idx, 0] + 0.4,
                        #               coords_arrays[front_idx][front_elem_idx, 1] + 0.2),
                        #              bbox=dict(boxstyle="Square,pad=0.3",
                        #                        fc="white", lw=0.5), fontsize=10)
                        plt.annotate(
                            annotation,
                            xy=(x, y),
                            xytext=(x + front_elem_idx * 0.1 * np.sign(x), y + front_elem_idx * 0.1),  # Offset direction depends on x position
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->", lw=0.5, color='gray'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.5, alpha=0.8),
                            fontsize=12,
                            ha='left' if x > 0 else 'right'
                        ).draggable()

        if filename is not None:
            plt.savefig(filename + '.' + save_format, dpi=300, quality=94, format=save_format)
        plt.show()
