#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _clean_text_form(text):
    """Clean raw factor parameter strings like '{power: 1.0}' from text_form."""
    # Remove '{power: 1.0}' (power of 1 is implicit)
    text = re.sub(r'\{power:\s*1\.0\}', '', text)
    # Convert '{power: N}' to '^N'
    text = re.sub(r'\{power:\s*([^}]+)\}', r'^{\1}', text)
    # Remove other single-param braces like '{freq: 2.0}' -> keep as (freq=2.0)
    text = re.sub(r'\{(\w+):\s*([^}]+)\}', r'(\1=\2)', text)
    return text


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
                        raw = annotations[front_idx][front_elem_idx]
                        # Split multi-equation latex into individual equations
                        raw = raw.replace('\\begin{eqnarray*} ', '').replace('\\end{eqnarray*}', '')
                        raw = raw.replace('{power: ', '^{')
                        parts = [p.strip() for p in raw.split('\\\\') if p.strip()]
                        annotation = '\n'.join(['$' + p.rstrip(', ') + '$' for p in parts])
                        plt.annotate(
                            annotation,
                            xy=(x, y),
                            xytext=(x + front_elem_idx * 0.1 * np.sign(x), y + front_elem_idx * 0.1),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->", lw=0.5, color='gray'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.5, alpha=0.8),
                            fontsize=12,
                            ha='left' if x > 0 else 'right'
                        ).draggable()

        if filename is not None:
            plt.savefig(filename + '.' + save_format, dpi=300, quality=94, format=save_format)
        plt.show()

    def plot_pareto_per_equation(self, plot_level=1, annotate_best=True,
                                 filename=None, save_format='eps'):
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = False

        SMALL_SIZE = 12
        mpl.rc('font', size=SMALL_SIZE)
        mpl.rc('axes', titlesize=SMALL_SIZE)

        # Collect all solutions across plotted levels, preserving system identity
        all_solutions = []
        for front_idx in range(min(plot_level, len(self.pareto_frontier.levels))):
            for solution in self.pareto_frontier.levels[front_idx]:
                all_solutions.append((front_idx, solution))

        if len(all_solutions) == 0:
            return

        first_solution = all_solutions[0][1]
        var_names = first_solution.vals.equation_keys
        n_eq = len(var_names)
        n_obj_total = len(first_solution.obj_fun)
        n_obj_types = n_obj_total // n_eq

        assert n_obj_types >= 2, (
            f'Need at least 2 objective types per equation for 2D plot, '
            f'got {n_obj_types} (total objectives: {n_obj_total}, equations: {n_eq})')

        # Assign colors: same system gets same color across all subplots
        n_systems = len(all_solutions)
        cmap = plt.cm.get_cmap('tab20', max(n_systems, 2))
        system_colors = [cmap(i) for i in range(n_systems)]

        fig, axes = plt.subplots(1, n_eq, figsize=(7 * n_eq, 6), squeeze=False)
        axes = axes[0]

        for eq_idx in range(n_eq):
            ax = axes[eq_idx]
            dim_x = eq_idx           # first objective type for this equation
            dim_y = eq_idx + n_eq    # second objective type for this equation

            # Plot each system as a point with its assigned color
            front0_idx = 0
            for sys_idx, (front_idx, solution) in enumerate(all_solutions):
                x = solution.obj_fun[dim_x]
                y = solution.obj_fun[dim_y]
                ax.scatter(x, y, color=system_colors[sys_idx], s=50,
                           edgecolors='k' if front_idx == 0 else 'none',
                           linewidths=0.5, zorder=3)

                if annotate_best and front_idx == 0:
                    try:
                        raw = solution.vals[var_names[eq_idx]].latex_form
                        raw = raw.replace('{power: ', '^{')
                        annotation = '$' + raw + '$'
                    except (AttributeError, KeyError):
                        annotation = str((round(x, 4), round(y, 4)))
                    # Offset in points, staggered by front-0 index to avoid overlap
                    offset_x = 20
                    offset_y = 20 + front0_idx * 30
                    ax.annotate(
                        annotation,
                        xy=(x, y),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", lw=0.5, color='gray'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.5, alpha=0.8),
                        fontsize=9,
                        ha='left'
                    ).draggable()
                    front0_idx += 1

            # Set axis limits excluding zeros (filter to positive values if possible)
            all_x = [s.obj_fun[dim_x] for _, s in all_solutions]
            all_y = [s.obj_fun[dim_y] for _, s in all_solutions]
            nonzero_x = [v for v in all_x if v > 0]
            nonzero_y = [v for v in all_y if v > 0]

            if nonzero_x:
                ax.set_xlim(left=min(nonzero_x) * 0.8)
            if nonzero_y:
                ax.set_ylim(bottom=min(nonzero_y) * 0.8)

            ax.set_title(f'Equation for {var_names[eq_idx]}')
            ax.set_xlabel('Objective 1 (fitness)')
            ax.set_ylabel('Objective 2')
            ax.grid(True)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename + '.' + save_format, dpi=300, format=save_format)
        plt.show()
