#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _clean_text_form(text):
    """Clean raw factor parameter strings like '{power: 1.0}' from text_form."""
    # Remove '{power: 1}' / '{power: 1.0}' (power of 1 is implicit)
    text = re.sub(r'\{power:\s*1(\.0+)?\}', '', text)
    # Convert '{power: N}' to '^{N}', trimming a trailing '.0' from float powers
    text = re.sub(r'\{power:\s*(\d+)(\.0+)?\}', r'^{\1}', text)
    text = re.sub(r'\{power:\s*([^}]+)\}', r'^{\1}', text)
    # Remove other single-param braces like '{freq: 2.0}' -> keep as (freq=2.0)
    text = re.sub(r'\{(\w+):\s*([^}]+)\}', r'(\1=\2)', text)
    return text


def _annotation_text_box(annot, renderer):
    """Display-space bbox of the annotation's TEXT BOX only.

    ``Annotation.get_window_extent`` unions the text with the arrow down to
    the anchored data point, so it keeps "overlapping" neighbors however far
    the text is moved -- useless for collision checks. The FancyBboxPatch
    around the text is what visually collides.
    """
    patch = annot.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer)
    return mpl.text.Text.get_window_extent(annot, renderer)


def _resolve_annotation_overlaps(fig, annotations, max_iter=50, step=12):
    """Greedy de-overlap pass for point-attached annotations: while any two
    annotation TEXT boxes intersect in display space, push the later one
    further along its own vertical offset direction. Bounded by ``max_iter``
    sweeps; silently skipped on backends without an accessible renderer.
    """
    if len(annotations) < 2:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return

    for _ in range(max_iter):
        moved = False
        boxes = [_annotation_text_box(annot, renderer) for annot in annotations]
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if boxes[i].overlaps(boxes[j]):
                    offset_x, offset_y = annotations[j].xyann
                    direction = 1 if offset_y >= 0 else -1
                    annotations[j].xyann = (offset_x, offset_y + direction * step)
                    moved = True
        if not moved:
            break
        fig.canvas.draw()


def _choose_axis_scale(values):
    """Pick an axis scale for a set of plotted objective values.

    Returns ``(scale, scale_kwargs)`` for ``Axes.set_xscale``/``set_yscale``:

    - ``'log'`` when every value is positive and the positive spread exceeds
      two orders of magnitude (typical for discrepancy-based fitness);
    - ``'symlog'`` when zero (or negative) values coexist with such a wide
      positive spread -- keeps zero-valued solutions visible instead of
      pushing them off a log axis;
    - ``'linear'`` otherwise.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 'linear', {}
    positive = [v for v in finite if v > 0]
    if len(positive) >= 2 and max(positive) / min(positive) > 100:
        if len(positive) == len(finite):
            return 'log', {}
        return 'symlog', {'linthresh': 0.5 * min(positive)}
    return 'linear', {}


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
        plt.close()

    def plot_pareto_per_equation(self, dimensions: tuple = (0, 1), plot_level=1,
                                 annotate_best=True, filename=None,
                                 save_format='eps', show=True):
        """Plot the Pareto levels per equation of the system: one FIGURE per
        variable, with the objective types from ``dimensions`` on the x/y
        axes (``obj_fun`` is laid out grouped by type:
        ``[type0_eq0, type0_eq1, ..., type1_eq0, ...]``).

        Equations repeated across systems -- e.g. several Pareto-optimal
        systems sharing one u-equation while differing in the v-equation --
        are plotted and annotated ONCE, represented by their lowest-front
        occurrence. Annotations alternate above/below their points with
        growing offsets (and stay draggable for manual cleanup). Axis
        scales follow ``_choose_axis_scale`` (log for wide-ranged positive
        fitness, symlog when zeros must stay visible, linear otherwise).

        With ``filename`` given, each figure is saved as
        ``{filename}_{var}.{save_format}``. Returns the list of created
        figures (or None for an empty front); with ``show=False`` the
        figures are neither shown nor closed, so callers/tests can inspect
        or save them.
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = False

        SMALL_SIZE = 12
        mpl.rc('font', size=SMALL_SIZE)
        mpl.rc('axes', titlesize=SMALL_SIZE)

        assert len(dimensions) == 2, (
            'The pareto levels are projected on the 2D plane, therefore '
            'exactly two objective types are processible')

        # Collect all solutions across plotted levels, lowest front first.
        all_solutions = []
        for front_idx in range(min(plot_level, len(self.pareto_frontier.levels))):
            for solution in self.pareto_frontier.levels[front_idx]:
                all_solutions.append((front_idx, solution))

        if len(all_solutions) == 0:
            return None

        first_solution = all_solutions[0][1]
        var_names = first_solution.vals.equation_keys
        n_eq = len(var_names)
        n_obj_total = len(first_solution.obj_fun)
        n_obj_types = n_obj_total // n_eq

        assert n_obj_types >= 2, (
            f'Need at least 2 objective types per equation for 2D plot, '
            f'got {n_obj_types} (total objectives: {n_obj_total}, equations: {n_eq})')
        assert max(dimensions) < n_obj_types, (
            f'Requested objective types {dimensions}, but only {n_obj_types} '
            f'types are present')

        cmap = plt.get_cmap('tab10')
        figures = []

        for eq_idx in range(n_eq):
            var = var_names[eq_idx]
            dim_x = dimensions[0] * n_eq + eq_idx
            dim_y = dimensions[1] * n_eq + eq_idx

            # Deduplicate per figure: several systems can carry the SAME
            # equation for this variable -- plot each unique equation once,
            # keyed by its structural fingerprint. dict preserves insertion
            # order, so the lowest-front occurrence is the representative.
            unique_equations = {}
            for front_idx, solution in all_solutions:
                equation = solution.vals[var]
                try:
                    key = equation.terms_labels
                except AttributeError:
                    key = getattr(equation, 'text_form', repr(equation))
                if key not in unique_equations:
                    unique_equations[key] = (front_idx, solution)

            entries = list(unique_equations.values())

            fig, ax = plt.subplots(figsize=(8, 6))

            front0_points = []
            front0_idx = 0
            for front_idx, solution in entries:
                x = solution.obj_fun[dim_x]
                y = solution.obj_fun[dim_y]
                if front_idx == 0:
                    ax.scatter(x, y, color=cmap(front0_idx % 10), s=60,
                               edgecolors='k', linewidths=0.5, zorder=3)
                    front0_points.append((x, y, solution))
                    front0_idx += 1
                else:
                    # Deeper fronts: light-gray context points, unannotated.
                    ax.scatter(x, y, color='0.7', s=35, zorder=2)

            if annotate_best and front0_points:
                # Position-aware annotation placement: boxes lean toward
                # the plot interior -- below points in the upper half of
                # the cloud, above points in the lower half, and to the
                # left of points near the right edge -- so they never
                # escape past the title or the axes. Within each side the
                # distance is staggered to keep neighbors from piling up;
                # boxes stay draggable for manual cleanup.
                y_med = float(np.median([p[1] for p in front0_points]))
                x_med = float(np.median([p[0] for p in front0_points]))
                x_max_pt = max(p[0] for p in front0_points)
                side_counts = {1: 0, -1: 0}
                placed_annotations = []
                for x, y, solution in front0_points:
                    try:
                        cleaned = _clean_text_form(solution.vals[var].latex_form)
                        annotation = '$' + cleaned + '$' if cleaned.strip() \
                            else str((round(x, 4), round(y, 4)))
                    except (AttributeError, KeyError):
                        annotation = str((round(x, 4), round(y, 4)))
                    side = -1 if y > y_med else 1
                    offset_y = side * (25 + 18 * side_counts[side])
                    side_counts[side] += 1
                    at_right_edge = x == x_max_pt and x > x_med
                    offset_x = -15 if at_right_edge else 15
                    annot = ax.annotate(
                        annotation,
                        xy=(x, y),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", lw=0.5, color='gray'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  lw=0.5, alpha=0.8),
                        fontsize=9,
                        ha='right' if at_right_edge else 'left'
                    )
                    annot.draggable()
                    placed_annotations.append(annot)

            # Scale selection over everything plotted in this figure; no
            # manual limits -- zero-valued solutions must stay in view.
            xs = [s.obj_fun[dim_x] for _, s in entries]
            ys = [s.obj_fun[dim_y] for _, s in entries]
            x_scale, x_kwargs = _choose_axis_scale(xs)
            y_scale, y_kwargs = _choose_axis_scale(ys)
            ax.set_xscale(x_scale, **x_kwargs)
            ax.set_yscale(y_scale, **y_kwargs)
            ax.margins(x=0.1, y=0.15)

            ax.set_title(f'Equation for {var}')
            x_hint = ' (fitness)' if dimensions[0] == 0 else ''
            y_hint = ' (complexity)' if dimensions[1] == 1 else ''
            ax.set_xlabel(f'Objective {dimensions[0]}{x_hint}')
            ax.set_ylabel(f'Objective {dimensions[1]}{y_hint}')
            ax.grid(True, which='both', alpha=0.4)

            fig.tight_layout()
            if annotate_best and front0_points:
                _resolve_annotation_overlaps(fig, placed_annotations)

            if filename is not None:
                fig.savefig(f'{filename}_{var}.{save_format}', dpi=300,
                            format=save_format, bbox_inches='tight')
            figures.append(fig)

        if show:
            plt.show()
            for fig in figures:
                plt.close(fig)
        return figures
