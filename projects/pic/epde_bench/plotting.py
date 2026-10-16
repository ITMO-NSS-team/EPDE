"""Quick looks at a :class:`Problem` -- one function for every problem class."""

import numpy as np


def plot_problem(problem, data=None, max_panels: int = 4):
    """ODE/system: time series. 1-D PDE: the (t, x) field. 2-D and 3-D PDE:
    snapshots at a few times (3-D: the middle slice of the last axis).
    Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    data = problem.data if data is None else data
    t = np.unique(np.asarray(problem.grids[0]))
    variables = list(data)

    if problem.dim == 0:
        fig, axes = plt.subplots(len(variables), 1, figsize=(9, 2.2 * len(variables)), squeeze=False)
        for ax, var in zip(axes[:, 0], variables):
            ax.plot(t, data[var], lw=1)
            ax.set_ylabel(var)
        axes[-1, 0].set_xlabel(problem.axis_names[0])
    elif problem.dim == 1:
        x = np.unique(np.asarray(problem.grids[1]))
        fig, axes = plt.subplots(1, len(variables), figsize=(5.5 * len(variables), 4), squeeze=False)
        for ax, var in zip(axes[0], variables):
            im = ax.imshow(data[var].T, origin='lower', aspect='auto', cmap='RdBu_r',
                           extent=[t[0], t[-1], x[0], x[-1]])
            ax.set_xlabel(problem.axis_names[0])
            ax.set_ylabel(problem.axis_names[1])
            ax.set_title(var)
            fig.colorbar(im, ax=ax)
    else:
        moments = np.linspace(0, len(t) - 1, min(max_panels, len(t))).astype(int)
        fig, axes = plt.subplots(len(variables), len(moments),
                                 figsize=(3.2 * len(moments), 2.8 * len(variables)), squeeze=False)
        for row, var in enumerate(variables):
            field = data[var]
            if field.ndim == 4:                                  # (t, x, y, z): middle z slice
                field = field[..., field.shape[-1] // 2]
            vmin, vmax = np.nanpercentile(field, [1, 99])
            for col, k in enumerate(moments):
                ax = axes[row, col]
                ax.imshow(field[k], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'{var}, {problem.axis_names[0]} = {t[k]:.3g}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(f'{problem.name}: {problem.title}')
    fig.tight_layout()
    return fig


def plot_front(objectives, labels=None, title='Pareto front'):
    """Scatter of the first two objectives of the final front."""
    import matplotlib.pyplot as plt
    pts = np.array([o[:2] for o in objectives if o is not None and len(o) >= 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1])
        for i, (a, b) in enumerate(pts):
            ax.annotate(str(i) if labels is None else labels[i], (a, b), fontsize=8)
    ax.set_xlabel('objective 1 (discrepancy)')
    ax.set_ylabel('objective 2')
    ax.set_title(title)
    fig.tight_layout()
    return fig
