#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:11:05 2022

@author: maslyaev
"""

import numpy as np
import time
import warnings

import matplotlib.pyplot as plt

import multiprocessing as mp
from typing import Union

from abc import ABC
from epde.preprocessing.cheb import process_point_cheb

# Spawning a multiprocessing Pool only pays off for large grids. Under the
# spawn start method (Windows/macOS-default) every worker re-imports the caller,
# which costs seconds per worker; for a few-hundred-point ODE that dwarfs the
# actual per-point Chebyshev fit. Below this many points run the serial map
# instead -- identical output, no spawn overhead, and no dependence on the
# caller guarding ``__main__``.
_MP_MIN_POINTS = 3000

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
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
    plt.title(title)
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')
    plt.show()
    plt.close(fig)

def axis_orders(max_order, ndim: int) -> list:
    """``max_order`` (one int, or one per axis; a single-entry sequence counts for every axis) as
    one int per axis."""
    if np.ndim(max_order) == 0:
        return [int(max_order)] * ndim
    orders = [int(order) for order in max_order]
    if len(orders) == 1:
        return orders * ndim
    if len(orders) != ndim:
        raise ValueError(f'max_order needs one entry per axis ({ndim}), got {len(orders)}.')
    return orders


class AbstractDeriv(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list], *args, **kwargs):
        raise NotImplementedError('Calling abstract differentiation object')

    def edge_reach(self, data: np.ndarray, max_order: Union[int, list], **kwargs):
        """How many samples from each end of every axis this calculator's derivatives are
        corrupted by its edge treatment (one int per axis), given the data and the kwargs it is
        called with; ``boundary_width='auto'`` excludes them. None: unknown (a calculator that does
        not say)."""
        return None


#: Orders of accuracy of the central differences: ``np.gradient`` (2), the five-point stencil (4).
FD_ACCURACIES = (2, 4)


def _check_accuracy(accuracy) -> int:
    if accuracy not in FD_ACCURACIES:
        raise ValueError(f'accuracy must be one of {FD_ACCURACIES}, got {accuracy!r}.')
    return int(accuracy)


def _five_point_applies(grid) -> bool:
    """The five-point stencil needs five samples on a uniform grid."""
    grid = np.asarray(grid, dtype=float)
    if grid.size < 5:
        return False
    steps = np.diff(grid)
    return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))


def _central4(data: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
    """Fourth-order central differences along ``axis`` (the five-point stencil) on a uniform grid;
    the two samples at each end take ``np.gradient``'s second-order rules (``edge_order=2``)."""
    grid = np.asarray(grid, dtype=float)
    steps = np.diff(grid)
    if grid.size < 5:
        raise ValueError(f'accuracy=4 needs at least 5 samples along axis {axis}, got {grid.size}.')
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        raise ValueError(f'accuracy=4 needs a uniform grid along axis {axis}.')
    h = float(np.mean(steps))
    u = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    out = np.gradient(u, h, axis=0, edge_order=2)
    out[2:-2] = (-u[4:] + 8.0 * u[3:-1] - 8.0 * u[1:-3] + u[:-4]) / (12.0 * h)
    return np.moveaxis(out, 0, axis)


class AdaptiveFiniteDeriv(AbstractDeriv):
    """Finite differences, nested once per order: central along every axis, by ``np.gradient``
    (``accuracy=2``) or the five-point stencil (``accuracy=4``; the ``FD`` preprocessor's default).
    An axis the stencil cannot serve -- fewer than five samples, or uneven spacing -- takes the
    second-order rule, with an ``EPDEUsageWarning``. The one-sided time rules
    (``ForwardIncrementDeriv``, ``BackwardIncrementDeriv``) keep these on the other axes."""

    def __init__(self):
        pass

    def edge_reach(self, data: np.ndarray, max_order: Union[int, list], accuracy: int = 2,
                   **kwargs) -> list:
        """``np.gradient`` is first-order accurate at the two end samples, and order ``k`` nests
        ``k`` gradients, each pulling the inexact samples one step further in: ``k`` per end
        (measured: exactly ``k`` on smooth fields, orders 1-4). The five-point stencil falls back
        to second order on two samples per end: ``2 k``."""
        reach = axis_orders(max_order, np.ndim(data))
        return reach if _check_accuracy(accuracy) == 2 else [2 * k for k in reach]

    def _grad(self, data: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
        """First derivative along ``axis``: the one place the difference rule is chosen."""
        if getattr(self, '_accuracy', 2) == 4:
            if _five_point_applies(grid):
                return _central4(data, grid, axis)
            if axis not in self.__dict__.setdefault('_second_order_axes', set()):
                self._second_order_axes.add(axis)
                import epde.globals as global_var
                warnings.warn(f'FD accuracy 4: axis {axis} has {np.size(grid)} samples'
                              f'{"" if np.size(grid) < 5 else " on an uneven grid"}; the five-point stencil '
                              'needs five evenly spaced samples, so this axis takes the second-order rule.',
                              global_var.EPDEUsageWarning, stacklevel=2)
        return np.gradient(data, grid, axis=axis)

    def differentiate(self, data: np.ndarray, max_order: Union[int, list],
                      mixed: bool = False, axis=None, *grids) -> list:
        if isinstance(max_order, int):
            max_order = [max_order,] * data.ndim
        if any([ord_ax != max_order[0] for ord_ax in max_order]) and mixed:
            raise Exception(
                'Mixed derivatives can be taken only if the orders are same along all axes.')
        if data.ndim != len(grids) and (len(grids) != 1 or data.ndim != 2):
            print(data.ndim, len(grids))
            raise ValueError('Data dimensionality does not fit passed grids.')

        if len(grids) == 1 and data.ndim == 2:
            assert np.any(data.shape) == 1
            derivs = []
            dim_idx = 0 if data.shape[1] == 1 else 1
            grad = self._grad(data, grids[0], dim_idx)
            derivs.append(grad)
            ord_marker = max_order[dim_idx] if axis is None else max_order[axis]
            if ord_marker > 1:
                higher_ord_der_axis = None if mixed else dim_idx
                ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                derivs.extend(self.differentiate(grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
        else:
            derivs = []
            if axis == None:
                for dim_idx in np.arange(data.ndim):
                    if max_order[dim_idx] == 0:
                        continue
                    grad = self._grad(data, grids[dim_idx], dim_idx)
                    derivs.append(grad)
                    ord_marker = max_order[dim_idx] if axis is None else max_order[axis]
                    if ord_marker > 1:
                        higher_ord_der_axis = None if mixed else dim_idx
                        ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                        derivs.extend(self.differentiate(
                            grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
            else:
                grad = self._grad(data, grids[axis], axis)
                derivs.append(grad)
                if max_order[axis] > 1:
                    ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                    derivs.extend(self.differentiate(
                        grad, ord_reduced, False, axis, *grids))
        return derivs

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mixed: bool = False, accuracy: int = 2) -> np.ndarray:
        self._accuracy = _check_accuracy(accuracy)
        self._second_order_axes = set()             # the axes already warned about, this call
        grid_unique = [np.unique(ax_grid) for ax_grid in grid]

        derivs = self.differentiate(data, max_order, mixed, None, *grid_unique)
        derivs = np.vstack([der.reshape(-1) for der in derivs]).T
        return derivs


class ForwardIncrementDeriv(AdaptiveFiniteDeriv):
    """Forward-in-time (Ito) derivatives: ``(u[k+lag] - u[k]) / (t[k+lag] - t[k])`` along the
    time axis, central differences along every other axis.

    A time-symmetric derivative cannot see a drift that is the gradient of a function of the
    state once the process is stationary: ``E[u' g(u)] = d/dt E[G(u)] = 0`` for ``g = G'``. So
    for stochastic data (Langevin dynamics, active particles) confinement and repulsion fit
    with coefficient ~0 under the central or polynomial preprocessors. A forward increment is
    not symmetric in time, and ``E[u(t+h) - u(t) | u(t)] / h`` is the drift itself (the first
    Kramers-Moyal coefficient). ``lag`` > 1 averages over measurement noise.

    Use with an UNSMOOTHED state, as ``FD`` with ``scheme='forward'`` does: a smoother mixes
    future samples into ``u(t)`` and correlates it with the increment it is regressed against.

    The last ``lag * order`` samples along the time axis have no forward increment. They
    repeat the last defined value and must lie inside the domain boundary;
    ``Trajectory.build`` refuses a narrower boundary (see ``required_boundary``).
    """
    #: Where along the time axis the increment is undefined (for the boundary guard's message).
    undefined_end = 'last'

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mixed: bool = False, lag: int = 1, time_axis: int = 0, accuracy: int = 2) -> np.ndarray:
        if int(lag) != lag or lag < 1:
            raise ValueError(f'lag must be a positive integer number of samples, got {lag!r}.')
        self._lag, self._time_axis = int(lag), time_axis
        return super().__call__(data, grid, max_order, mixed, accuracy=accuracy)

    def _grad(self, data: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
        if axis != self._time_axis:
            return super()._grad(data, grid, axis)
        h = self._lag
        if data.shape[axis] <= h:
            kind = 'forward' if self.undefined_end == 'last' else 'backward'
            raise ValueError(f'{data.shape[axis]} samples along the time axis leave no {kind} '
                             f'increment at lag {h}.')
        u = np.moveaxis(data, axis, 0)
        step = (np.asarray(grid)[h:] - np.asarray(grid)[:-h]).reshape((-1,) + (1,) * (u.ndim - 1))
        return np.moveaxis(self._increment(u, step, h), 0, axis)

    @staticmethod
    def _increment(u: np.ndarray, step: np.ndarray, h: int) -> np.ndarray:
        """``(u[k+h] - u[k]) / step`` at sample ``k``; the last ``h`` repeat the last defined one."""
        out = np.empty(u.shape, dtype=float)
        out[:-h] = (u[h:] - u[:-h]) / step
        out[-h:] = out[-h - 1]
        return out

    @staticmethod
    def required_boundary(lag: int, max_order: int) -> int:
        """Samples at the end of the time axis without an increment of order ``max_order``."""
        return int(lag) * int(max_order)

    def edge_reach(self, data: np.ndarray, max_order: Union[int, list], lag: int = 1,
                   time_axis: int = 0, accuracy: int = 2, **kwargs) -> list:
        """:meth:`required_boundary` on the time axis, central differences elsewhere."""
        reach = super().edge_reach(data, max_order, accuracy=accuracy)
        if time_axis < len(reach):
            reach[time_axis] = self.required_boundary(lag, axis_orders(max_order, np.ndim(data))[time_axis])
        return reach


class BackwardIncrementDeriv(ForwardIncrementDeriv):
    """Backward-in-time derivatives: ``(u[k] - u[k-lag]) / (t[k] - t[k-lag])`` along the time axis,
    central differences along every other axis (``FD`` with ``scheme='backward'``). The first
    ``lag * order`` samples have no backward increment; they repeat the first defined value and
    must lie inside the domain boundary, as the forward increment's last ones."""
    undefined_end = 'first'

    @staticmethod
    def _increment(u: np.ndarray, step: np.ndarray, h: int) -> np.ndarray:
        out = np.empty(u.shape, dtype=float)
        out[h:] = (u[h:] - u[:-h]) / step
        out[:h] = out[h]
        return out


# class ANNBasedFiniteDifferences(AbstractDeriv):
#     def __init__(self):
#         self._internal_ann = None

#     def differentiate(self, data: np.ndarray, max_order: Union[int, list],
#                       mixed: bool = False, axis=None, *grids) -> list:

class PolynomialDeriv(AbstractDeriv):
    '''
    Derivatives of local Chebyshev fits (``cheb.process_point_cheb``): at every sample, along every
    axis, a polynomial of degree ``poly_order`` (None: the axis' derivative order + 1) is fitted by
    least squares to the ``polynomial_window`` samples centred on it and differentiated there. The
    ``polynomial_window // 2 + 1`` samples nearest each end have no centred window and take
    one-sided 6-point finite differences instead (``fin_der.FDderivatives``, orders <= 5).

    :meth:`resolve` reconciles the window and the degree with the derivative orders before anything
    is fitted, so the two can not silently disagree.
    '''
    #: Today's window, kept by ``polynomial_window='auto'`` whenever it fits.
    DEFAULT_WINDOW = 9
    #: The highest order the one-sided edge stencils of ``fin_der.FDderivatives`` provide.
    MAX_EDGE_ORDER = 5
    #: Samples one edge stencil spans.
    EDGE_STENCIL = 6

    def __init__(self):
        pass

    @classmethod
    def _fits(cls, window: int, size: int) -> bool:
        # The edge samples take EDGE_STENCIL points inward from themselves: the last of the
        # window // 2 + 1 left-edge samples needs window // 2 + EDGE_STENCIL <= size.
        return window <= size and window // 2 + cls.EDGE_STENCIL <= size

    @classmethod
    def resolve(cls, max_order: Union[int, list, tuple], shape: tuple,
                polynomial_window='auto', poly_order=None) -> tuple:
        """``(window, poly_order)`` to differentiate data of ``shape`` up to ``max_order`` with, or
        ValueError when the setting conflicts with the orders:

        * an order above :attr:`MAX_EDGE_ORDER` (the edge stencils stop there);
        * an explicit ``poly_order`` below an axis' order (that derivative would be identically 0);
        * an even window (the centred window is mis-indexed), or one that does not fit an axis
          with a derivative (with its edge stencils);
        * a window below the degree + 1 (an underdetermined fit). A window equal to the degree is
          let through: ``cheb`` drops the degree by one, which still differentiates to the order.

        ``polynomial_window='auto'`` is :attr:`DEFAULT_WINDOW` or, for a higher degree, the smallest
        odd window >= degree + 2, shrunk to fit the shortest axis with a derivative.
        """
        orders = axis_orders(max_order, len(shape))
        sizes = [size for size, order in zip(shape, orders) if order > 0]
        top = max(orders) if sizes else 0
        if top > cls.MAX_EDGE_ORDER:
            raise ValueError(
                f"The 'poly' preprocessor differentiates up to order {cls.MAX_EDGE_ORDER}: its "
                f'one-sided edge stencils stop there, got max_deriv_order {tuple(orders)}. Use '
                "'FD' or 'spectral' for higher orders.")
        if poly_order is None or poly_order == 'auto':
            poly_order, degree = None, top + 1
        else:
            if int(poly_order) != poly_order or poly_order < top:
                raise ValueError(
                    f'poly_order={poly_order!r} is below the derivative order {top}: every '
                    'derivative above the polynomial degree is identically zero. Leave '
                    'poly_order=None (order + 1) or raise it.')
            degree = int(poly_order)
        if not sizes:
            return (cls.DEFAULT_WINDOW if polynomial_window == 'auto' else polynomial_window,
                    poly_order)
        shortest = min(sizes)
        if polynomial_window == 'auto':
            window = max(cls.DEFAULT_WINDOW, degree + 2 + (degree + 2 + 1) % 2)
            while window > 1 and not cls._fits(window, shortest):
                window -= 2
            if window < degree + 1 and not (window == degree and degree - 1 >= top):
                raise ValueError(
                    f"An axis of {shortest} samples is too short for the 'poly' preprocessor at "
                    f'derivative order {top} (a window of at least {degree + 1} samples plus the '
                    f'{cls.EDGE_STENCIL}-point edge stencils). Use \'FD\' or \'spectral\'.')
            return window, poly_order
        if int(polynomial_window) != polynomial_window or polynomial_window < 1 \
                or polynomial_window % 2 == 0:
            raise ValueError(
                f'polynomial_window must be an odd positive int (the window is centred on the '
                f"sample), got {polynomial_window!r}; or 'auto'.")
        window = int(polynomial_window)
        if not cls._fits(window, shortest):
            raise ValueError(
                f'polynomial_window={window} does not fit an axis of {shortest} samples: it '
                f'needs window <= samples and window // 2 + {cls.EDGE_STENCIL} <= samples (the '
                "edge stencils). Pass a smaller window or 'auto'.")
        if window < degree + 1 and not (window == degree and degree - 1 >= top):
            raise ValueError(
                f'polynomial_window={window} is too narrow for a degree-{degree} fit (needs at '
                f'least {degree + 1} samples, or {degree} with the degree dropped to {degree - 1} '
                f'>= the derivative order {top}).')
        return window, poly_order

    def edge_reach(self, data: np.ndarray, max_order: Union[int, list], polynomial_window='auto',
                   poly_order=None, **kwargs) -> list:
        """The one-sided band ``window // 2 + 1`` on every axis with a derivative, whatever the
        order: every order comes from the same window, and the band is the same width at every
        order (measured: exactly 5 for the window of 9, orders 1-4)."""
        window, _ = self.resolve(max_order, np.shape(data), polynomial_window, poly_order)
        return [window // 2 + 1 if order > 0 else 0
                for order in axis_orders(max_order, np.ndim(data))]

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mp_poolsize: int, polynomial_window='auto', poly_order=None) -> np.ndarray:
        polynomial_window, poly_order = self.resolve(max_order, np.shape(data),
                                                     polynomial_window, poly_order)
        polynomial_boundary = polynomial_window//2 + 1
        index_array = []

        for idx, _ in np.ndenumerate(data):
            index_array.append((idx, data, grid, polynomial_window, max_order,
                                polynomial_boundary, poly_order))

        if mp_poolsize > 1 and len(index_array) >= _MP_MIN_POINTS:
            pool = mp.Pool(mp_poolsize)
            derivatives = pool.map_async(process_point_cheb, index_array)
            pool.close()
            pool.join()
            derivatives = derivatives.get()
        else:
            derivatives = list(map(process_point_cheb, index_array))

        return np.array(derivatives)


class SpectralDeriv(AbstractDeriv):
    '''
    Fourier (spectral) derivatives along every axis, after https://github.com/KimKitsurag1,
    adapted to match AbstractDeriv. Output columns follow ``define_derivatives``: for each axis in
    turn, its orders 1 .. max_order (pure derivatives only, like ``AdaptiveFiniteDeriv``).

    Per axis, ``periodic`` is one of:

    * ``True``: plain FFT differentiation of the samples, exact to round-off for a smooth field
      sampled over whole periods with the last sample one spacing short of the period (the FFT grid
      convention).
    * ``'included'``: periodic with the last sample repeating the first; it is differentiated as
      above without it, and takes the first sample's derivative.
    * ``False``: the FFT of raw non-periodic samples wraps the jump between the two ends into every
      derivative (relative error 77 on exp(2t), 0.2 inside a Lorenz window even after cropping 10%
      of each end). Instead the field minus ``q(x) = a x + b x^2``, whose slope matches 4th-order
      one-sided differences at both ends, has zero end slopes, so its even (mirror) extension is
      C^2 across the seam; that extension is differentiated by FFT and ``q'``, ``q''`` are added
      back analytically. Interior error on Lorenz ~1e-9 (central differences: 5e-5). Needs >= 5
      samples along the axis. Orders >= 3 see the extension's jump in the third derivative.
    * ``'auto'`` (default): decided from the data, axis by axis, by the seam test of
      ``epde.integrate.data_profile`` (the basis solver's, so both read periodicity the same way):
      an axis is periodic when the field continues across the seam -- the jump between the last
      and the first sample, and the second difference across them -- no worse than it varies
      inside (largest first and second differences), under either endpoint convention ('excluded'
      preferred when both pass). Measured on the repository data: AC, Burgers, KdV and KS periodic
      in x, the wave (Dirichlet) and every NS wake axis and time axis not. A wrong "periodic" is
      the costly error (forcing the NS wake's y axis periodic raises the x-momentum residual
      10-20x), and the test errs the other way: a periodic axis it misses still gets the
      boundary-corrected derivative (~1e-5 on a smooth periodic field). The verdicts of the last
      call are kept in ``last_periodic``.

    ``n`` low-passes the derivative with a Butterworth filter of order ``steepness``: frequencies
    above the ``n``-th distinct non-negative frequency of the axis' own FFT (``n = 1`` is the mean,
    ``n = 2`` the first harmonic, ...) are damped by ``1 / (1 + (f / f_n)^(2 steepness))``. ``None``
    (default) applies no filter at all. ``n`` and ``periodic`` take one value for every axis or a
    sequence with one entry per axis (``None`` entries of ``n`` leave that axis unfiltered).

    The spacing is the mean step of each axis: the grid has to be uniform (``EpdeSearch`` warns
    when it is not).
    '''

    PERIODIC_OPTIONS = (False, True, 'included', 'auto')
    #: A seam ratio up to 1 + SEAM_TIE passes. A pure harmonic puts its largest first or second
    #: difference exactly at the seam (sin and cos sampled from 0), a tie at 1 that round-off would
    #: decide either way. Measured margins: periodic data 0.0004-0.81, the closest non-periodic axis
    #: (the NS wake's y) 1.54.
    SEAM_TIE = 1e-9

    def __init__(self):
        self.last_periodic = None

    @classmethod
    def axis_periodicity(cls, field: np.ndarray, axis: int):
        """The data-driven verdict for one axis: True ('excluded' convention), 'included' or False,
        by ``data_profile``'s seam ratio over all lines of ``field`` along ``axis`` (<= 1 passes,
        up to :attr:`SEAM_TIE`)."""
        from epde.integrate.data_profile import _seam_ratio
        lines = np.moveaxis(np.asarray(field, dtype=float), axis, -1)
        lines = lines.reshape(-1, lines.shape[-1])
        if lines.shape[1] < 4 or not np.all(np.isfinite(lines)):
            return False
        if _seam_ratio(lines, 'excluded') <= 1.0 + cls.SEAM_TIE:
            return True
        if _seam_ratio(lines, 'included') <= 1.0 + cls.SEAM_TIE:
            return 'included'
        return False

    @staticmethod
    def _per_axis(value, ndim: int, name: str) -> list:
        if value is None or np.ndim(value) == 0:
            return [value] * ndim
        value = list(value)
        if len(value) != ndim:
            raise ValueError(f'{name} needs one entry per axis ({ndim}), got {len(value)}.')
        return value

    def resolve_periodic(self, field: np.ndarray, periodic='auto') -> list:
        """``periodic`` (one value or one per axis) as one of True / 'included' / False per axis,
        'auto' entries decided by :meth:`axis_periodicity`."""
        resolved = []
        for axis, value in enumerate(self._per_axis(periodic, np.ndim(field), 'periodic')):
            if isinstance(value, (bool, np.bool_)):
                resolved.append(bool(value))
            elif isinstance(value, str) and value in ('included', 'auto'):
                resolved.append(self.axis_periodicity(field, axis) if value == 'auto' else 'included')
            else:
                raise ValueError(f'periodic entries must be one of {self.PERIODIC_OPTIONS}, got {value!r}.')
        return resolved

    #: The boundary-corrected (non-periodic) derivative's edge zone by order, measured on smooth
    #: fields against the error over the middle half: none at order 1 (the ends are as accurate as
    #: the interior) and the end sample at order 2.
    NON_PERIODIC_REACH = {1: 0, 2: 1}
    #: From order 3 the even extension's jump in the third derivative leaves an error decaying only
    #: as 1 / distance from the end (1e-1 next to it, 1e-2 thirty samples in, at every grid size),
    #: so no band makes those orders accurate; this share of the axis per end is excluded.
    HIGH_ORDER_FRACTION = 0.1

    def edge_reach(self, data: np.ndarray, max_order: Union[int, list], periodic='auto',
                   **kwargs) -> list:
        """0 on a periodic axis (resolved like :meth:`differentiate` does); :attr:`NON_PERIODIC_REACH`
        on a non-periodic one, and from order 3 :attr:`HIGH_ORDER_FRACTION` of the axis, with a
        warning."""
        data = np.asarray(data, dtype=float)
        reach = []
        flags = self.resolve_periodic(data, periodic)
        for axis, order in enumerate(axis_orders(max_order, data.ndim)):
            if order == 0 or flags[axis]:
                reach.append(0)
            elif order in self.NON_PERIODIC_REACH:
                reach.append(self.NON_PERIODIC_REACH[order])
            else:
                import warnings
                import epde.globals as global_var
                band = max(1, int(round(self.HIGH_ORDER_FRACTION * data.shape[axis])))
                warnings.warn(
                    f'Spectral derivatives of order {order} on the non-periodic axis {axis} are '
                    'inaccurate everywhere: the error decays only as 1 / distance from the ends. '
                    f'Their edge reach is taken as {band} samples per end; prefer \'FD\' or \'poly\' '
                    'for this order, or pass periodic=True if the axis is periodic.',
                    global_var.EPDEUsageWarning, stacklevel=2)
                reach.append(band)
        return reach

    @staticmethod
    def cutoff_frequency(n, size: int, spacing: float):
        """The ``n``-th distinct non-negative frequency of ``np.fft.fftfreq(size, spacing)`` (clipped
        to the Nyquist frequency), or None for ``n=None``."""
        if n is None:
            return None
        if int(n) != n or n < 2:
            raise ValueError(f'n has to be an integer >= 2 (n = 1 keeps only the mean), got {n!r}.')
        return (min(int(n), size // 2 + 1) - 1) / (size * spacing)

    @staticmethod
    def butterworth_filter(freqs: np.ndarray, cutoff, steepness) -> np.ndarray:
        """``freqs`` damped by the Butterworth low-pass of order ``steepness`` with cut-off frequency
        ``cutoff``; unchanged for ``cutoff=None``."""
        if cutoff is None:
            return freqs
        return freqs / (1 + (freqs / cutoff) ** (2 * steepness))

    @staticmethod
    def _end_slopes(lines: np.ndarray, spacing: float):
        """4th-order one-sided first derivatives at both ends of the last axis."""
        c = np.array([-25., 48., -36., 16., -3.]) / 12.
        return (lines[..., :5] @ c) / spacing, -(lines[..., ::-1][..., :5] @ c) / spacing

    def axis_derivatives(self, field: np.ndarray, spacing: float, axis: int, max_order: int,
                         n=None, steepness=1, periodic=False) -> list:
        """``[((axis,) * k, d^k field / dx_axis^k) for k = 1 .. max_order]``; ``periodic`` is True,
        'included' or False (resolve 'auto' with :meth:`axis_periodicity` first)."""
        if max_order < 1:
            return []
        f = np.moveaxis(np.asarray(field, dtype=float), axis, -1)
        size = f.shape[-1]
        if periodic == 'included':
            f_open = np.moveaxis(f[..., :-1], -1, axis)
            derivs = self.axis_derivatives(f_open, spacing, axis, max_order, n, steepness, True)
            return [(key, np.concatenate([d, np.take(d, [0], axis=axis)], axis=axis)) for key, d in derivs]
        cutoff = self.cutoff_frequency(n, size, spacing)
        if periodic:
            work, poly = f, None
        else:
            if size < 5:
                raise ValueError(f'The non-periodic spectral derivative needs at least 5 samples along '
                                 f'axis {axis}, got {size}; pass periodic=True for this axis if it is.')
            x = np.arange(size) * spacing
            s0, s1 = self._end_slopes(f, spacing)
            a, b = s0[..., None], ((s1 - s0) / (2 * x[-1]))[..., None]
            rest = f - (a * x + b * x * x)
            work, poly = np.concatenate([rest, rest[..., -2:0:-1]], axis=-1), (a, b, x)
        m = work.shape[-1]
        k = 2j * np.pi * self.butterworth_filter(np.fft.fftfreq(m, d=spacing), cutoff, steepness)
        projection = np.fft.fft(work, axis=-1)
        derivs = []
        for order in range(1, max_order + 1):
            factor = k ** order
            if order % 2 == 1 and m % 2 == 0:
                factor[m // 2] = 0.      # an odd derivative of the Nyquist mode is not real: drop it
            deriv = np.real(np.fft.ifft(projection * factor, axis=-1))[..., :size]
            if poly is not None and order <= 2:
                a, b, x = poly
                deriv = deriv + (a + 2 * b * x if order == 1 else 2 * b)
            derivs.append(((axis,) * order, np.moveaxis(deriv, -1, axis)))
        return derivs

    def differentiate(self, field: np.ndarray, grid: list, max_order: Union[int, list],
                      mixed: bool = False, n=None, steepness=1, periodic='auto') -> dict:
        """``{(axis,) * order: derivative}`` in ``define_derivatives`` order. ``grid`` holds one
        coordinate vector per axis."""
        if mixed:
            raise NotImplementedError(
                'SpectralDeriv computes pure derivatives only: EPDE offers no mixed-derivative '
                'tokens (supplementary.define_derivatives).')
        if steepness <= 0:
            raise ValueError(f'steepness has to be positive, got {steepness!r}.')
        orders = self._per_axis(max_order, field.ndim, 'max_order')
        ns = self._per_axis(n, field.ndim, 'n')
        self.last_periodic = periodic = self.resolve_periodic(field, periodic)
        derivatives = {}
        for axis in range(field.ndim):
            coords = np.asarray(grid[axis], dtype=float)
            spacing = (coords[-1] - coords[0]) / (coords.size - 1)
            for key, deriv in self.axis_derivatives(field, spacing, axis, int(orders[axis]), ns[axis],
                                                    steepness, periodic[axis]):
                derivatives[key] = deriv
        return derivatives

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list],
                 mixed: bool = False, n=None, steepness=1, periodic='auto') -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if isinstance(grid, np.ndarray) and grid.ndim == 1 and data.ndim == 1:
            grid = [grid]
        coords = [np.unique(np.asarray(ax_grid)) for ax_grid in grid]      # as AdaptiveFiniteDeriv
        if len(coords) != data.ndim or any(c.size != s for c, s in zip(coords, data.shape)):
            raise ValueError(f'The grid ({[c.size for c in coords]} coordinates per axis) does not fit '
                             f'the data of shape {data.shape}.')
        derivatives = self.differentiate(data, coords, max_order, mixed, n, steepness, periodic).values()
        return np.vstack([der.reshape(-1) for der in derivatives]).T

class TotalVariation(AbstractDeriv):
    @staticmethod
    def initial_guess(data: np.ndarray, dimensionality: tuple):
        grad = np.array([np.gradient(data, axis=dim_idx) for dim_idx, dim in enumerate(dimensionality)])
        # print(grad.shape)
        # w = np.array([np.zeros(dimensionality) for idx in np.arange(len(dimensionality)**2)], 
        #              dtype = np.ndarray).reshape([len(dimensionality), len(dimensionality),] + list(dimensionality))
        w = np.array([np.gradient(grad[int(idx/2.)], axis = idx % 2) 
                      for idx in np.arange(len(dimensionality)**2)]).reshape([len(dimensionality), 
                                                                              len(dimensionality),] + list(dimensionality))
        
        lap_mul = np.array([np.zeros(dimensionality) 
                            for idx in np.arange(len(dimensionality)**2)]).reshape([len(dimensionality), 
                                                                                    len(dimensionality),] + list(dimensionality))
        return grad, w, lap_mul

    @staticmethod
    def admm_step(data: np.ndarray, steps: list, initial_u: np.ndarray, initial_w: np.ndarray, 
                  initial_lap: np.ndarray, lbd: float, reg_strng: float, c_const: float) -> tuple:
        '''
        *data* has to be already Fourier-transformed
        All inputs initial_u, initial_w & initial_lap have to be transformed by DFT.
        '''
        def soft_thresholding(arg: np.ndarray, lbd: float) -> np.ndarray:
            norm = np.linalg.norm(arg)
            return max(norm - lbd, 0) * arg / norm
        
        def diff_factor(N: int, d: float = 1.) -> np.ndarray:
            #freqs = np.fft.fftfreq(N, d = d)
            freqs = np.arange(N)
            return np.exp(-2*np.pi*freqs/N) - 1
        
        initial_u = np.fft.fftn(initial_u, s = initial_u.shape[1:])
        # print(initial_u[0].shape)
        # print(np.max(np.abs(initial_u[0])), np.min(np.abs(initial_u[0])))
        Heatmap(np.abs(initial_u[0]), title = 'FFT')
        initial_w_fft = np.fft.fftn(initial_w, s = initial_w.shape[2:])
        initial_lap_fft = np.fft.fftn(initial_lap, s = initial_lap.shape[2:])
        
        diff_factors = np.array([np.moveaxis(np.broadcast_to(diff_factor(dim_size, d = steps[comp_idx]),
                                                             shape = initial_u[0].shape),
                                             source = -1, destination = comp_idx)
                                 for comp_idx, dim_size in enumerate(initial_u[0].shape)])
        
        lbd_inv = lbd**(-1)
        
        u_freq = np.copy(initial_u)
        
        for grad_idx in range(initial_u.shape[0]):
            u_nonzero_freq = np.zeros_like(u_freq[grad_idx])
            
            section_len = u_freq[grad_idx].shape[grad_idx]
            putting_shape = np.ones(u_freq[grad_idx].ndim, dtype = np.int8)
            putting_shape[grad_idx] = section_len-1
            
            putting_args = {'indices' : np.arange(1, u_freq[grad_idx].shape[grad_idx]).reshape(putting_shape),
                            'axis' : grad_idx}
            # print(f'putting_args {putting_args}')
            taking_args = {'indices' : range(1, u_freq[grad_idx].shape[grad_idx]),
                           'axis' : grad_idx}
            # print(f'taking_args {taking_args}')
            
            def take(arr: np.ndarray, taking_args: dict = taking_args):
                return np.take(arr, **taking_args)

            denum_part = lbd_inv * np.sum([np.abs(take(factor))**2 for factor in diff_factors])            
            partial_sum = [take(diff_factors[arg_idx]) * (take(initial_w_fft[grad_idx, arg_idx]) - 
                                                          take(initial_lap_fft[grad_idx, arg_idx])) 
                           for arg_idx in range(u_freq.shape[0])]
            print([(np.min(elem), np.max(elem)) for elem in partial_sum]) # * take(data) * take(data)
            print(np.min(take(data)), np.max(take(data)))
            print('3rd term:', np.min(reg_strng * take(diff_factors[grad_idx]) ), np.max(reg_strng * take(diff_factors[grad_idx]) ))
            
            grad_nonzero_upd = (lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data) /
                                (denum_part + reg_strng/np.abs(take(diff_factors[grad_idx]))))
            print('shit shape', (lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)).shape)
            print(np.max(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)),
                  np.min(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)))
            Heatmap(np.real(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)), title = 'partial')
            
            np.put_along_axis(u_nonzero_freq, values = grad_nonzero_upd, **putting_args)
            u_freq[grad_idx] = u_nonzero_freq
        Heatmap(np.real(u_freq[0]), title = 'inside the optimization')            
    
        initial_u = np.real(np.fft.ifftn(u_freq, u_freq.shape[1:]))
        for i in range(initial_w.shape[0]):
            for j in range(initial_w.shape[1]):
                initial_w[i, j] = soft_thresholding(np.gradient(initial_u[j], axis = i) + initial_lap[i, j], lbd)
                
        for i in range(initial_w.shape[0]):
            for j in range(initial_w.shape[1]):
                initial_lap[i, j] = c_const*(initial_lap[i, j] + np.gradient(initial_u[j], axis = i)
                                             - initial_w[i, j])
    
        time.sleep(15)
        return initial_u, initial_w, initial_lap

    def optimize_with_admm(self, data, lbd: float, reg_strng: float, c_const: float, nsteps: int = 1e5):
        u, w, lap_mul = self.initial_guess(data=data, dimensionality=data.shape)
        data_fft = np.fft.fftn(data)
        print(f'For some reason has to be abysmal: {np.min(np.real(data_fft)), np.max(np.real(data_fft))}')
        for epoch in range(int(nsteps)):
            print(epoch)
            if epoch % 100 == 0:
                Heatmap(u[1], title=str(epoch))
                plt.plot(u[1, :, int(u.shape[2]/2.)])
                plt.show()
                plt.close()
            u, w, lap_mul = self.admm_step(data = data_fft, steps = np.ones(data.ndim), initial_u = u, initial_w = w, 
                                           initial_lap=lap_mul, lbd = lbd, reg_strng = reg_strng, 
                                           c_const = c_const)
        return u
        
    # def differentiate(self, data: np.ndarray, max_order: Union[int, list],
    #                   mixed: bool = False, axis=None, *grids) -> list:
    #     if isinstance(max_order, int):
    #         max_order = [max_order,] * data.ndim
    #     else:
    #         max_order = np.full_like(max_order, np.max(max_order))
        
    #     if len(grids) == 1 and data.ndim == 2:
