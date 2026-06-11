#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

from abc import ABC
from typing import Callable, Union

import numpy as np
from functools import reduce
import copy
import torch
# device = torch.device('cpu')

import matplotlib.pyplot as plt

from epde.solver.data import Domain
from epde.solver.models import Fourier_embedding, mat_model
from epde.preprocessing.smoothers import NN
from numpy.lib.stride_tricks import sliding_window_view

from epde import _loop_stats


# Default behaviour for GramSetup's sliding-window CV: ``True`` treats
# each axis as periodic so windows near the boundary wrap around (the
# last few windows span data[end-k:end] + data[0:window_size-k]).
# ``False`` is the legacy linear semantics where the number of windows
# along an axis is ``N - window_size + 1`` and no wrap occurs.
_DEFAULT_CIRCULAR_CV = True


def _windowed_take(arr: np.ndarray, dim: int, window_size: int,
                    num_horizons: int, step_size: int,
                    circular: bool) -> np.ndarray:
    """Return ``sliding_window_view`` along ``dim``, optionally with
    circular padding so windows near the boundary wrap to the start.

    Caller computes ``num_horizons`` (the number of valid start positions
    along ``dim``) and ``step_size`` (subsampling stride). Under circular
    mode, ``num_horizons == arr.shape[dim]`` and the input is padded by
    ``window_size - 1`` samples copied from the start at the end via
    ``np.pad(..., mode='wrap')``. Under linear mode, ``num_horizons ==
    arr.shape[dim] - window_size + 1`` and no padding is applied.
    """
    if circular:
        pad = [(0, 0)] * arr.ndim
        pad[dim] = (0, window_size - 1)
        arr = np.pad(arr, pad, mode='wrap')
    windows = sliding_window_view(arr, window_shape=window_size, axis=dim)
    return windows.take(indices=range(0, num_horizons, step_size), axis=dim)


def retry_until_unique(*, predicate, mutate, max_iter: int, stats_name: str):
    """Bounded retry loop: keep mutating a candidate until ``predicate`` holds.

    Centralizes the (a) attempt-counter (b) cap-bound (c) ``_loop_stats``
    bookkeeping that the term-replacement loops share. Each call site
    owns the candidate object and decides the cap-hit policy
    (warn-accept, return False, raise, silently continue, etc.) based on
    the returned ``success`` flag.

    Args:
        predicate: zero-arg callable returning ``True`` when the
            candidate is acceptable. Called once per attempt before the
            mutation; if it returns ``True`` on the first call, no
            mutation is performed and ``attempts == 1``.
        mutate: zero-arg callable invoked between attempts to randomize
            the candidate (e.g. ``term.randomize()``). Not called after
            the final attempt.
        max_iter: maximum number of predicate checks. The site is
            expected to use ``100`` to match the cap normalized across
            ``Equation.__init__``, ``Equation.add_random_term``, and
            ``simplify_equation.replace_term``.
        stats_name: key passed to ``_loop_stats.record``; see
            ``EPDE_LOOP_STATS=1`` instrumentation.

    Returns:
        ``(success, attempts)`` -- ``success`` is ``True`` iff the
        predicate held within the cap; ``attempts`` is the number of
        predicate evaluations performed (1..max_iter).

    Related canonical retry sites that intentionally do NOT use this
    helper because their cap-hit policy is too entangled:
        - ``OffspringUpdater.unique_offspring`` (nested cap, sector
          skip) -- moeadd_specific.py.
        - ``InitialParetoLevelSorting.unique_candidate`` (raises
          ``RuntimeError`` per [[project_rps_bidirectional]]) -- ditto.
        - ``TermMutation.unique_term`` (post-loop revert) -- mutations.py.
        - ``EquationCrossover.duplicate_offspring`` (post-assembly
          single gate, not a loop) -- variation.py.
    """
    attempts = 0
    success = False
    for _ in range(max_iter):
        attempts += 1
        if predicate():
            success = True
            break
        mutate()
    _loop_stats.record(stats_name, attempts, max_iter)
    return success, attempts



class BasicDeriv(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Trying to create abstract differentiation method')
    
    def take_derivative(self, u: torch.Tensor, args: torch.Tensor, axes: list):
        raise NotImplementedError('Trying to differentiate with abstract differentiation method')


class AutogradDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: Union[torch.nn.Sequential, torch.Tensor], args: torch.Tensor, 
                        axes: list = [], component: int = 0):
        if not args.requires_grad:
            args.requires_grad = True
        if axes == [None,]:
            return u(args)[..., component].reshape(-1, 1)
        if isinstance(u, NN) or isinstance(u, torch.nn.Sequential):
            comp_sum = u(args)[..., component].sum(dim = 0)
        elif isinstance(u, torch.Tensor):
            raise TypeError('Autograd shall have torch.nn.Sequential as its inputs.')
        else:
            print(f'u.shape, {u.shape}')
            comp_sum = u.sum(dim = 0)
        for axis in axes:
            output_vals = torch.autograd.grad(outputs = comp_sum, inputs = args, create_graph=True)[0]
            comp_sum = output_vals[:, axis].sum()
        output_vals = output_vals[:, axes[-1]].reshape(-1, 1)
        return output_vals

class FDDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: np.ndarray, args: np.ndarray, 
                        axes: list = [], component: int = 0):
        
        if not isinstance(args, torch.Tensor):
            args = args.detach().cpu().numpy()

        output_vals = u[..., component].reshape(args.shape)
        if axes == [None,]:
            return output_vals
        for axis in axes:
            output_vals = np.gradient(output_vals, args.reshape(-1)[1] - args.reshape(-1)[0], axis = axis, edge_order=2)  
        return output_vals

def create_solution_net(equations_num: int, domain_dim: int, use_fourier = True, #  mode: str, domain: Domain 
                        fourier_params: dict = None, device = 'cpu'):
    '''
    fft_params have to be passed as dict with entries like: {'L' : [4,], 'M' : [3,]}
    '''
    L_default, M_default = 4, 10
    if use_fourier:
        if fourier_params is None:
            if domain_dim == 1:
                fourier_params = {'L' : [L_default],
                              'M' : [M_default]}
            else:
                fourier_params = {'L' : [L_default] + [None,] * (domain_dim - 1), 
                              'M' : [M_default] + [None,] * (domain_dim - 1)}
        fourier_params['device'] = device
        four_emb = Fourier_embedding(**fourier_params)
        if device == 'cuda':
            four_emb = four_emb.cuda()
        net_default = torch.nn.ModuleList([four_emb,])
    else:
        net_default = torch.nn.ModuleList([])
    linear_inputs = net_default[0].out_features if use_fourier else domain_dim
    
    if domain_dim == 1:            
        hidden_neurons = 128 # 64 #
    else:
        hidden_neurons = 112 # 54 #

    operators = net_default + torch.nn.ModuleList([torch.nn.Linear(linear_inputs, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, hidden_neurons, device=device),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden_neurons, equations_num, device=device)])
    return torch.nn.Sequential(*operators)

def exp_form(a, sign_num: int = 4):
    if np.isclose(a, 0):
        return 0.0, 0
    exp = np.floor(np.log10(np.abs(a)))
    return np.around(a / 10**exp, sign_num), int(exp)


def rts(value, sign_num: int = 5):
    """
    Round to a ``sign_num`` of significant digits.
    """
    if value == 0:
        return 0
    magn_top = np.log10(value)
    idx = -(np.sign(magn_top)*np.ceil(np.abs(magn_top)) - sign_num)
    if idx - sign_num > 1:
        idx -= 1
    return np.around(value, int(idx))


def train_ann(args: list, data: np.ndarray, epochs_max: int = 500, batch_frac = 0.5, 
              dim = None, model = None, device = 'cpu'):
    if dim is None:
        dim = 1 if np.any([s == 1 for s in data.shape]) and data.ndim == 2 else data.ndim
    # assert len(args) == dim, 'Dimensionality of data does not match with passed grids.'
    data_size = data.size
    if model is None:
        model = torch.nn.Sequential(
                                    torch.nn.Linear(dim, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 256, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(256, 64, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(64, 1024, device=device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(1024, 1, device=device)
                                    )
    
    model.to(device)
    data_grid = np.stack([arg.reshape(-1) for arg in args])
    grid_tensor = torch.from_numpy(data_grid).float().T.to(device)
    # grid_tensor.to(device)
    data = torch.from_numpy(data.reshape(-1, 1)).float().to(device)
    # print(data.size)
    # data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = int(data_size * batch_frac)

    t = 0

    print('grid_flattened.shape', grid_tensor.shape, 'field.shape', data.shape)

    loss_mean = 1000
    min_loss = np.inf
    losses = []
    while loss_mean > 2e-3 and t < epochs_max:

        permutation = torch.randperm(grid_tensor.size()[0])

        loss_list = []

        for i in range(0, grid_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid_tensor[indices], data[indices]
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean = np.mean(loss_list)
        if loss_mean < min_loss:
            best_model = model
            min_loss = loss_mean
        losses.append(loss_mean)
        # if global_var.verbose.show_ann_loss:
        #     print('Surface training t={}, loss={}'.format(t, loss_mean))
        t += 1
    print_loss = True
    if print_loss:
        fig = plt.figure()
        plt.plot(losses)
        plt.grid()
        plt.show()
        plt.close(fig)
    return best_model

def use_ann_to_predict(model, recalc_grids: list):
    data_grid = np.stack([grid.reshape(-1) for grid in recalc_grids])
    recalc_grid_tensor = torch.from_numpy(data_grid).float().T
    recalc_grid_tensor = recalc_grid_tensor #.to(device)

    return model(recalc_grid_tensor).detach().numpy().reshape(recalc_grids[0].shape)

def flatten(obj):
    '''
    Method to flatten list, passed as ``obj`` - the function parameter.
    '''
    assert type(obj) == list

    for idx, elem in enumerate(obj):
        if not isinstance(elem, (list, tuple)):
            obj[idx] = [elem,]
    return reduce(lambda x, y: x+y, obj)

def factor_params_to_str(factor, set_default_power=False, power_idx=0):
    """Canonical (label, params) tuple for a single Factor.

    This is the **single source of truth** for the cache-key /
    structural-identity tuple format used by ``Factor.cache_label``,
    ``Term.cache_label`` (via per-factor recursion), and any tensor
    cache lookup keyed on a factor. Anything that needs to identify a
    factor by ``(label, params)`` MUST call this helper rather than
    rebuilding the tuple inline -- see [[feedback_label_format_coupling]].

    Quantization for structural dedup is a **separate** concern, handled
    by ``Factor.structural_label`` via ``Factor._quantized_params``;
    that path is keyed on ``(label, quantized_params)`` and bucketizes
    continuous-tolerance params (e.g. trig ``freq``).
    """
    param_label = np.copy(factor.params)
    if set_default_power:
        param_label[power_idx] = 1.
    return (factor.label, tuple(param_label))


def detect_similar_terms(base_equation_1, base_equation_2):
    """Three-way split of each equation's terms by **exact** structural identity.

    Returns ``([same1, similar1, different1], [same2, similar2, different2])``
    where each inner list contains ``Term`` objects from the corresponding
    parent equation, classified by ``Term.factors_labels`` membership:

    - **same**: term's ``factors_labels`` appears in BOTH equations
      (set intersection).
    - **similar**: term's ``factors_labels`` appears only in THIS
      equation (set difference).
    - **different**: **always empty** under the current set-based
      partition -- every term's ``factors_labels`` is, by construction,
      a member of its own equation's ``terms_labels``, so the
      ``else`` branch is unreachable. Preserved as the third list
      element to keep the tuple shape stable for callers that destructure
      positionally.

    Used by ``epde.operators.singleobjective.variation.EquationCrossover``;
    the multi-objective EquationCrossover uses its own hybrid
    random-partition logic instead (see [[project_rps_bidirectional]]).
    Identity bucketing is delegated to ``Term.factors_labels`` -- which
    routes through ``Factor.structural_label`` and so honors the
    continuous-tolerance quantization defined per family.
    """
    all_first_equation_terms = base_equation_1.terms_labels
    all_second_equation_terms = base_equation_2.terms_labels

    same_terms_from_eq1 = []
    same_terms_from_eq2 = []
    similar_terms_from_eq1 = []
    similar_terms_from_eq2 = []
    different_terms_from_eq1 = []
    different_terms_from_eq2 = []

    common_terms = all_first_equation_terms.intersection(all_second_equation_terms)

    for term in base_equation_1.structure:
        if term.factors_labels in common_terms:
            same_terms_from_eq1.append(term)
        elif term.factors_labels in (all_first_equation_terms - all_second_equation_terms):
            similar_terms_from_eq1.append(term)
        else:
            different_terms_from_eq1.append(term)

    for term in base_equation_2.structure:
        if term.factors_labels in common_terms:
            same_terms_from_eq2.append(term)
        elif term.factors_labels in (all_second_equation_terms - all_first_equation_terms):
            similar_terms_from_eq2.append(term)
        else:
            different_terms_from_eq2.append(term)

    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]

def filter_powers(gene):
    gene_filtered = []

    for token_idx in range(len(gene)):
        total_power = sum([factor.param(name = 'power') for factor in gene
                           if gene[token_idx].partial_equlaity(factor)])#gene.count(gene[token_idx])
        # ``copy_for_power_update`` is a Factor-specific cheap clone --
        # only ``_params`` gets its own storage (set_param writes to it
        # in place); the rest of the factor's slots are aliased. Was
        # ~200μs per ``copy.deepcopy(factor)`` and ran 237k times per
        # lv_new rep -- the largest residual deepcopy after the
        # mutation/crossover round.
        powered_token = gene[token_idx].copy_for_power_update()

        power_idx = np.inf
        for param_idx, param_info in powered_token.params_description.items():
            if param_info['name'] == 'power':
                max_power = param_info['bounds'][1]
                power_idx = param_idx
                break
        powered_token.set_param(
            total_power if total_power < max_power else max_power,
            idx=power_idx,
        )
        if powered_token not in gene_filtered:
            gene_filtered.append(powered_token)
    return gene_filtered


def define_derivatives(var_name='u', dimensionality=1, max_order=2):
    """
    Method for generating derivative keys

    Args:
        var_name (`str`): name of input data dependent variable
        dimensionality (`int`): dimensionallity of data
        max_order (`int`|`list`): max order of delivative
    
    Returns:
        deriv_names (`list` with `str` values): keys for epde
        var_deriv_orders (`list` with `int` values): keys for enter to solver
    """
    deriv_names = []
    var_deriv_orders = []
    if isinstance(max_order, int):
        max_order = [max_order for dim in range(dimensionality)]
    for var_idx in range(dimensionality):
        for order in range(max_order[var_idx]):
            var_deriv_orders.append([var_idx,] * (order+1))
            if order == 0:
                deriv_names.append('d' + var_name + '/dx' + str(var_idx))
            else:
                deriv_names.append(
                    'd^'+str(order+1) + var_name + '/dx'+str(var_idx)+'^'+str(order+1))
    print('Deriv orders after definition', var_deriv_orders)
    return deriv_names, var_deriv_orders


def population_sort(input_population):
    individ_fitvals = [
        individual.fitness_value if individual.fitness_calculated else 0 for individual in input_population]
    pop_sorted = [x for x, _ in sorted(
        zip(input_population, individ_fitvals), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))


def normalize_ts(Input):
    matrix = np.copy(Input)
    if np.ndim(matrix) == 0:
        raise ValueError(
            
            'Incorrect input to the normalizaton: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return matrix
    else:
        for i in np.arange(matrix.shape[0]):
            std = np.std(matrix[i])
            if std != 0:
                matrix[i] = (matrix[i] - np.mean(matrix[i])) / std
            else:
                matrix[i] = 1
        return matrix

def minmax_normalize(matrix):
    """
    Apply min-max normalization to a matrix.
    For 1D arrays: returns as-is
    For 2D+ arrays: normalizes each row to [0, 1] range
    """
    matrix = np.copy(matrix)

    if np.ndim(matrix) == 0:
        raise ValueError('Incorrect input to the normalization: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return 2 * (matrix - matrix.min()) / (matrix.max() - matrix.min()) - 1
    else:
        for i in np.arange(matrix.shape[0]):
            if matrix[i].max() != matrix[i].min():
                matrix[i] = 2 * (matrix[i] - matrix[i].min()) / (matrix[i].max() - matrix[i].min()) - 1
            else:
                matrix[i] = np.zeros_like(matrix[i])
        return matrix


def _cholesky_solve_batched(A, b):
    """Solve ``A @ x = b`` batched over the leading axis using Cholesky.

    ``A`` is assumed symmetric positive-definite (shape ``(batch, n, n)``);
    ``b`` is the RHS ``(batch, n, 1)``. Returns ``(x, L)`` where ``x`` is
    the solution and ``L`` is the lower-triangular factor (so the caller
    can reuse it for iterative refinement). If Cholesky fails on any batch
    entry, returns ``(None, None)`` to signal "use the lstsq fallback".

    numpy doesn't ship a batched triangular solver, so the two triangular
    solves go through ``np.linalg.solve`` -- still SPD-stable and ~1.5x
    faster than feeding the full ``A`` to ``np.linalg.solve``.
    """
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None, None
    try:
        z = np.linalg.solve(L, b)
        x = np.linalg.solve(L.transpose(0, 2, 1), z)
    except np.linalg.LinAlgError:
        return None, L
    return x, L


def _per_batch_lstsq(A, b):
    """Per-batch SVD-based least-squares solve. Used as the safety net
    when Cholesky reports the equilibrated batch is non-SPD. Returns
    weights of shape ``(batch, n, 1)`` matching the input RHS layout so
    the caller can compose with subsequent matrix products without
    reshaping.
    """
    batch_size = A.shape[0]
    n = A.shape[1]
    out = np.empty((batch_size, n, 1))
    for i in range(batch_size):
        sol, *_ = np.linalg.lstsq(A[i], b[i, :, 0], rcond=None)
        out[i, :, 0] = sol
    return out


class GramSetup:
    """Precomputed batched normal-equation matrices for fast active-mask
    solves. Splits :func:`calculate_weights` into a setup phase (compute
    ``X^T diag(w) X`` and ``X^T diag(w) y`` per window-batch per dimension,
    using the FULL augmented feature matrix) and a solve phase (slice each
    full Gram matrix by an active-feature mask and solve). The setup is
    mask-independent; only the solve depends on which columns are active.

    Used by :class:`PhysicsInformedLasso.fit`, whose outer RFE loop calls
    ``calculate_weights`` per shrinking column subset. With this split the
    expensive ``X^T diag(w) X`` matmul runs ONCE per fit and each outer
    iter only pays the cost of an (active × active) solve. The math is
    exact: a sub-block of a Gram matrix equals the Gram of the
    corresponding sub-columns.
    """

    def __init__(self, X, y, sample_weights, grid_shape,
                 circular_cv: bool = _DEFAULT_CIRCULAR_CV):
        n_samples = X.shape[0]
        # Always augment X with the intercept column so callers can toggle
        # ``fit_intercept`` via the active mask's last bit rather than
        # re-running setup.
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        n_features_aug = X_aug.shape[1]

        X_grid = X_aug.reshape(*grid_shape, n_features_aug)
        y_grid = y.reshape(*grid_shape)
        sample_weights_grid = sample_weights.reshape(*grid_shape)

        self.n_features_aug = n_features_aug
        self.grid_shape = grid_shape
        self._per_dim = []

        for dim in range(len(grid_shape)):
            window_size = grid_shape[dim] // 2
            # Circular: every position along the axis is a valid window
            # start (the input is virtually periodic); linear: only the
            # first ``window_size + 1`` positions yield a full window.
            num_horizons = grid_shape[dim] if circular_cv else window_size + 1
            step_size = max(1, num_horizons // 30)

            X_windows = _windowed_take(X_grid, dim, window_size,
                                       num_horizons, step_size, circular_cv)
            y_windows = _windowed_take(y_grid, dim, window_size,
                                       num_horizons, step_size, circular_cv)
            w_windows = _windowed_take(sample_weights_grid, dim, window_size,
                                       num_horizons, step_size, circular_cv)

            X_windows = np.moveaxis(X_windows, dim, 0)
            y_windows = np.moveaxis(y_windows, dim, 0)
            w_windows = np.moveaxis(w_windows, dim, 0)
            X_windows = np.moveaxis(X_windows, -2, -1)

            batch_size = X_windows.shape[0]
            X_batch = X_windows.reshape(batch_size, -1, n_features_aug)
            y_batch = y_windows.reshape(batch_size, -1)
            weights_batch = w_windows.reshape(batch_size, -1, 1)

            XTW = X_batch.transpose(0, 2, 1) * weights_batch.transpose(0, 2, 1)
            XTWX_full = XTW @ X_batch
            XTWy_full = XTW @ y_batch[..., None]

            # Per-batch column scales for equilibration in :meth:`solve`.
            # ``diag`` is the per-feature L2 norm squared (weighted) of the
            # underlying X columns; ``sqrt`` brings it back to a column-
            # norm scale. The ``1e-30`` floor is a degenerate-column guard
            # (well below any meaningful data scale) so ``1/scale`` stays
            # finite for near-zero columns.
            diag = np.diagonal(XTWX_full, axis1=1, axis2=2)
            scales = np.sqrt(np.maximum(np.abs(diag), 1e-30))

            self._per_dim.append((XTWX_full, XTWy_full, scales))

    def solve(self, active_mask=None, ridge_rel=None, ridge_floor=None):
        """Solve the normal equations for the active-feature subset across
        every window-batch in every spatial dimension. ``active_mask`` is a
        length-``n_features_aug`` boolean array; pass ``None`` for the full
        set (equivalent to the legacy ``fit_intercept=True`` path). Returns
        weights of shape ``(total_windows_across_dims, active_count)``.

        Stability strategy (preserves the Gram-sub-block precompute trick):

        1. **Column equilibration**: rescale columns by
           ``1/sqrt(diag(XTWX))`` so the equilibrated Gram has unit
           diagonals and a much smaller effective condition number than
           the raw ``XTWX`` (which carries the squared condition number
           of the underlying ``sqrt(W) X``).
        2. **Cholesky on the equilibrated SPD batch** (with batched LU
           fallback if scipy's batched triangular solve isn't available
           on this numpy). Cholesky has tighter backward error than LU
           and is ~2x faster on SPD inputs.
        3. **One step of iterative refinement** on the original (un-
           equilibrated) system, recovering 6-8 decimal digits that
           normal-equation conditioning costs.
        4. **Per-batch lstsq safety net** for any window-batch where
           Cholesky fails (non-SPD after equilibration -- rare).

        ``ridge_rel`` / ``ridge_floor`` are kept as no-op kwargs for
        backward compatibility with callers from the previous adaptive-
        ridge era; the equilibrated solve does not need a per-feature
        ridge, only a tiny flat ``1e-10`` on the unit-diagonal matrix.
        """
        if active_mask is None:
            active_mask = np.ones(self.n_features_aug, dtype=bool)
        active_size = int(active_mask.sum())

        all_weights = []
        for XTWX_full, XTWy_full, scales_full in self._per_dim:
            # Two-step boolean slice. Boolean indexing copies, so the
            # result is a fresh array we can modify in place without
            # corrupting the cached full Gram.
            XTWX_a = XTWX_full[:, active_mask, :][:, :, active_mask]
            XTWy_a = XTWy_full[:, active_mask, :]
            s_a = scales_full[:, active_mask]                     # (batch, k)
            inv_s = 1.0 / s_a                                      # (batch, k)

            # Equilibrate: A = D^-1 XTWX D^-1, b = D^-1 XTWy. After this
            # the diagonal of A is 1 by construction; the off-diagonals
            # are the correlation coefficients between the underlying
            # columns of sqrt(W) X.
            A = XTWX_a * inv_s[:, :, None] * inv_s[:, None, :]
            b = XTWy_a * inv_s[:, :, None]

            # Tiny flat ridge on the equilibrated diagonal (now ~1 by
            # construction) to keep Cholesky well-defined when columns
            # are exactly collinear.
            idx = np.arange(active_size)
            A[:, idx, idx] += 1e-10

            batch_size = A.shape[0]
            w_norm, L = _cholesky_solve_batched(A, b)
            if w_norm is None:
                # Cholesky failed somewhere in the batch; per-entry
                # lstsq safety net on the equilibrated system.
                w_norm = _per_batch_lstsq(A, b)

            # Iterative refinement on the ORIGINAL system to claw back
            # digits lost to normal-equation condition squaring.
            # w0 = D^-1 w_norm is the candidate solution in original
            # coordinates; the residual r = XTWy - XTWX @ w0 measures
            # how much it misses the original equation; the correction
            # dw_norm solves the same equilibrated system on D^-1 r and
            # is unscaled back to dw.
            w0 = w_norm * inv_s[:, :, None]
            r = XTWy_a - XTWX_a @ w0
            r_norm = r * inv_s[:, :, None]
            if L is not None:
                try:
                    z = np.linalg.solve(L, r_norm)
                    dw_norm = np.linalg.solve(L.transpose(0, 2, 1), z)
                except np.linalg.LinAlgError:
                    dw_norm = _per_batch_lstsq(A, r_norm)
            else:
                dw_norm = _per_batch_lstsq(A, r_norm)
            w = w0 + dw_norm * inv_s[:, :, None]

            all_weights.append(w.squeeze(-1))
        return np.vstack(all_weights)

    @classmethod
    def precompute_super(cls, Z, sample_weights, grid_shape,
                          circular_cv: bool = _DEFAULT_CIRCULAR_CV):
        """Build a per-dim super-Gram over ``Z_aug = column_stack(Z, ones)``
        once. Returns an opaque dict consumed by :meth:`from_full` to
        derive per-target GramSetup views via pure slicing.

        Used by EqRightPartSelector's term-sweep: instead of rebuilding
        the windowed XTWX matrix for each candidate target column (which
        does the same reshape/window/matmul on the SAME underlying Z),
        precompute once over the full Z and slice out target-specific
        sub-blocks. The math is exact -- (Z[:, ~t])^T W Z[:, ~t] is the
        sub-block of (Z_aug)^T W Z_aug at rows/cols ``~t U intercept``.

        ``Z`` is shape (n_samples, n_terms); ``sample_weights`` is the
        flat per-sample weight vector; ``grid_shape`` is the same shape
        ``GramSetup.__init__`` consumes. ``circular_cv`` mirrors the
        ``__init__`` flag -- callers that flow through both paths
        (PhysicsInformedLasso single-call + EqRPS super-Gram sweep) must
        keep these values aligned for the per-target views to remain
        sub-blocks of the super-Gram (the math requires identical window
        sets across the two passes).
        """
        n_samples, n_terms = Z.shape
        Z_aug = np.hstack([Z, np.ones((n_samples, 1))])
        n_features_aug = n_terms + 1

        Z_grid = Z_aug.reshape(*grid_shape, n_features_aug)
        sw_grid = sample_weights.reshape(*grid_shape)
        per_dim_super = []

        for dim in range(len(grid_shape)):
            window_size = grid_shape[dim] // 2
            num_horizons = grid_shape[dim] if circular_cv else window_size + 1
            step_size = max(1, num_horizons // 30)

            Z_windows = _windowed_take(Z_grid, dim, window_size,
                                       num_horizons, step_size, circular_cv)
            w_windows = _windowed_take(sw_grid, dim, window_size,
                                       num_horizons, step_size, circular_cv)

            Z_windows = np.moveaxis(Z_windows, dim, 0)
            w_windows = np.moveaxis(w_windows, dim, 0)
            Z_windows = np.moveaxis(Z_windows, -2, -1)

            batch_size = Z_windows.shape[0]
            Z_batch = Z_windows.reshape(batch_size, -1, n_features_aug)
            weights_batch = w_windows.reshape(batch_size, -1, 1)

            ZTW = Z_batch.transpose(0, 2, 1) * weights_batch.transpose(0, 2, 1)
            XTWX_super = ZTW @ Z_batch

            diag = np.diagonal(XTWX_super, axis1=1, axis2=2)
            scales_super = np.sqrt(np.maximum(np.abs(diag), 1e-30))

            per_dim_super.append((XTWX_super, scales_super))

        return {
            'per_dim_super': per_dim_super,
            'n_features_aug': n_features_aug,
            'grid_shape': grid_shape,
            'n_terms': n_terms,
            # Cached so downstream VWSRSparsity can derive per-target
            # ``target`` / ``features`` by slicing instead of re-calling
            # objective.evaluate(normalize=True) -- which would force
            # another vstack + transpose of the same term evaluations
            # for every candidate target_idx in the sweep.
            'Z': Z,
        }

    @classmethod
    def from_full(cls, super_data, target_idx_in_terms):
        """Construct a per-target GramSetup view from precomputed
        super-Gram data via slicing.

        ``super_data`` is the dict returned by :meth:`precompute_super`.
        ``target_idx_in_terms`` is the column index within Z (the terms
        portion) that the caller wants as the regression target; the
        intercept column stays in the feature set automatically.

        The returned object has the same ``n_features_aug``, ``grid_shape``
        and ``_per_dim`` shape contract as a regular ``GramSetup`` so
        downstream code (``PhysicsInformedLasso.fit`` -> ``solve``) is
        unchanged.
        """
        per_dim_super = super_data['per_dim_super']
        n_features_aug_super = super_data['n_features_aug']
        grid_shape = super_data['grid_shape']

        if not (0 <= target_idx_in_terms < n_features_aug_super - 1):
            raise IndexError(
                f'target_idx_in_terms={target_idx_in_terms} out of range '
                f'[0, {n_features_aug_super - 1}) for super-Gram with '
                f'{n_features_aug_super - 1} terms (+1 intercept).'
            )

        active = np.ones(n_features_aug_super, dtype=bool)
        active[target_idx_in_terms] = False

        instance = cls.__new__(cls)
        instance.n_features_aug = int(active.sum())  # n_terms (incl. intercept)
        instance.grid_shape = grid_shape
        instance._per_dim = []
        for XTWX_super, scales_super in per_dim_super:
            XTWX_target = XTWX_super[:, active, :][:, :, active]
            # XTWy = (Z_aug[:, ~t])^T W Z[:, t] = column t of XTWX_super
            # at rows in ``active``.
            XTWy_target = XTWX_super[:, active,
                                     target_idx_in_terms:target_idx_in_terms + 1]
            scales_target = scales_super[:, active]
            instance._per_dim.append((XTWX_target, XTWy_target, scales_target))
        return instance


def calculate_weights(X, y, sample_weights, grid_shape, fit_intercept=True):
    """
    Vectorized calculation of weights across sliding windows.
    Dynamically handles whether the intercept should be fit.

    Single-shot wrapper over :class:`GramSetup`: builds the precomputed
    Gram once and immediately solves with the requested intercept policy.
    Callers that solve the same Gram against many active masks (e.g.
    :class:`PhysicsInformedLasso.fit`) should instantiate ``GramSetup``
    directly and call ``.solve(active_mask)`` per iteration to avoid
    re-running the expensive ``X^T diag(w) X`` matmul.
    """
    setup = GramSetup(X, y, sample_weights, grid_shape)
    active_mask = np.ones(setup.n_features_aug, dtype=bool)
    if not fit_intercept:
        # GramSetup always augments with the intercept column; drop it
        # from the active set to mimic the legacy ``fit_intercept=False``
        # branch (which never augmented in the first place).
        active_mask[-1] = False
    return setup.solve(active_mask)
