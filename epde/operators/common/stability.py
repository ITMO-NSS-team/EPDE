"""Stability-estimation subsystem for the EPDE structural search.

Two coefficient-stability estimators used by the sparsity / fitness /
right-part operators to score and prune candidate library terms:

* ``GramSetup`` -- axis-aligned sliding-window CV (the ``gram_mode='axis'``
  backup), with ``calculate_weights`` as a single-shot wrapper.
* ``VaryingCoefSetup`` -- the default ``gram_mode='vcoef'`` varying-coefficient
  stability, summed by ``vc_stability_total_lr``; basis resolution resolved
  by ``resolve_vc_modes_from_input`` / ``taylor_microscale``.

Relocated verbatim from ``epde.supplementary``. ``import epde.globals`` is
kept lazy (inside the bodies that use it) to avoid a circular import:
globals.py imports utilities from supplementary at module top level.
"""
import numpy as np
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


def taylor_microscale(field: np.ndarray, grid_shape, axis: int,
                       eps: float = 1e-30,
                       deriv_field: np.ndarray = None) -> float:
    """Return the Taylor microscale of ``field`` along ``axis``.

    ``lambda^2 = <u^2> / <(du/dx)^2>``, with unit grid spacing -- the
    result has units of grid points so it sets the locality scale
    directly. Constant or near-constant fields produce ``inf``.

    When ``deriv_field`` is supplied (the already-computed
    ``d(field)/d(axis)``), use it directly instead of ``np.gradient``;
    cache fast path that also gives a higher-quality estimate on noisy
    data when the EPDE pool stored ANN-smoothed derivatives.
    """
    f = np.asarray(field).reshape(grid_shape)
    if deriv_field is None:
        g = np.gradient(f, axis=axis)
    else:
        g = np.asarray(deriv_field).reshape(grid_shape)
    num = float(np.mean(f * f))
    den = float(np.mean(g * g))
    if den <= eps or not np.isfinite(den):
        return float('inf')
    return float(np.sqrt(num / den))


def _cached_primary_var_names() -> list:
    """List bare variable labels currently in ``global_var.tensor_cache``.

    Bare labels are entries with no ``/`` or ``^`` -- ``u``, ``v``, ``p``,
    etc. Derivative labels (``du/dx0``, ``d^2u/dx1^2``) are filtered out.
    Returns ``[]`` when the cache is uninitialised or empty.
    """
    import epde.globals as _gv
    tc = getattr(_gv, 'tensor_cache', None)
    if tc is None:
        return []
    try:
        numpy_dict = tc.memory_default.get('numpy', {})
    except Exception:
        return []
    out = []
    for k in numpy_dict:
        if not isinstance(k, tuple) or len(k) != 2:
            continue
        label = k[0]
        if (isinstance(label, str) and '/' not in label
                and '^' not in label and label not in out):
            out.append(label)
    return out


def resolve_vc_modes_from_input(grid_shape, main_var=None,
                                k_max: int = 6, k_min: int = 2):
    """One-shot resolve of per-axis varying-coefficient basis modes ``K_d``
    from the cached source variable's Taylor microscale, cached per
    ``(grid_shape, main_var)`` so every candidate equation in one run shares
    the SAME basis resolution.

    ``K_d = clip(ceil(n_d / lambda_d) + 1, k_min, k_max)``: roughly one
    cosine mode per coherence length along axis ``d``, ``+1`` for the
    constant term. The microscale is taken as the *minimum* over variables
    (fastest-varying wins) so the basis can resolve every variable's
    structure; constant / near-constant fields (``lambda_d -> inf``)
    collapse to ``k_min`` (constant + one mode).

    Returns a tuple ``(K_0, ..., K_{D-1})``; ``None`` on cache miss with no
    cached variable (caller falls back to the target-field path).
    """
    import epde.globals as _gv
    key = (tuple(int(n) for n in grid_shape), main_var, 'vc')
    cache = getattr(_gv, 'vc_modes_cache', None)
    if cache is None:
        return None
    if key in cache:
        return cache[key]

    if main_var is not None:
        var_names = [main_var]
    else:
        var_names = _cached_primary_var_names()
    if not var_names:
        return None

    tc = _gv.tensor_cache
    D = len(grid_shape)
    lam_min = [float('inf')] * D
    for var in var_names:
        try:
            field = tc.get((var, (1.0,)))
        except Exception:
            continue
        for d in range(D):
            lam = taylor_microscale(field, grid_shape, d)
            if lam < lam_min[d]:
                lam_min[d] = lam
    if all(not np.isfinite(l) for l in lam_min):
        return None

    modes = []
    for d in range(D):
        n_d = int(grid_shape[d])
        lam = lam_min[d]
        if not np.isfinite(lam) or lam <= 0:
            K_d = k_min
        else:
            K_d = int(np.ceil(n_d / lam)) + 1
        K_d = max(k_min, min(int(K_d), int(k_max)))
        modes.append(int(K_d))
    result = tuple(modes)
    cache[key] = result
    return result


class VaryingCoefSetup:
    """Varying-coefficient stability estimator -- the default
    ``gram_mode='vcoef'`` path, alternative to the axis-aligned
    ``GramSetup`` (the ``gram_mode='axis'`` backup).

    Instead of measuring each coefficient's dispersion over local
    sub-regions, model each term's coefficient as a smooth function of
    position ``beta_j(x) = gamma_{j,0} + sum_d sum_{k>=1} gamma_{j,d,k}
    B_k(x_d)`` (additive low-frequency cosine basis, so the column count is
    LINEAR in the spatial dimension). A term is homogeneous (true) iff its
    *non-constant* energy is small relative to its constant part; the
    per-term stability score is

        score_j = (Var(gamma_{j,0}) + NC_j) / gamma_{j,0}^2

    with ``NC_j = sum_{k>=1} max(gamma_{j,d,k}^2 - lam*Var(gamma_{j,d,k}), 0)``
    the noise-debiased non-constant energy: significance of the constant
    coefficient plus region-variation, over the squared constant part. The
    locality scale is the basis resolution ``K`` -- resolved once per
    ``(grid_shape, main_var)`` from the Taylor microscale and shared by every
    candidate -- so there is no locality hyperparameter to tune per dataset.

    Contract: exposes ``score(active_mask)`` returning a length-active vector
    aligned to the active feature columns, a drop-in for
    ``PhysicsInformedLasso.get_cv``'s return. The ``precompute_super`` /
    ``from_full`` pair mirrors ``GramSetup`` so the EqRPS term-sweep builds
    the expanded Gram ONCE and slices per candidate target.

    Noise handling: a frequency-scaled ridge ``rho*k^2`` on the gamma solve
    (smoothing-spline analogue -- high modes shrunk hard) plus a chi-square
    bias subtraction of the expected noise energy from ``NC_j``, both scaled
    by the homogeneous-fit residual variance, so a true term's *expected*
    score is ~0 rather than a positive noise floor.
    """

    is_vcoef = True

    def __init__(self, X, y, sample_weights, grid_shape, main_var: str = None,
                 modes=None, k_max=None, freq_coef=None, eps_rel: float = 1e-6,
                 fit_intercept: bool = True):
        grid_shape = tuple(int(n) for n in grid_shape)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n_samples = X.shape[0]
        # Mirror calculate_weights' intercept policy: only append the constant
        # (ones) column when the equation actually fits an intercept
        # (weights_final[-1] != 0). A PDE has no constant term, so its intercept
        # is zeroed; including it would fit gamma_0 ~ 0 and blow up the
        # 1/t^2 score, dominating the summed stability objective.
        X_aug = (np.hstack([X, np.ones((n_samples, 1))]) if fit_intercept
                 else X)
        modes = self._resolve_modes(modes, grid_shape, main_var, k_max)
        w = np.asarray(sample_weights, dtype=float).reshape(-1)
        Bvals, mode_k = self._basis_values(grid_shape, modes)
        G, Phiy, _, B = self._gram(X_aug, w, Bvals, mode_k, y=y)

        self.G = G
        self.Phiy = Phiy
        self.B = int(B)
        self.basis_mode_k = mode_k
        # Basis values (N, B) kept for beta(x) reconstruction in
        # ``beta_field_stats`` (diagnostic report-form comparison). Only the
        # direct-construction path stores it; ``from_full`` leaves it None.
        self._Bvals = Bvals
        self.n_features = X_aug.shape[1]
        self.grid_shape = grid_shape
        self.N_eff = float(np.sum(w))
        y_flat = np.asarray(y, dtype=float).reshape(-1)
        self.yWy = float(np.dot(y_flat, w * y_flat))
        self.freq_coef = self._cfg_freq(freq_coef)
        self.eps_rel = float(eps_rel)

    # ------------------------------------------------------------------ #
    # Config / basis helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cfg_freq(freq_coef):
        if freq_coef is not None:
            return float(freq_coef)
        import epde.globals as _gv
        return float(getattr(_gv, 'vc_freq_coef', 1.0))

    @classmethod
    def _resolve_modes(cls, modes, grid_shape, main_var, k_max):
        if modes is not None:
            return tuple(int(m) for m in modes)
        import epde.globals as _gv
        if k_max is None:
            k_max = int(getattr(_gv, 'vc_k_max', 6))
        resolved = resolve_vc_modes_from_input(
            grid_shape, main_var=main_var, k_max=k_max)
        if resolved is not None:
            return resolved
        # Fallback (no cached field): a modest fixed resolution per axis.
        return tuple(min(int(k_max), 3) for _ in grid_shape)

    @staticmethod
    def _basis_values(grid_shape, modes):
        """Return ``(Bvals (N, B), mode_k (B,))`` -- the additive cosine
        design columns over the flattened grid. Column 0 is the constant;
        the rest are zero-mean unit-mean-square DCT-II cosines
        ``sqrt(2) cos(pi k (i+0.5)/n_d)`` per axis, ``k=1..K_d-1``.
        """
        grid_shape = tuple(int(n) for n in grid_shape)
        D = len(grid_shape)
        N = int(np.prod(grid_shape))
        cols = [np.ones(N, dtype=float)]
        mode_k = [0.0]
        for d in range(D):
            n_d = grid_shape[d]
            Kd = int(modes[d])
            idx = np.arange(n_d, dtype=float)
            for k in range(1, Kd):
                m1d = np.sqrt(2.0) * np.cos(np.pi * k * (idx + 0.5) / n_d)
                shp = [1] * D
                shp[d] = n_d
                full = np.broadcast_to(m1d.reshape(shp), grid_shape)
                cols.append(np.ascontiguousarray(full).reshape(-1))
                mode_k.append(float(k))
        Bvals = np.stack(cols, axis=1)
        return Bvals, np.asarray(mode_k, dtype=float)

    @staticmethod
    def _gram(F_aug, w, Bvals, mode_k, y=None):
        """Build the expanded Gram ``G = Phi^T W Phi`` (and ``Phi^T W y`` if
        ``y`` given) where ``Phi[:, f*B + b] = F_aug[:, f] * Bvals[:, b]``
        (block-major by feature, so feature ``f`` owns columns
        ``[f*B : (f+1)*B]`` and its constant column is ``f*B``).
        """
        N, Fc = F_aug.shape
        B = Bvals.shape[1]
        Phi = (F_aug[:, :, None] * Bvals[:, None, :]).reshape(N, Fc * B)
        WPhi = w[:, None] * Phi
        G = Phi.T @ WPhi
        Phiy = None
        if y is not None:
            Phiy = Phi.T @ (w * np.asarray(y, dtype=float).reshape(-1))
        return G, Phiy, mode_k, B

    # ------------------------------------------------------------------ #
    # Super-Gram precompute / per-target slicing (EqRPS term sweep)
    # ------------------------------------------------------------------ #
    @classmethod
    def precompute_super(cls, Z, sample_weights, grid_shape, main_var=None,
                          modes=None, k_max=None, freq_coef=None,
                          eps_rel: float = 1e-6):
        grid_shape = tuple(int(n) for n in grid_shape)
        n_samples, n_terms = Z.shape
        Z_aug = np.hstack([Z, np.ones((n_samples, 1))])
        modes = cls._resolve_modes(modes, grid_shape, main_var, k_max)
        w = np.asarray(sample_weights, dtype=float).reshape(-1)
        Bvals, mode_k = cls._basis_values(grid_shape, modes)
        G_super, _, _, B = cls._gram(Z_aug, w, Bvals, mode_k, y=None)
        return {
            'mode': 'vcoef',
            'G_super': G_super,
            'B': int(B),
            'basis_mode_k': mode_k,
            'n_features_aug': n_terms + 1,
            'grid_shape': grid_shape,
            'n_terms': n_terms,
            'N_eff': float(np.sum(w)),
            'freq_coef': cls._cfg_freq(freq_coef),
            'eps_rel': float(eps_rel),
            'Z': Z,
        }

    @classmethod
    def from_full(cls, super_data, target_idx_in_terms):
        """Per-target view: features = all terms except ``target`` plus the
        intercept; ``Phi^T W y`` is the target's constant column of the
        super-Gram (since the target equals its own constant-expansion
        column). Mirrors ``GramSetup.from_full``.
        """
        G_super = super_data['G_super']
        B = int(super_data['B'])
        n_feat_super = int(super_data['n_features_aug'])  # n_terms + 1
        n_terms = int(super_data['n_terms'])
        if not (0 <= target_idx_in_terms < n_terms):
            raise IndexError(
                f'target_idx_in_terms={target_idx_in_terms} out of range '
                f'[0, {n_terms}) for vcoef super-Gram.')

        active_global = [i for i in range(n_feat_super)
                         if i != target_idx_in_terms]
        cols = np.concatenate(
            [np.arange(i * B, (i + 1) * B) for i in active_global])
        target_const = target_idx_in_terms * B

        inst = cls.__new__(cls)
        inst.is_vcoef = True
        inst.G = G_super[np.ix_(cols, cols)]
        inst.Phiy = G_super[cols, target_const].copy()
        inst.yWy = float(G_super[target_const, target_const])
        inst.B = B
        inst.basis_mode_k = super_data['basis_mode_k']
        inst.n_features = len(active_global)
        inst.grid_shape = super_data['grid_shape']
        inst.N_eff = float(super_data['N_eff'])
        inst.freq_coef = float(super_data['freq_coef'])
        inst.eps_rel = float(super_data['eps_rel'])
        inst._Bvals = None     # super path: beta(x) reconstruction unavailable
        return inst

    # ------------------------------------------------------------------ #
    # Per-term stability score
    # ------------------------------------------------------------------ #
    def _solve_gammas(self, active_mask=None):
        """Frisch-Waugh block solve for the per-active-feature gammas: the
        constant part ``gamma_0`` is the features-only weighted OLS coefficient
        (decoupled from the basis modes, so collinear systems recover it
        correctly), and the modes are fit to the constant-fit residual with a
        noise-adaptive frequency ridge. Returns a dict with ``gamma`` (basis
        coefficients, block-major by feature), ``var`` (Cov diagonal), and the
        layout (``nf``, ``B``, ``mk``, ``mean_power``). ``None`` if no active
        feature.
        """
        n = self.n_features
        if active_mask is None:
            active_mask = np.ones(n, dtype=bool)
        active_feats = np.where(active_mask)[0]
        nf = int(active_feats.size)
        if nf == 0:
            return None

        B = self.B
        mk = self.basis_mode_k
        cols = (active_feats[:, None] * B
                + np.arange(B)[None, :]).reshape(-1)
        G_a = self.G[np.ix_(cols, cols)].astype(float).copy()
        Phiy_a = self.Phiy[cols].astype(float).copy()
        col_mode = np.tile(mk, nf)
        const_local = np.arange(nf) * B
        is_mode = np.ones(nf * B, dtype=bool)
        is_mode[const_local] = False
        mode_local = np.where(is_mode)[0]

        # --- Constant block: features-only weighted OLS (the standard SINDy
        # coefficient gamma_0). Decoupling it from the basis modes (Frisch-
        # Waugh: fit constants, then fit modes to the constant-fit residual) is
        # what keeps the recovered coefficient correct on collinear systems. The
        # joint expanded solve let the near-collinear modulated columns steal
        # signal from the constant (e.g. lorenz g0=2.3 instead of 10, despite a
        # well-conditioned const block) -> spurious non-constant energy and a
        # wrong magnitude. Here g0 = exactly the features-only OLS, untouched by
        # the modes.
        Ac = G_a[np.ix_(const_local, const_local)]
        bc = Phiy_a[const_local]
        dC = np.sqrt(np.maximum(np.diag(Ac), 1e-30))
        AcN = Ac / np.outer(dC, dC)
        AcN[np.diag_indices_from(AcN)] += 1e-10
        try:
            AcN_inv = np.linalg.inv(AcN)
        except np.linalg.LinAlgError:
            AcN_inv = np.linalg.pinv(AcN)
        g0_vec = (AcN_inv @ (bc / dC)) / dC

        rss = self.yWy - float(g0_vec @ bc)
        sigma2 = max(rss, 0.0) / max(self.N_eff - nf, 1.0)
        mean_power = self.yWy / max(self.N_eff, 1.0)
        noise_rel = min(max(sigma2 / (mean_power + 1e-30), 0.0), 1.0)

        gamma = np.zeros(nf * B)
        gamma[const_local] = g0_vec

        # --- Mode block: fit the constant-fit residual r = y - X gamma_0 onto
        # the basis-modulated columns (r is the OLS residual, so it is already
        # W-orthogonal to the constant columns -> gamma_0 stays untouched). The
        # noise-adaptive frequency ridge rho*k^2 still shrinks high modes; on
        # clean data the modes simply stay ~0 (a true constant coefficient).
        if mode_local.size:
            Amm = G_a[np.ix_(mode_local, mode_local)]
            Amc = G_a[np.ix_(mode_local, const_local)]
            b_m = Phiy_a[mode_local] - Amc @ g0_vec
            dM = np.sqrt(np.maximum(np.diag(Amm), 1e-30))
            AmmN = Amm / np.outer(dM, dM)
            kmax2 = max(float(np.max(mk)) ** 2, 1.0)
            cm = col_mode[mode_local]
            ridge_m = 1e-6 + self.freq_coef * noise_rel * (cm ** 2 / kmax2)
            AmmN[np.diag_indices_from(AmmN)] += ridge_m
            import epde.globals as _gv
            if getattr(_gv, 'vc_mode_decouple', False):
                # Block-diagonalise by feature: drop the cross-feature mode
                # collinearity blocks so each term's modes are fit only to what
                # ITS OWN modulated columns explain of the constant-fit
                # residual. A true constant-coef term then keeps modes ~0 (B
                # small) even when collinear grid-modulated cousins are present,
                # instead of borrowing their variation. Within-feature mode
                # correlations (same feature, different k) are preserved.
                feat_of_mode = mode_local // B
                AmmN = AmmN * (feat_of_mode[:, None] == feat_of_mode[None, :])
            try:
                AmmN_inv = np.linalg.inv(AmmN)
            except np.linalg.LinAlgError:
                AmmN_inv = np.linalg.pinv(AmmN)
            gamma[mode_local] = (AmmN_inv @ (b_m / dM)) / dM

        return {'gamma': gamma, 'active_feats': active_feats,
                'nf': nf, 'B': B, 'mk': mk, 'mean_power': mean_power}

    def beta_field_stats(self, active_mask=None):
        """Per-active-feature stats of the reconstructed coefficient field
        ``beta_j(x) = sum_b gamma_{j,b} B_b(x)`` over the grid -- ``mu``, ``std``,
        ``median``, ``mad`` (length nf). Lets callers compare report-form
        variants (std/mu, mad/median, squared) on the SAME gammas. Requires the
        basis values (direct-construction path only); ``None`` otherwise.
        """
        Bvals = getattr(self, '_Bvals', None)
        sol = self._solve_gammas(active_mask)
        if sol is None or Bvals is None:
            return None
        gamma, nf, B = sol['gamma'], sol['nf'], sol['B']
        mu = np.empty(nf); sd = np.empty(nf)
        med = np.empty(nf); mad = np.empty(nf)
        for i in range(nf):
            beta = Bvals @ gamma[i * B:(i + 1) * B]     # (N,) coefficient field
            mu[i] = float(np.mean(beta))
            sd[i] = float(np.std(beta))
            m = float(np.median(beta))
            med[i] = m
            mad[i] = float(np.median(np.abs(beta - m)))
        return {'mu': mu, 'std': sd, 'median': med, 'mad': mad}

    def score(self, active_mask=None, *, component=None):
        """Per-active-feature instability score (biased NC, uncapped):

            score_j = NC_j / gamma_{j,0}^2,   NC_j = sum_{k>=1} gamma_{j,k}^2.

        The non-constant energy of the term's varying coefficient ``beta_j(x)``
        over the squared constant part ``gamma_{j,0}^2``. A true (homogeneous)
        term fits a well-identified CONSTANT coefficient -> modes ~0 -> score ~0
        (kept); a spurious term whose coefficient VARIES across the region has
        large mode energy -> large score. ``gamma_0`` is the Frisch-Waugh
        features-only OLS coefficient, so a surviving real term's ``gamma_0``
        stays away from 0 and the ratio is finite without a cap; a degenerate
        ~0-coefficient form (``gamma_0^2 == 0``) maps to +inf so trivial forms
        (e.g. ``u_xx = 0``) are pushed off the Pareto front.

        ``component`` is accepted for backward compatibility but IGNORED: the
        significance (``Var(gamma_0)``) and debias channels were removed, so only
        the biased NC energy remains. Aligned to the active feature columns -- a
        drop-in for the ``get_cv`` return (``active_thresholds = score *
        max_corr`` in ``PhysicsInformedLasso.fit``); the equation-level objective
        sums it over the non-zero terms (see ``vc_stability_total_lr``).
        """
        sol = self._solve_gammas(active_mask)
        if sol is None:
            return np.zeros(0)
        gamma = sol['gamma']
        nf, B, mk = sol['nf'], sol['B'], sol['mk']
        is_const = mk == 0
        nonconst = ~is_const
        scores = np.empty(nf)
        for i in range(nf):
            g = gamma[i * B:(i + 1) * B]
            g0 = float(g[is_const][0]) if np.any(is_const) else 0.0
            C = g0 ** 2
            nc = float(np.sum(g[nonconst] ** 2))
            scores[i] = nc / C if C > 0.0 else np.inf
        return np.nan_to_num(scores)


def vc_stability_total_lr(features, target, sample_weights, grid_shape,
                          main_var: str = None, fit_intercept: bool = True):
    """Equation-level varying-coefficient stability: the SUM over the equation's
    terms of the per-term ``score`` ``NC/gamma_0^2`` (biased non-constant energy;
    the ``gram_mode='vcoef'`` replacement for the inline ``total_lr``).
    Lower = more stable. Pass non-zero-term features (``evaluate(normalize=
    False)``) and ``fit_intercept = weights_final[-1] != 0`` so neither
    zero-weight terms nor a zeroed intercept (a ~0 coefficient that would blow
    up the 1/gamma_0^2 ratio) enter the sum.
    """
    X = np.asarray(features)
    if X.ndim == 1:
        X = X[:, None]
    setup = VaryingCoefSetup(X, target, sample_weights, grid_shape,
                             main_var=main_var, fit_intercept=fit_intercept)
    return float(np.sum(setup.score(None)))


def calculate_weights(X, y, sample_weights, grid_shape, fit_intercept=True,
                      gram_cls=None, gram_kwargs=None):
    """
    Vectorized calculation of weights across sliding windows.
    Dynamically handles whether the intercept should be fit.

    Single-shot wrapper over :class:`GramSetup`: builds the precomputed
    Gram once and immediately solves with the requested intercept policy.
    Callers that solve the same Gram against many active masks (e.g.
    :class:`PhysicsInformedLasso.fit`) should instantiate ``GramSetup``
    directly and call ``.solve(active_mask)`` per iteration to avoid
    re-running the expensive ``X^T diag(w) X`` matmul.

    ``gram_cls`` selects the construction strategy: default ``None`` ->
    ``GramSetup`` (axis-aligned sliding windows, the backup path).
    ``gram_kwargs`` is forwarded to the chosen class's constructor. The
    varying-coefficient default (``gram_mode='vcoef'``) does not route
    through here -- ``PhysicsInformedLasso.fit`` instantiates
    ``VaryingCoefSetup`` directly.
    """
    if gram_cls is None:
        gram_cls = GramSetup
    gram_kwargs = gram_kwargs or {}
    setup = gram_cls(X, y, sample_weights, grid_shape, **gram_kwargs)
    active_mask = np.ones(setup.n_features_aug, dtype=bool)
    if not fit_intercept:
        # GramSetup always augments with the intercept column; drop it
        # from the active set to mimic the legacy ``fit_intercept=False``
        # branch (which never augmented in the first place).
        active_mask[-1] = False
    return setup.solve(active_mask)
