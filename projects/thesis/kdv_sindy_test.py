"""KdV SINDy seeded-equation diagnostic.

Bypasses MOEA/D search entirely. Seeds the truth KdV equation directly
into the EPDE pipeline, runs ``VWSRSparsity`` + ``L2LRFitness`` on it,
and prints which terms survive the sparsity step plus the per-term
fitness / coefficient-stability / AIC metrics.

Also evaluates a "truncated truth" variant where one term has been
dropped, so we can see by how much the metrics worsen when the
nonlinearity is missing.

Default config matches the thesis NEW pipeline (L2LRFitness +
VWSRSparsity) on the same kdv_sindy.mat data the production search
uses, so the metrics here are directly comparable to what RPS's
term-sweep would see during the search.

Usage:
    python projects/thesis/kdv_sindy_test.py [--gram-mode axis|vcoef]
        [--seed N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np

import epde.globals as global_var
from epde import globals as gv
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import WAPEDiscrepancy, Instability
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde.operators.common.stability import (
    VaryingCoefSetup, calculate_weights,
)

from thesis_runner import (  # noqa: E402
    _boundary_for, _build_token_pool, _configure_preprocessor,
    _construct_search, _set_seeds, load_config, pipeline_settings,
)


# Truth equations are read from configs/<system>.yaml -> truth_equations.
# For multi-equation systems (lorenz, lv, ns) the list has one entry per
# variable; ``translate_equation`` consumes a dict {var: symbolic}.

# Truth YAMLs label standalone grid tokens ``x_0{...}`` / ``x_1{...}`` (legacy
# per-axis names), but the pool registers a single ``x`` family with ``dim`` as
# an inner parameter. ``translate_equation`` matches the bare label against the
# pool, so collapse ``x_N{`` -> ``x{`` first (the ``dim:N`` param already
# disambiguates the axis). Without this, ode (``x_0``) and pde_divide (``x_1``)
# fail to seed. Same normalisation as ``kdv_sindy_sweep`` (kept local to avoid a
# circular import, since that module imports from this one).
_GRID_LABEL_RE = re.compile(r'\bx_\d+(?=\{)')


def _normalize_grid_labels(symbolic):
    """Rewrite ``x_0{...}`` -> ``x{...}`` on a string or dict-of-strings."""
    if isinstance(symbolic, dict):
        return {k: _GRID_LABEL_RE.sub('x', v) for k, v in symbolic.items()}
    return _GRID_LABEL_RE.sub('x', symbolic)


def build_pool_only(cfg, pipeline_kwargs):
    """Run the thesis builder up to (but not including) ``.fit``.

    Returns a primed ``EpdeSearch`` whose pool is built and
    preprocessor is configured, so ``translate_equation`` can resolve
    every token. The optimizer is never instantiated, so this is fast.
    """
    coords, data, variable_names, dim = cfg.load_data()
    additional_tokens = _build_token_pool(cfg, coords, dim)
    search = _construct_search(cfg, coords, pipeline_kwargs)
    _configure_preprocessor(search, cfg)
    f = cfg.hparams['fit']
    max_deriv_order = f.get('max_deriv_order')
    if max_deriv_order is None:
        max_deriv_order = (2,) + (4,) * dim
    else:
        max_deriv_order = tuple(max_deriv_order)
    search.create_pool(
        data=data,
        variable_names=variable_names,
        max_deriv_order=max_deriv_order,
        additional_tokens=additional_tokens,
        data_fun_pow=f['data_fun_pow'],
        deriv_fun_pow=f['deriv_fun_pow'],
        fourier_layers=f['fourier_layers'],
    )
    return search


def make_fit_operator():
    """Build a gene-level ``SolverFreeFitness`` (WAPE discrepancy +
    instability) + ``VWSRSparsity`` chain -- the new-pipeline solver-free
    fitness.

    We intentionally do NOT map to chromosome level here: the test
    drives the gene-level operator per-equation by hand with
    ``force_out_of_place=True`` so the sparsity sub-operator actually
    runs (the chromosome wrapper short-circuits sparsity when
    ``fitness_calculated`` is False, leaving the seeded weights
    untouched).
    """
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    disc = WAPEDiscrepancy()
    fit_op = SolverFreeFitness(list(op_params.keys()),
                               objectives=[disc, Instability()], primary=disc)
    fit_op.params = op_params
    fit_op.set_suboperators({
        'sparsity': VWSRSparsity(),
        'coeff_calc': LinRegBasedCoeffsEquation(),
    })
    return fit_op


def _per_term_cv_stats(weights_arr: np.ndarray) -> dict:
    """Per-feature distribution stats over a (M, n_features) per-window
    OLS weight stack. Returns arrays of shape (n_features,).
    """
    if weights_arr.ndim == 1:
        weights_arr = weights_arr[:, None]
    mu = weights_arr.mean(axis=0)
    std = weights_arr.std(axis=0, ddof=1)
    median = np.median(weights_arr, axis=0)
    mad = np.median(np.abs(weights_arr - median), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_sqr_std = np.nan_to_num((std ** 2) / (mu ** 2))
        cv_lin_std = np.nan_to_num(std / np.abs(mu))
        cv_sqr_mad = np.nan_to_num((mad ** 2) / (median ** 2))
    return {
        'mu': mu, 'std': std, 'median': median, 'mad': mad,
        'cv_sqr_std': cv_sqr_std,
        'cv_lin_std': cv_lin_std,
        'cv_sqr_mad': cv_sqr_mad,
        'M': int(weights_arr.shape[0]),
    }


def _gram_kwargs_for_current_mode() -> tuple:
    """Diagnostic re-compute of per-window weights uses the axis backup
    ``GramSetup`` (``calculate_weights``'s default), so this returns
    ``(None, None)``. The vcoef default does not route through
    ``calculate_weights`` -- it scores via ``VaryingCoefSetup`` directly.
    """
    return (None, None)


def inspect_truth(symbolic, search, fit_op, all_vars=('u',)):
    """Seed the truth equation; for every non-target term, print the
    per-window OLS coefficient distribution, all three CV formulas, and
    the LASSO threshold each term would face on iteration 1. Then run
    sparsity to see which actually get dropped.

    The per-term table answers the user's question: "which term is
    zeroed, and why" -- the "why" is whichever has CV * max_corr > rho.

    ``symbolic`` may be either a single string (single-equation system
    like KdV) or a dict mapping variable name to symbolic string
    (multi-equation systems like Lorenz).
    """
    metaparams = {
        ('sparsity', v): {'optimizable': False, 'value': 1e-6}
        for v in all_vars
    }
    soeq = translate_equation(symbolic, search.pool, all_vars=list(all_vars))
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.metaparameters = metaparams
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True

    print(f'\n{"=" * 84}\nTRUTH per-term diagnostic\n{"=" * 84}')
    print(f'  symbolic: {symbolic}')

    for v in all_vars:
        eq = soeq.vals[v]
        target_idx = eq.target_idx
        feat_terms = [t for i, t in enumerate(eq.structure) if i != target_idx]
        target_term = eq.structure[target_idx]
        print(f'\n  [{v}] target term:  {target_term.name}')

        # --- step 1: dump per-window OLS weights via calculate_weights
        # using the same Gram class the LASSO solver will use.
        _, target, features = eq.evaluate(normalize=True, return_val=False)
        g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]
        data_shape = global_var.grid_cache.inner_shape
        gram_cls, gram_kwargs = _gram_kwargs_for_current_mode()
        weights = np.array(calculate_weights(
            features, target, g_fun_vals, data_shape, True,
            gram_cls=gram_cls, gram_kwargs=gram_kwargs,
        ))  # shape (M, n_features_aug) -- last column is intercept
        stats = _per_term_cv_stats(weights)
        M = stats['M']

        # --- varying-coefficient scores on the SAME (features, target),
        # aligned to [feat_terms..., <intercept>]. ``fit`` is the
        # unbounded 1/significance^2 in-fit Lasso driver; ``report`` is
        # the bounded [0,1] Pareto-objective form.
        try:
            _vc = VaryingCoefSetup(features, target, g_fun_vals,
                                   data_shape, main_var=v)
            vc_score = _vc.score(None)
        except Exception as _e:
            vc_score = np.full(features.shape[1] + 1, np.nan)

        # --- step 2: LASSO threshold context
        # PhysicsInformedLasso uses:
        #     X_aug = [features, ones]
        #     X_T_y = X_aug.T @ target
        #     max_corr = max(|X_T_y[active]|)
        #     active_thresholds[j] = active_cv[j] * max_corr
        # We pick the LIVE formula (sqr(mad/median)) for the threshold so
        # the user sees exactly what the LASSO solver applies. The other
        # two CVs are still printed alongside for comparison.
        X_aug = np.hstack([features, np.ones((features.shape[0], 1))])
        # weighted X_T_y so the threshold lines up with the
        # PhysicsInformedLasso interior weighting.
        X_T_y = X_aug.T @ (g_fun_vals * target if g_fun_vals.shape == target.shape
                            else target)
        max_corr = float(np.max(np.abs(X_T_y)))
        # ``rho`` on iter 1 with all-zero active_coef and residual=y:
        # rho_j = X_aug[:, j] @ (g_fun_vals * y - X_aug @ 0) = X_T_y[j].
        rho_iter1 = X_T_y

        # --- step 3: per-term table
        names = [t.name for t in feat_terms] + ['<intercept>']
        n_print = len(names)
        print(f'  [{v}] per-window weight stack: M={M}, n_features_aug={n_print}')
        print(f'  [{v}] max_corr (=max(|X^T y| over all features)): {max_corr:.6g}')
        print(f'  [{v}] {"#":<3} {"term":<55} '
              f'{"mu":>10} {"median":>10} {"std":>10} {"mad":>10} '
              f'{"sqr(s/mu)":>10} {"lin(s/|mu|)":>11} {"sqr(mad/med)":>12} '
              f'{"vc(var0/C)":>11} '
              f'{"thr=cv*mc":>10} {"|rho1|":>10} {"killed?":>8}')
        for j, name in enumerate(names):
            cv_live = stats['cv_sqr_mad'][j]
            threshold_live = cv_live * max_corr
            killed = abs(rho_iter1[j]) < threshold_live
            vcs = vc_score[j] if j < len(vc_score) else float('nan')
            print(
                f'  [{v}] {j:<3} {name[:55]:<55} '
                f'{stats["mu"][j]:>10.3g} {stats["median"][j]:>10.3g} '
                f'{stats["std"][j]:>10.3g} {stats["mad"][j]:>10.3g} '
                f'{stats["cv_sqr_std"][j]:>10.3g} '
                f'{stats["cv_lin_std"][j]:>11.3g} '
                f'{stats["cv_sqr_mad"][j]:>12.3g} '
                f'{vcs:>11.3g} '
                f'{threshold_live:>10.3g} {abs(rho_iter1[j]):>10.3g} '
                f'{("YES" if killed else "no"):>8}'
            )

        # --- step 4: actually run sparsity to confirm the prediction.
        sweep_fitness = fit_op.apply(eq, {}, force_out_of_place=True)
        print(f'\n  [{v}] after VWSRSparsity.apply:')
        for j, name in enumerate(names):
            if j < len(eq.weights_internal):
                w_post = eq.weights_internal[j]
            else:
                continue
            mark = ' <-- DROPPED' if w_post == 0.0 else ''
            print(f'      {j}: {name[:55]:<55} w_internal={w_post!r}{mark}')
        print(f'  [{v}] sparsity-pass fitness: {sweep_fitness!r}')
        # Run the fitness pass so coef_stability gets updated for ref.
        eq.fitness_calculated = False
        eq.stability_calculated = False
        fit_op.apply(eq, {}, force_out_of_place=False)
        print(f'  [{v}] weights_final: {eq.weights_final}')
        print(f'  [{v}] fitness_value: {eq.fitness_value!r}')
        print(f'  [{v}] coef_stability (live formula): {eq.coefficients_stability!r}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--system', default='kdv',
                   help="System name (loads configs/<system>.yaml). "
                        "Truth equations come from cfg.truth_equations.")
    p.add_argument('--gram-mode', default='vcoef',
                   choices=('axis', 'vcoef'),
                   help="Gram / stability strategy (default: vcoef = "
                        "varying-coefficient stability). 'axis' = legacy "
                        "axis-aligned sliding-window backup (var/mu^2 CV).")
    p.add_argument('--anchor-on-residual', action='store_true', default=False,
                   help="In 'max_corr' mode, anchor on the working residual "
                        "max|X^T r| instead of the raw target max|X^T y|.")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    gv.set_gram_config(args.gram_mode)
    gv.set_anchor_on_residual(args.anchor_on_residual)
    cfg = load_config(args.system)
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    pipeline_kwargs = pipeline_settings('new')
    _set_seeds(args.seed)

    print(f'{args.system} SINDy seeded-equation test')
    print(f'  gram_mode={args.gram_mode}')
    print(f'  anchor_on_residual={args.anchor_on_residual}')
    print(f'  seed={args.seed}')

    search = build_pool_only(cfg, pipeline_kwargs)
    fit_op = make_fit_operator()
    coords, data, variable_names, dim = cfg.load_data()

    # ``SystemCfg`` only retains canonical-token truth; the raw symbolic
    # strings live in the YAML. Re-read them here so ``translate_equation``
    # can build the seeded SoEq. Multi-equation systems (lorenz, lv, ns)
    # need a dict {var: symbolic} so each variable gets the right gene.
    import yaml as _yaml
    cfg_path = os.path.join(_THIS_DIR, 'configs', f'{args.system}.yaml')
    with open(cfg_path) as f:
        truth_eqs = _yaml.safe_load(f).get('truth_equations') or []
    if len(variable_names) == 1:
        seeded = truth_eqs[0]
    else:
        seeded = {var: truth_eqs[i] for i, var in enumerate(variable_names)}
    # Collapse legacy ``x_N{`` grid labels to the pool's ``x{`` so seeding
    # works for systems whose truth uses standalone coordinate tokens
    # (ode: x_0, pde_divide: x_1).
    seeded = _normalize_grid_labels(seeded)

    inspect_truth(seeded, search, fit_op, all_vars=tuple(variable_names))
    return 0


if __name__ == '__main__':
    sys.exit(main())
