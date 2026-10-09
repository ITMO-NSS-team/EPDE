"""Score exact drift equations through EPDE's own operators (no evolution).

Builds the same search as ``swarm_drift_epde.py`` (forward increments, knee sparsity, 10 robots
of one recording, all interaction tokens), creates the pool, then for each candidate form:

  (a) FIXED support: OLS refit on the written structure (LinRegBasedCoeffsEquation), then the
      fitness host -- the objective values the form would carry if it reached the front;
  (b) RPS: the right-part selector's own fit step (sparsity -> coefficients -> score) for the
      dx/dt (dy/dt) target -- does knee keep every written term, and is the target accepted?

    python score_drift_forms.py --file N40_PWM20_D41 --robots 10 --lag 10
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# insert, not append: a stale pip-installed epde would otherwise shadow the repo
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../..')))
sys.path.insert(0, HERE)

import numpy as np

from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.interface.search_config import active_config
from epde import CacheStoredTokens
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import Discrepancy, Instability
from epde.operators.common.right_part_selection import EqRightPartSelector
from epde.operators.common.sparsity import build_sparsity_operator
from epde.operators.utils.template import add_base_param_to_operator

from swarm_data import swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates, contact_features
from swarm_drift_epde import LABELS

P = '{power: 1.0}'
FORMS = {  # name -> right-hand-side token stems (component appended per equation)
    'push':            ['push'],
    'wall':            ['wall'],
    'push+wall':       ['push', 'wall'],
    'push+wall+gear':  ['push', 'wall', 'gear'],
    'push+cross-push': ['push', 'cross-push'],      # the front's push_x + push_y variant
}


def build(file_key, robots, lag, sparsity, instability=None):
    path = next(p for p in swarm_files() if swarm_meta(p)["tag"] == file_key)
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    feats = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    chosen = np.linspace(0, x.shape[1] - 1, robots).astype(int)
    search = EpdeSearch(multiobjective_mode=True, use_solver=False, sparsity_cls=sparsity,
                        **({'instability_metric': instability} if instability else {}),
                        boundary_width=lag + 1, default_preprocessor_type='forward',
                        preprocessor_kwargs={'lag': lag}, verbose_params={'show_iter_idx': False})
    _, domain = search.createDomain((t,), ID=0)
    fam = CacheStoredTokens(token_type='interaction', token_labels=LABELS, token_tensors=None,
                            params_ranges={'power': (1, 1)}, params_equality_ranges=None, meaningful=True)
    trajs = [search.createTrajectory({'x': x[:, k] - xc, 'y': y[:, k] - yc}, domain, cache_id=i,
                                     cached_token_tensors={fam: {lab: feats[lab][:, k] for lab in LABELS}})[1]
             for i, k in enumerate(chosen)]
    search._run_optimization = lambda *a, **k: None       # pool + active config, no evolution
    search.fit(data=trajs, max_deriv_order=(1,), data_fun_pow=1, equation_terms_max_number=5,
               additional_tokens=[fam], equation_factors_max_number={'factors_num': [1], 'probas': [1.0]})
    return search


def operators():
    disc = Discrepancy()
    host = SolverFreeFitness(['penalty_coeff'], objectives=[disc, Instability()], primary=disc)
    add_base_param_to_operator(host, {})
    cfg = active_config().objectives
    sparsity = build_sparsity_operator(cfg.sparsity_cls, cfg.sparsity_kwargs)
    selector = EqRightPartSelector()
    selector.set_suboperators({'fitness_calculation': host, 'sparsity': sparsity,
                               'coeff_calc': LinRegBasedCoeffsEquation()})
    return host, selector, type(sparsity).__name__


def form_text(stems, comp):
    other = 'y' if comp == 'x' else 'x'
    terms = [f"1.0 * {('push_' + other) if s == 'cross-push' else f'{s}_{comp}'}{P}" for s in stems]
    return ' + '.join(terms) + f' + 1.0 = d{comp}/dx0{P}'


def objective_values(eq, host):
    host.apply(eq, {})
    return {f.value_attr: float(getattr(eq, f.value_attr)) for f in host.objectives}


def terms_of(eq):
    tgt = eq.target_idx
    return {eq.structure[k].name.replace(P, ''): float(eq.weights_final[eq.weight_index(k, tgt)])
            for k in range(len(eq.structure)) if k != tgt}, float(eq.weights_final[-1])


def score(search, host, selector, stems):
    texts = {c: form_text(stems, c) for c in ('x', 'y')}
    out = {}
    for mode in ('fixed', 'rps'):
        system = translate_equation(dict(texts), search.pool, all_vars=['x', 'y'])
        for eq in system.vals:
            comp = eq.target.name[1]                       # 'dx/dx0...' -> 'x'
            eq.main_var_to_explain = comp
            if mode == 'fixed':
                support = np.ones(len(eq.structure))
                eq.weights_internal = support
                eq.weights_internal_evald = True
                LinRegBasedCoeffsEquation().apply(eq, {})
                accepted = True
            else:
                eq.weights_internal_evald = eq.weights_final_evald = False
                _, subop_args = selector.parse_suboperator_args(arguments={})
                accepted = selector._fit_and_score(eq, subop_args) is not None
            coefs, icpt = terms_of(eq)
            out[(mode, comp)] = dict(accepted=accepted, coefs=coefs, icpt=icpt, **objective_values(eq, host))
    return texts, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="N40_PWM20_D41")
    ap.add_argument("--robots", type=int, default=10)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--sparsity", default="knee")
    ap.add_argument("--instability", default=None, help="instability metric, e.g. cv (default: chi2)")
    a = ap.parse_args()
    search = build(a.file, a.robots, a.lag, a.sparsity, a.instability)
    print(f"instability metric in use: {active_config().objectives.instability_metric}")
    host, selector, sp_name = operators()
    print(f"\n{a.file}, {a.robots} robots, lag {a.lag}; sparsity operator {sp_name}; "
          f"objective attrs {[f.value_attr for f in host.objectives]}")
    for name, stems in FORMS.items():
        texts, res = score(search, host, selector, stems)
        print(f"\n=== {name}:  {texts['x']}")
        for (mode, comp), r in res.items():
            vals = '  '.join(f"{k} {v:.4f}" if abs(v) < 1e3 else f"{k} {v:.3e}"
                             for k, v in r.items() if k not in ('accepted', 'coefs', 'icpt'))
            coefs = ', '.join(f"{k} {v:+.2f}" for k, v in r['coefs'].items())
            print(f"  {mode:5s} d{comp}/dt: {'accepted' if r['accepted'] else 'DECLINED'}  {vals}  | {coefs}, c0 {r['icpt']:+.2f}")
