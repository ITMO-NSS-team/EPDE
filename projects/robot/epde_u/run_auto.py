"""Battery runner (common/run_common.py) with an all-auto mode (EPDE_AUTO=1): boundary_width,
population_size, training_epochs and token dimensionality left to EPDE's 'auto' defaults; everything else as
the common battery. Reports what each resolved to. Writes auto_<system><tag>.json next to this file.

One system through EPDE's assignment mode with ONE search space and ONE preprocessing for every system.

USER (Oct 20 2026): "make sure that search space as general and as common as possible for all systems as well as
preprocessing etc", with: trig frequencies from the data; M = 6; screen vs no screen; 5 epochs; full records.
The pic/data scripts are NOT run: each system's full record is loaded here (same files and grids as the
script's loader), and every system gets the same settings:

  preprocessing  finite differences, no smoothing (set_preprocessor 'FD'); boundary 10% of each axis's points
                 at each end (round, at least 1)
  variables      powers 1..3 (data_fun_pow=3)
  derivatives    time axis up to order 2, every space axis up to order 4; powers 1..2 (deriv_fun_pow=2)
  grid tokens    one per axis (t, x), powers 1..2
  trig tokens    sin / cos on every axis, the 2 strongest spectral peaks of the data (freq_from_data=2)
  terms          1 or 2 factors, probabilities 0.65 / 0.35
  objectives     assignment mode, WAPE + dyad, no solver
  budget         population 16, 5 epochs
  K / M / B      EPDE_TERMS, EPDE_SELECTED, EPDE_SCREEN (B unset = every subset of <= M columns)

python run_common.py <system>; env EPDE_SEED, EPDE_TAG, EPDE_EPOCHS, EPDE_POP. Writes common_<system><tag>.json
next to this file. Nothing is written to the repo.
"""
import json
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))     # projects/robot/epde_u -> the repository root
DATA = os.path.join(REPO, 'projects', 'pic', 'data')
# Nov 8 (wire U): EPDE_REPO imports epde from another checkout (a worktree of the pre-edit HEAD for the reference
# arms); the data stay under REPO.
sys.path.insert(0, os.environ.get('EPDE_REPO') or REPO)

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')

import epde  # noqa: E402
assert os.path.abspath(epde.__file__).startswith(os.environ.get('EPDE_REPO') or REPO), epde.__file__
from epde.interface.interface import EpdeSearch  # noqa: E402
from epde.interface.prepared_tokens import GridTokens, TrigonometricTokens  # noqa: E402
import epde.operators.common.subset_search as SS  # noqa: E402

for _name in ('visualize_solutions', 'plot_pareto'):
    setattr(EpdeSearch, _name, lambda self, *a, **k: None)


def _path(*parts):
    return os.path.join(DATA, *parts)


# ---------------------------------------------------------------------------------------------------------------
# Data: the full record of each system, axes (t[, x]) as 1-D coordinate arrays.
# ---------------------------------------------------------------------------------------------------------------
SPEC = {}           # robot systems (robot_systems.py): extra token families, prep / orders / factors overrides


def load(name):
    import robot_systems as RS
    if RS.is_robot(name):
        axes, fields, spec = RS.load(name)
        SPEC.clear()
        SPEC.update(spec)
        return axes, fields
    if name == 'ode':                       # ode.py ODE_simple_discovery: x = sin t + 1.3 cos t
        t = np.linspace(0, 4 * np.pi, 200)
        return [t], {'u': np.sin(t) + 1.3 * np.cos(t)}
    if name == 'vdp':
        return [np.arange(0, 16.0, 0.05)], {'u': np.load(_path('vdp', 'vdp_data.npy'))}
    if name == 'lv':                        # full record (lv.py keeps the first 150 of 301)
        d = np.load(_path('lv', 'data_20.npy'))
        return [np.load(_path('lv', 't_20.npy'))], {'u': d[:, 0], 'v': d[:, 1]}
    if name == 'lorenz':                    # full record (lorenz.py keeps the first 1000 of 100000)
        d = np.load(_path('lorenz', 'lorenz.npy'))
        return [np.load(_path('lorenz', 't.npy'))], {'u': d[:, 0], 'v': d[:, 1], 'w': d[:, 2]}
    if name == 'wave':
        t = np.linspace(0, 1, 81)
        return [t, t.copy()], {'u': np.loadtxt(_path('wave', 'wave_sln_80.csv'), delimiter=',').T}
    if name == 'heat_solar':
        z = np.load(_path('heat_solar', 'heat_soil_uniform_1d_p1.npz'))
        return [z['t'], z['x']], {'u': z['u'].squeeze().T}
    if name in ('burgers', 'kdv'):
        m = loadmat(_path('burgers', 'burgers.mat') if name == 'burgers' else _path('kdv', 'kdv_sindy.mat'))
        return [np.ravel(m['t']), np.ravel(m['x'])], {'u': np.real(m['usol']).T}
    if name == 'ac':
        return [np.linspace(0., 1., 51), np.linspace(-1., 0.984375, 128)], {'u': np.load(_path('ac', 'ac_data.npy'))}
    if name == 'ks':
        m = loadmat(_path('ks', 'kuramoto_sivishinky.mat'))
        return [np.ravel(m['tt']), np.ravel(m['x'])], {'u': m['uu'].T}
    if name in ('pde_compound', 'pde_divide'):
        f = 'PDE_compound.npy' if name == 'pde_compound' else 'PDE_divide.npy'
        return [np.linspace(0, 0.5, 251), np.linspace(1, 2, 100)], {'u': np.load(_path(name, f))}
    if name == 'ns':                        # ns.py ns_data: the first 50 of 200 times, axes (t, y, x)
        m = loadmat(_path('ns', 'cylinder_nektar_wake.mat'))
        x, y = np.unique(m['X_star'][:, 0]), np.unique(m['X_star'][:, 1])
        t = np.ravel(m['t'])
        shape = (len(t), len(y), len(x))
        u = m['U_star'][:, 0, :].T.reshape(shape)[:50]
        v = m['U_star'][:, 1, :].T.reshape(shape)[:50]
        p = m['p_star'].T.reshape(shape)[:50]
        return [t[:50], y, x], {'u': u, 'v': v, 'p': p}
    raise KeyError(name)


# ---------------------------------------------------------------------------------------------------------------
# True forms: (kind, target, [terms]), '1' = the constant. Term keys: factor keys sorted and joined by '*'; a
# factor key is label[^power][~freq]@dim -- trig factors carry their frequency ('~1' when within 1% of 1), so an
# identity in sin t / cos t is credited only at the frequency that makes it exact.
# ---------------------------------------------------------------------------------------------------------------
_ODE_FORMS = [('exact', 'd^2u/dx0^2', ['u']),
              ('wronsk_s', 'du/dx0*sin~1@0', ['cos~1@0*u', '1']),
              ('wronsk_c', 'cos~1@0*du/dx0', ['sin~1@0*u', '1']),
              ('quad', 'du/dx0*sin~1@0', ['u^2', 'du/dx0*u', '1']),
              ('quad_c', 'cos~1@0*du/dx0', ['u^2', 'du/dx0*u', '1']),
              ('sol_x_ut', 'du/dx0*u', ['du/dx0*sin~1@0', 'cos~1@0*du/dx0']),
              # u_t^2 + u^2 = 1 + 1.3^2: the energy identity (the first smoke run put it on L0).
              ('energy', 'du/dx0^2', ['u^2', '1'])]
TRUTH = {
    'ode': {'u': _ODE_FORMS},
    'vdp': {'u': [('exact', 'd^2u/dx0^2', ['du/dx0', 'du/dx0*u^2', 'u'])]},
    'lv': {'u': [('exact', 'du/dx0', ['u', 'u*v'])], 'v': [('exact', 'dv/dx0', ['v', 'u*v'])]},
    'lorenz': {'u': [('exact', 'du/dx0', ['u', 'v'])],
               'v': [('exact', 'dv/dx0', ['u', 'u*w', 'v'])],
               'w': [('exact', 'dw/dx0', ['u*v', 'w'])]},
    'wave': {'u': [('exact', 'd^2u/dx0^2', ['d^2u/dx1^2'])]},
    'heat_solar': {'u': [('exact', 'du/dx0', ['d^2u/dx1^2'])]},
    'burgers': {'u': [('exact', 'du/dx0', ['du/dx1*u', 'd^2u/dx1^2'])]},
    'kdv': {'u': [('exact', 'du/dx0', ['du/dx1*u', 'd^3u/dx1^3'])]},
    'ks': {'u': [('exact', 'du/dx0', ['du/dx1*u', 'd^2u/dx1^2', 'd^4u/dx1^4'])]},
    'ac': {'u': [('exact', 'du/dx0', ['d^2u/dx1^2', 'u', 'u^3'])]},
    'pde_compound': {'u': [('exact', 'du/dx0', ['d^2u/dx1^2*u', 'du/dx1^2'])]},
    # x u_t = -2 u_x + 0.5 x u_xx on this grid (pde_divide.py); x is the grid token of axis 1.
    'pde_divide': {'u': [('exact', 'du/dx0*x@1', ['du/dx1', 'd^2u/dx1^2*x@1'])]},
    # Navier-Stokes on axes (t, y, x): dx0 = t, dx1 = y, dx2 = x. Each equation needs a derivative of its own
    # variable as target: u holds x-momentum or continuity, v y-momentum or continuity, p either momentum.
    # 'cont' = the momentum law with its convective self-term rewritten by continuity (u u_x = -u v_y,
    # v v_y = -v u_x), which counts as true (USER rule).
    'ns': {'u': [('x_mom', 'du/dx0', ['du/dx2*u', 'du/dx1*v', 'dp/dx2', 'd^2u/dx2^2', 'd^2u/dx1^2']),
                 ('x_mom_cont', 'du/dx0', ['dv/dx1*u', 'du/dx1*v', 'dp/dx2', 'd^2u/dx2^2', 'd^2u/dx1^2']),
                 ('continuity', 'du/dx2', ['dv/dx1'])],
           'v': [('y_mom', 'dv/dx0', ['dv/dx2*u', 'dv/dx1*v', 'dp/dx1', 'd^2v/dx2^2', 'd^2v/dx1^2']),
                 ('y_mom_cont', 'dv/dx0', ['dv/dx2*u', 'du/dx2*v', 'dp/dx1', 'd^2v/dx2^2', 'd^2v/dx1^2']),
                 ('continuity', 'dv/dx1', ['du/dx2'])],
           'p': [('x_mom', 'dp/dx2', ['du/dx0', 'du/dx2*u', 'du/dx1*v', 'd^2u/dx2^2', 'd^2u/dx1^2']),
                 ('x_mom_cont', 'dp/dx2', ['du/dx0', 'dv/dx1*u', 'du/dx1*v', 'd^2u/dx2^2', 'd^2u/dx1^2']),
                 ('y_mom', 'dp/dx1', ['dv/dx0', 'dv/dx2*u', 'dv/dx1*v', 'd^2v/dx2^2', 'd^2v/dx1^2']),
                 ('y_mom_cont', 'dp/dx1', ['dv/dx0', 'dv/dx2*u', 'du/dx2*v', 'd^2v/dx2^2', 'd^2v/dx1^2'])]},
}


def _nk(s):
    return '*'.join(sorted(x.strip() for x in s.split('*')))


def factor_key(factor):
    params = {d['name']: float(factor.params[i]) for i, d in factor.params_description.items()}
    key = factor.label
    power = params.get('power', 1.0)
    if power != 1.0:
        key += '^%g' % power
    if 'freq' in params:
        key += '~1' if abs(params['freq'] - 1.0) < 0.01 else '~%.4g' % params['freq']
    if 'dim' in params:
        key += '@%d' % int(round(params['dim']))
    return key


def term_key(term):
    return '*'.join(sorted(factor_key(f) for f in term.structure))


def truth_forms(name):
    return {v: [(kind, _nk(t), [k if k == '1' else _nk(k) for k in keys]) for kind, t, keys in forms]
            for v, forms in _truth(name).items()}


def _truth(name):
    import robot_systems as RS
    return RS.truth(name) if RS.is_robot(name) else TRUTH[name]


def active_struct(eq):
    """The equation's target + nonzero-weight terms (keys), and its constant."""
    if eq.target is None:
        return None, None
    tgt = eq.target_idx
    w = np.asarray(eq.weights_final, dtype=float)
    keys = {term_key(eq.structure[tgt])}
    for i, term in enumerate(eq.structure):
        if i != tgt and w[eq.weight_index(i, tgt)] != 0.0:
            keys.add(term_key(term))
    return frozenset(keys), float(w[-1])


# ---------------------------------------------------------------------------------------------------------------
# Screen recall probe: whenever the screen runs on an equation whose target and terms hold a true written form,
# is that form's subset (with the constant iff the form has one) among the screened subsets?
# ---------------------------------------------------------------------------------------------------------------
_screen = SS.screened_masks
RECALL = dict(screens=0, available=0, kept=0, over_cap=0, by_size={}, lost=[])
FORMS = {}
COUNTS = {}       # plain fits (PF._versions calls / versions kept) and assignment-mode equation calls


def probe_screen(X, y, sw, selected_max, screen_size, order=None):
    masks = _screen(X, y, sw, selected_max, screen_size, order)
    RECALL['screens'] += 1
    caller = sys._getframe(1).f_locals
    work, terms = caller.get('work'), caller.get('terms')
    if work is None or terms is None:
        RECALL['no_frame'] = RECALL.get('no_frame', 0) + 1
        return masks
    tkey = term_key(work.structure[work.target_idx])
    fkeys = [term_key(t) for t in terms]
    p = len(fkeys)
    for kind, t, keys in FORMS.get(work.main_var_to_explain, []):
        written = [t] + [k for k in keys if k != '1']
        if tkey not in written:
            continue
        rhs = [k for k in written if k != tkey]
        if not all(k in fkeys for k in rhs):
            continue
        mask = sum(1 << fkeys.index(k) for k in rhs) | ((1 << p) if '1' in keys else 0)
        size = bin(mask).count('1')
        if size > selected_max:
            RECALL['over_cap'] += 1
            continue
        kept = mask in masks
        RECALL['available'] += 1
        RECALL['kept'] += kept
        tally = RECALL['by_size'].setdefault(size, [0, 0])
        tally[0] += 1
        tally[1] += kept
        if not kept and len(RECALL['lost']) < 50:
            RECALL['lost'].append(dict(var=work.main_var_to_explain, kind=kind, target=tkey, size=size,
                                       n_cols=p + 1))
    return masks


SS.screened_masks = probe_screen


def env_int(name, default=None):
    value = os.environ.get(name)
    return int(value) if value else default


def main():
    name = sys.argv[1]
    FORMS.update(truth_forms(name))
    seed = env_int('EPDE_SEED')
    notes = []
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        notes.append('seed %d (PYTHONHASHSEED %s)' % (seed, os.environ.get('PYTHONHASHSEED')))
    K, M, B = env_int('EPDE_TERMS', 12), env_int('EPDE_SELECTED', 6), env_int('EPDE_SCREEN')
    epochs, pop = env_int('EPDE_EPOCHS', 5), env_int('EPDE_POP', 16)
    # USER Oct 28 "remove max row count option": domain.evaluation_max_rows is gone from EPDE; EPDE_ROWS refuses.
    if os.environ.get('EPDE_ROWS'):
        raise SystemExit('EPDE_ROWS (evaluation_max_rows) was removed; use EPDE_STRIDE (per-axis strides)')
    rows = None
    # USER Oct 28 ("max rows is bad ... we mix time and space"): per-axis strides as JSON, e.g. '[2, 1, 1]' (every 2nd
    # time slice, space whole) -> domain.evaluation_stride; derivatives still on the full grid.
    stride = json.loads(os.environ['EPDE_STRIDE']) if os.environ.get('EPDE_STRIDE') else None
    auto = bool(env_int('EPDE_AUTO', 0))
    # The plain / NSGA-II arms (USER Oct 21): EPDE_RPS_MODE=plain, EPDE_ENGINE=nsga2.
    rps_mode = os.environ.get('EPDE_RPS_MODE', 'assignment')
    engine = os.environ.get('EPDE_ENGINE', 'moeadd')
    # USER Oct 22: pde_compound with other differentiations (EPDE_PREP = 'poly' / 'spectral'; never 'ANN').
    prep = os.environ.get('EPDE_PREP', 'FD')
    # The preprocessor's kwargs as JSON, e.g. '{"accuracy": 4}' (USER Oct 22: pde_compound, 4th-order FD).
    # USER Oct 23 made FD accuracy 4 EPDE's default; unset here, FD keeps the battery's accuracy 2 so the runs stay
    # comparable with the earlier ones (pass '{"accuracy": 4}', or '{}' for EPDE's default, to change it).
    prep_kwargs = json.loads(os.environ.get('EPDE_PREP_KWARGS') or ('{"accuracy": 2}' if prep == 'FD' else '{}'))
    # USER Oct 22 (fronts and weights): evolution.niching / complementary_mating_prob, passed only when set.
    niche_kw = {}
    # USER Oct 23: dyad_t removed ("remove dyad_t, leave full only"); 'dyad' (every axis) is the battery's metric.
    instability = os.environ.get('EPDE_INSTABILITY') or 'dyad'
    if os.environ.get('EPDE_NICHING'):
        niche_kw['niching'] = os.environ['EPDE_NICHING']
    if os.environ.get('EPDE_MATING'):
        niche_kw['complementary_mating_prob'] = float(os.environ['EPDE_MATING'])
    # Oct 27 (lv investigation): operator parameter overrides as JSON, e.g.
    # '{"ChromosomeCrossover": {"equation_exchange_prob": 0.5}}'; passed only when set.
    if os.environ.get('EPDE_OPERATORS'):
        niche_kw['operators'] = json.loads(os.environ['EPDE_OPERATORS'])
    # Nov 8 (wire U into EPDE): plain-mode parameters as JSON, e.g. '{"variable_order_seed": 0}'; passed only when set.
    if os.environ.get('EPDE_RPS_PARAMS'):
        niche_kw['system_rps_params'] = json.loads(os.environ['EPDE_RPS_PARAMS'])
    if rps_mode == 'plain':
        B = None                            # plain mode has no subset screen
        # Oct 27: plain mode's cap is equation_terms_max_number - 1 and it refuses the subset cap M (assignment
        # only), so the battery's cap of M kept terms besides the target is K = M + 1.
        if M is not None:
            K, M = M + 1, None
    import epde.operators.common.plain_fit as PF
    import epde.operators.common.right_part_selection as RPS
    COUNTS.update(plain_calls=0, plain_versions=0, assignment_calls=0)
    versions_fn, variants_fn = PF._versions, RPS._equation_variants

    def counted_versions(*args, **kwargs):
        out = versions_fn(*args, **kwargs)
        COUNTS['plain_calls'] += 1
        COUNTS['plain_versions'] += len(out)
        return out

    def counted_variants(*args, **kwargs):
        COUNTS['assignment_calls'] += 1
        return variants_fn(*args, **kwargs)
    PF._versions, RPS._equation_variants = counted_versions, counted_variants

    axes, fields = load(name)
    # Nov 5 2026 (weak-enhance plan): EPDE_FORM ('strong' default / 'weak' = domain.evaluation_form) and EPDE_NOISE
    # (fraction of each field's std as white Gaussian noise, rng [EPDE_SEED or 0, system index] as the weak smoke, so
    # one (system, seed) has the same noise in every arm and form). Weak rows pick their own window lattice and refuse
    # evaluation_stride. The new records go to the sidecar auto_<sys><tag>_form.json (strong clean JSON unchanged).
    import weakio
    import epde_digest
    form = os.environ.get('EPDE_FORM') or 'strong'
    if form not in ('strong', 'weak'):
        raise SystemExit('EPDE_FORM must be strong or weak, got %r' % form)
    noise = float(os.environ.get('EPDE_NOISE') or 0.0)
    if form == 'weak' and stride is not None:
        raise SystemExit('EPDE_FORM=weak refuses EPDE_STRIDE: weak rows pick their own window lattice')
    if noise:
        nrng = np.random.default_rng([seed or 0, list(TRUTH).index(name)])
        fields = {v: a + noise * a.std() * nrng.standard_normal(a.shape) for v, a in fields.items()}
    if form == 'weak':
        niche_kw['evaluation_form'] = 'weak'
    sidecar = dict(system=name, form=form, noise=noise, seed=seed, noise_digest=weakio.noise_digest(fields),
                   epochs_requested=os.environ.get('TRACE_EPOCHS'), trace_arm=os.environ.get('TRACE_ARM'),
                   epde_digest_start=epde_digest.digest())
    shape = tuple(len(a) for a in axes)
    for v, arr in fields.items():
        assert arr.shape == shape, (name, v, arr.shape, shape)
        assert np.all(np.isfinite(arr)), (name, v)
    dims = len(axes)
    boundary = [max(1, int(round(0.1 * n))) for n in shape]
    orders = (2,) + (4,) * (dims - 1)
    factors = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
    if SPEC:                                 # a robot system: its own overrides (robot_systems.py)
        orders, factors = tuple(SPEC['orders']), dict(SPEC['factors'])
        if SPEC.get('prep_kwargs') is not None:
            prep_kwargs = dict(SPEC['prep_kwargs'])
        print('ROBOT SPEC: orders %s, factors %s, prep %s, families %s' % (
            orders, factors, prep_kwargs, {k: sorted(v) for k, v in SPEC['families'].items()}))
    grids = np.meshgrid(*axes, indexing='ij') if dims > 1 else [axes[0]]
    labels = ['t', 'x', 'y'][:dims]

    def build(with_trig):
        if seed is not None:                # the same draws whether or not a first attempt failed
            random.seed(seed)
            np.random.seed(seed)
        budget = {}
        if stride is not None:
            budget['evaluation_stride'] = stride
        search = EpdeSearch(use_solver=False, multiobjective_mode=True, system_rps_mode=rps_mode,
                            engine=engine, discrepancy_metric='wape', instability_metric=instability,
                            verbose_params={'show_iter_idx': False}, device='cpu', **budget, **niche_kw)
        if SPEC.get('boundary') is not None:  # a robot system's data-forced boundary (token history)
            _, domain = search.createDomain(tuple(grids), boundary_width=list(SPEC['boundary']), ID=0)
        elif auto:
            _, domain = search.createDomain(tuple(grids), ID=0)
        else:
            _, domain = search.createDomain(tuple(grids), boundary_width=boundary, ID=0)
        search.set_preprocessor(default_preprocessor_type=prep, preprocessor_kwargs=prep_kwargs)
        if not auto:
            search.set_moeadd_params(population_size=pop, training_epochs=epochs)
        _, trajectory = search.createTrajectory(fields, domain, cache_id=0)
        dim_kw = {} if auto else {'dimensionality': dims - 1}
        extra_families = []                 # robot token families need the search's samples manager
        if SPEC:
            from epde import CacheStoredTokens
            for fam_name, labs in SPEC['families'].items():
                extra_families.append(CacheStoredTokens(
                    token_type=fam_name, token_labels=list(labs), token_tensors=dict(labs),
                    params_ranges={'power': (1, 1)}, params_equality_ranges=None, meaningful=True))
        tokens = ([TrigonometricTokens(freq_from_data=2, **dim_kw)] if with_trig else []) + \
                 [GridTokens(labels=labels, max_power=2, **dim_kw)] + extra_families
        search._auto_domain = domain
        fit_kwargs = dict(data=[trajectory], max_deriv_order=orders, data_fun_pow=3, deriv_fun_pow=2,
                          equation_terms_max_number=K, additional_tokens=tokens,
                          equation_factors_max_number=factors)
        # Oct 27: fit() refuses the subset cap / screen keywords outright (they are assignment-mode settings now);
        # passed only when set, i.e. in assignment mode.
        if M is not None:
            fit_kwargs['equation_selected_terms_max_number'] = M
        if B is not None:
            fit_kwargs['equation_screen_size'] = B
        return search, fit_kwargs

    settings = dict(factors=factors, robot_families=sorted(SPEC.get('families', {})), shape=shape, boundary=boundary, max_deriv_order=orders, K=K, M=M, B=B, rows=rows, stride=stride,
                    epochs=epochs,
                    population=pop, rps_mode=rps_mode, engine=engine, prep=prep, prep_kwargs=prep_kwargs,
                    instability=instability, **niche_kw)
    print('COMMON RUN %s: %s' % (name, settings))
    t0 = time.perf_counter()
    search, fit_kwargs = build(with_trig=True)
    try:
        search.fit(**fit_kwargs)
    except ValueError as err:
        # The one data-forced difference: a record without a spectral peak has no frequency to give the trig
        # family, and EPDE refuses to build it. That system runs without trig tokens.
        if 'freq_from_data' not in str(err):
            raise
        notes.append('no spectral peak in the data: trig family dropped (%s)' % str(err).split(' -- ')[0])
        print('NOTE: ' + notes[-1])
        RECALL.update(screens=0, available=0, kept=0, over_cap=0, by_size={}, lost=[])
        search, fit_kwargs = build(with_trig=False)
        search.fit(**fit_kwargs)
    settings['trig'] = not any('trig family dropped' in n for n in notes)
    elapsed = time.perf_counter() - t0
    if auto:
        settings.update(auto=True, boundary=list(search._auto_domain.boundary_width_per_axis),
                        population=len(search.optimizer.pareto_levels.population),
                        epochs=len(search.optimizer._pareto_history))
        print('AUTO RESOLVED: boundary %s, population %d, epochs %d' % (
            settings['boundary'], settings['population'], settings['epochs']))

    truth = {v: [(kind, frozenset([t] + [k for k in keys if k != '1'])) for kind, t, keys in forms]
             for v, forms in FORMS.items()}
    report = dict(system=name, seconds=round(elapsed, 1), notes=notes, settings=settings, levels=[])
    full_l0 = full_any = 0
    per_var_true = {v: 0 for v in truth}
    for lvl, level in enumerate(search._resulting_population):
        rows = []
        for sol in level:
            marks, consts = {}, {}
            for v in truth:
                struct, const = active_struct(sol.vals[v])
                marks[v] = next((kind for kind, s in truth[v] if s == struct), None)
                consts[v] = const
                per_var_true[v] += marks[v] is not None
            full = all(marks.values())
            full_any += full
            full_l0 += full and lvl == 0
            try:                            # robot campaign: the objectives and per-variable structure keys too
                obj = [float(x) for x in np.ravel(sol.obj_fun)]
            except Exception:
                obj = None
            keys = {}
            for v in truth:
                struct, _c = active_struct(sol.vals[v])
                eq = sol.vals[v]
                keys[v] = None if struct is None else dict(
                    target=term_key(eq.structure[eq.target_idx]),
                    terms=sorted(struct - {term_key(eq.structure[eq.target_idx])}))
            rows.append(dict(full=full, marks=marks, consts=consts, text=sol.text_form, obj=obj, keys=keys,
                             obj_order=list(sol.vals.equation_keys) if hasattr(sol.vals, 'equation_keys') else None))
            print('  L%d %s %s' % (lvl, 'FULL' if full else '    ',
                                   ' '.join('%s:%s' % (v, m or '-') for v, m in marks.items())))
        report['levels'].append(rows)
    print('\nCOMMON RESULT %s: %.1f s | settings %s | notes %s' % (name, elapsed, settings, notes))
    print('FULL on L0: %d | FULL anywhere: %d | members with a true equation per variable: %s'
          % (full_l0, full_any, per_var_true))
    print('SCREEN RECALL: %s' % {k: v for k, v in RECALL.items() if k != 'lost'})
    print('FIT COUNTS: %s' % COUNTS)
    report['fit_counts'] = dict(COUNTS)
    for row in RECALL['lost'][:10]:
        print('  LOST %s' % row)
    print('FIRST LEVEL:')
    for sol in search._resulting_population[0]:
        print('  ' + sol.text_form.replace('{power: 1.0}', '').replace('\n', '  |  '))
    report.update(full_l0=full_l0, full_any=full_any, per_var_true=per_var_true, screen_recall=RECALL)
    with open(os.path.join(HERE, 'auto_%s%s.json' % (name, os.environ.get('EPDE_TAG', ''))), 'w') as f:
        json.dump(report, f, indent=1, default=str)
    import epde.globals as _GV
    _weak = getattr(_GV.samples_manager, 'weak', None)
    sidecar.update(epochs_run=settings.get('epochs'), population=settings.get('population'),
                   seconds=round(elapsed, 1), weakio=weakio.stats(), epde_digest_end=epde_digest.digest())
    if _weak is not None:
        sidecar['weak_windows'] = {'%s/%s' % key: str(plan) for key, plan in _weak.plans.items()}
        sidecar['one_check'] = {'%s/%s' % key: weakio.one_check(key[1], key[0]) for key in _weak.plans}
    sidecar['digest_equal'] = sidecar['epde_digest_start'] == sidecar['epde_digest_end']
    with open(os.path.join(HERE, 'auto_%s%s_form.json' % (name, os.environ.get('EPDE_TAG', ''))), 'w') as f:
        json.dump(sidecar, f, indent=1, default=str)


if __name__ == '__main__':
    main()
