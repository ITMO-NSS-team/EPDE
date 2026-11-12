"""lv_trace.py for any battery system (TRACE_SYSTEM: lorenz, ns, lv, ...): plain + NSGA-II, auto settings, every
evaluated system classified per equation against run_auto's TRUTH forms of that variable -- 'law' (the equation's
target + kept terms are exactly one written true form, identities and continuity forms included), 'law+' (a true form
plus extra terms), 'other' -- and the exact system (every equation 'law') logged when it is evaluated.

Env switches: TRACE_SYSTEM, TRACE_TAG, TRACE_SEED, TRACE_FD_KWARGS, TRACE_ARM (comma list, e.g. pv), TRACE_FACTOR_MOVES,
TRACE_SELECTION (fate + co-presence log), TRACE_EPOCHS (the epoch cap of an auto run; 0 = the initial population
only), TRACE_STOP_SHADOW (EPDE's early stop never fires; four stop keys per epoch -> <sys>_trace<TAG>_stop.log and
_stop.jsonl). Outputs: <sys>_trace<TAG>.log / .out / .err, _arm.log, _fate.log, _copresence.log, _planes.log and
run_auto's auto_<sys>_trace<TAG>.json."""
import atexit
import collections
import functools
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get('TRACE_SYSTEM', 'lv')
TAG = os.environ.get('TRACE_TAG', '')
os.environ.update(EPDE_RPS_MODE='plain', EPDE_ENGINE='nsga2', EPDE_PREP='FD', EPDE_PREP_KWARGS='{"accuracy": 2}',
                  EPDE_SEED='0', EPDE_AUTO='1', EPDE_TERMS='12', EPDE_SELECTED='6', EPDE_INSTABILITY='dyad',
                  EPDE_TAG='_trace' + TAG)
if os.environ.get('TRACE_SEED'):
    os.environ['EPDE_SEED'] = os.environ['TRACE_SEED']
if os.environ.get('TRACE_FD_KWARGS'):
    os.environ['EPDE_PREP_KWARGS'] = os.environ['TRACE_FD_KWARGS']
if os.environ.get('TRACE_SELECTED'):         # Oct 31 cap test (USER): kept terms beside the target; plain K = M + 1
    os.environ['EPDE_SELECTED'] = os.environ['TRACE_SELECTED']
sys.argv = [sys.argv[0], NAME]
import run_auto as R  # noqa: E402
from epde.optimizers.nsga import optimizer as NO  # noqa: E402
import arm_pv  # noqa: E402  (its helpers; nothing is installed by the import)

FORMS = R.truth_forms(NAME)
VARS = list(FORMS)
LAWS = {v: [frozenset([t] + [k for k in keys if k != '1']) for _kind, t, keys in FORMS[v]] for v in VARS}
EXACT = tuple('law' for _ in VARS)
PREFIX = '%s_trace%s' % (NAME, TAG)
log = open(os.path.join(HERE, PREFIX + '.log'), 'w')
stats = collections.Counter()
calls = {'n': 0}
FIRST = {}
NEWLAB = {v: set() for v in VARS}
GEN = {'n': 0}
START = time.perf_counter()

# USER Oct 28 "add the log and rerun": EVERY evaluated system as compact equation ids, so identity-aware hitting times
# can be computed afterwards (identity_hits.py). Recording only; the search is untouched. Always on.
#   <PREFIX>_stream.tsv  insert, generation, plain fit calls, one equation id per variable (-1: no target)
#   <PREFIX>_eqs.jsonl   [id, variable, target key, sorted active term keys, constant fitted] once per new equation
# Keys are run_auto.term_key of the target and of the nonzero-weight terms (run_auto.active_struct).
STREAM = open(os.path.join(HERE, PREFIX + '_stream.tsv'), 'w')
EQS = open(os.path.join(HERE, PREFIX + '_eqs.jsonl'), 'w')
STREAM.write('# insert\tgeneration\tplain_calls\t%s\n' % '\t'.join(VARS))
EQ_ID = {}


def stream_write(system):
    try:
        ids = []
        for var in VARS:
            eq = system.vals[var]
            keys, const = R.active_struct(eq)
            if keys is None:
                ids.append('-1')
                continue
            tkey = R.term_key(eq.structure[eq.target_idx])
            key = (var, tkey, tuple(sorted(keys - {tkey})), bool(const))
            idx = EQ_ID.get(key)
            if idx is None:
                idx = EQ_ID[key] = len(EQ_ID)
                EQS.write(json.dumps([idx, var, tkey, list(key[2]), key[3]]) + '\n')
            ids.append(str(idx))
        STREAM.write('%d\t%d\t%s\t%s\n' % (calls['n'], GEN['n'], R.COUNTS.get('plain_calls'), '\t'.join(ids)))
        if calls['n'] % 2000 == 0:
            STREAM.flush()
            EQS.flush()
        return ids
    except Exception:
        stats['stream_errors'] += 1
        return None


# Nov 6 (improvement plan I5b): TRACE_PROV=1 adds <PREFIX>_inserts.jsonl / _prov.jsonl / _prov.log (prov_trace.py:
# per-insert archive log + the operator that created every new equation). Logging only; unset = nothing is imported.
PROV = os.environ.get('TRACE_PROV') == '1'
if PROV:
    import prov_trace as PT  # noqa: E402
    PT.setup(PREFIX, HERE, VARS)


def _close_stream():
    STREAM.close()
    EQS.close()


atexit.register(_close_stream)


def eq_class(eq, var):
    keys, _const = R.active_struct(eq)
    if keys is None:
        return 'none'
    if keys in LAWS[var]:
        return 'law'
    if any(law < keys for law in LAWS[var]):
        return 'law+'
    return 'other'


def classes(system):
    return tuple(eq_class(system.vals[v], v) for v in VARS)


def describe(system):
    try:
        obj = ['%.3g' % x for x in system.obj_fun]
    except Exception:
        obj = '?'
    return '%s obj %s | %s' % (classes(system), obj, system.text_form.split('\n{')[0].replace('\n', ' '))


OPT = {'epochs_run': 0}
real_optimize = NO.NSGAOptimizer.optimize


@functools.wraps(real_optimize)              # the interface filters optimize()'s kwargs by its signature
def remembering_optimize(self, *args, **kwargs):
    OPT['self'] = self
    OPT['epochs_requested'] = kwargs.get('epochs', args[0] if args else None)
    OPT['optimize_start'] = time.perf_counter() - START
    before = len(self._pareto_history)
    try:
        return real_optimize(self, *args, **kwargs)
    finally:
        OPT['epochs_run'] += len(self._pareto_history) - before
        OPT['mu'] = self._mu
        OPT['optimize_end'] = time.perf_counter() - START


NO.NSGAOptimizer.optimize = remembering_optimize

real_insert = NO.NSGAOptimizer._archive_insert


def traced_insert(self, system):
    from epde.optimizers.moeadd.supplementary import _structure_label
    calls['n'] += 1
    cls = classes(system)
    stats[cls] += 1
    if PROV:
        _n_ids = len(EQ_ID)
        _ids = stream_write(system)
        _tok = PT.begin(self, system, _ids, _n_ids, calls['n'], GEN['n']) if _ids is not None else None
    else:
        stream_write(system)
    if cls not in FIRST:
        FIRST[cls] = dict(insert=calls['n'], plain_calls=R.COUNTS.get('plain_calls'),
                          plain_versions=R.COUNTS.get('plain_versions'), generation=GEN['n'],
                          seconds=round(time.perf_counter() - START, 1))
    for var in VARS:
        NEWLAB[var].add(_structure_label(system.vals[var]))
    if cls == EXACT and os.environ.get('TRACE_REJECT_PROBE'):
        # Nov 2 probe (read-only, no RNG): WHY the archive would reject an exact system -- the archived twins (same
        # structures) and the archived systems that dominate it, by the same rules NSGAOptimizer._archive_insert uses.
        from epde.optimizers.moeadd.supplementary import check_dominance
        from epde.optimizers.nsga.optimizer import _values_dominate
        label = system.equations_labels
        twins = [m for m in self._archive if m.equations_labels == label]
        doms = [m for m in self._archive if m.equations_labels != label and check_dominance(m, system)]
        log.write('PROBE #%d: archive %d | twins %d (new dominates twin by value: %s) | dominators %d\n' % (
            calls['n'] + 0, len(self._archive), len(twins),
            [bool(_values_dominate(system, t)) for t in twins], len(doms)))
        for m in (twins + doms)[:6]:
            log.write('PROBE   %s %s\n' % ('TWIN' if any(m is t for t in twins) else 'DOMINATOR', describe(m)))
        log.flush()
    entered = real_insert(self, system)
    if PROV:
        PT.end(self, system, _tok, entered)
    if cls == EXACT:
        stats['exact_entered' if entered else 'exact_rejected'] += 1
        log.write('#%d EVALUATED %s -> %s\n' % (calls['n'], describe(system), 'ENTERED' if entered else 'REJECTED'))
        log.flush()
    return entered


NO.NSGAOptimizer._archive_insert = traced_insert

ARMS = {}
_WEAK_FORM = (os.environ.get('EPDE_FORM') or 'strong') == 'weak'   # Nov 5: weak-enhance plan
if os.environ.get('TRACE_ARM'):
    arm_names = [name.strip() for name in os.environ['TRACE_ARM'].split(',') if name.strip()]
    if {'pv', 'ja'} & set(arm_names):
        if {'pv', 'ja'} <= set(arm_names):
            raise SystemExit('TRACE_ARM: pv and ja cannot be combined')
        for var in ('TRACE_ARCHIVE_MATING', 'EPDE_NICHING', 'EPDE_MATING'):
            if os.environ.get(var):
                raise SystemExit('TRACE_ARM pv/ja own selection and crossover: %s is set' % var)
    # Oct 29 (approved storage plan, "v1 vs v2"): the LEAN bundle (nohist, syshist, snapmemo, transient, copyshare,
    # fitcache2 -- arm_order's canonical order) joins an arm set listed in LEAN_ALLOW beside this script
    # ('<system>:<arm set>' lines), each listed set first checked bit-identical (lean_check*.ps1). The arm log records it.
    _lean = os.path.join(HERE, 'LEAN_ALLOW')
    if os.path.exists(_lean) and not set(arm_names) & {'fitcache', 'fitcache2', 'nohist'}:
        with open(_lean, encoding='utf-8') as _fh:
            _lean_ok = {line.strip() for line in _fh if line.strip() and not line.startswith('#')}
        # Nov 5 (weak-enhance plan): weak rows certify their own lines ('<system>@weak:<arm set>', added by the
        # weak lean check only when IDENTICAL) and never take fitcache2 (its hits need the strong tensor cache).
        if (NAME + '@weak:' if _WEAK_FORM else NAME + ':') + ','.join(arm_names) in _lean_ok:
            arm_names = (['pv', 'nohist', 'syshist', 'snapmemo', 'transient', 'copyshare']
                         + [a for a in arm_names if a != 'pv'] + ([] if _WEAK_FORM else ['fitcache2']))
            os.environ['TRACE_ARM'] = ','.join(arm_names)
    for name in arm_names:
        ARMS[name] = importlib.import_module('arm_' + name)
        ARMS[name].install()
    # USER Oct 28 "run further runs with optimized version if its identical": the plain-fit cache (arm_fitcache) joins
    # the arm sets listed in FITCACHE_ALLOW beside this script (one TRACE_ARM value per line), each first checked
    # bit-identical (fc_check_arms.ps1). No file, or an unlisted arm set -> no cache.
    _allow = os.path.join(HERE, 'FITCACHE_ALLOW')
    if 'fitcache' not in arm_names and os.path.exists(_allow) and not _WEAK_FORM:
        with open(_allow, encoding='utf-8') as _fh:
            _allowed = {line.strip() for line in _fh if line.strip() and not line.startswith('#')}
        # a line is '<arm set>' (every system) or '<system>:<arm set>'
        if ','.join(arm_names) in _allowed or NAME + ':' + ','.join(arm_names) in _allowed:
            ARMS['fitcache'] = importlib.import_module('arm_fitcache')
            ARMS['fitcache'].install()

    def arm_report():
        with open(os.path.join(HERE, PREFIX + '_arm.log'), 'w') as out:
            out.write(json.dumps({name: arm.report() for name, arm in ARMS.items()}, indent=1, default=str))
    atexit.register(arm_report)

real_gen_truncate = NO.NSGAOptimizer._truncate


def counting_truncate(self):
    out = real_gen_truncate(self)
    GEN['n'] += 1
    return out


NO.NSGAOptimizer._truncate = counting_truncate

if os.environ.get('TRACE_SELECTION'):
    # As lv_trace: the fate of law carriers (a 'law' or 'law+' equation in some slot) and, per truncation, each
    # variable's exact law present / on its plane front / kept, and the exact system present / kept.
    import traceback
    import weakref
    from epde.optimizers.moeadd.supplementary import cycle_safe_nds
    from epde.optimizers.nsga import operators as NOPS
    FATE = {}
    FATES = []
    ALL = {'members_seen': 0, 'wins': 0, 'truncations': 0, 'plane_errors': 0}
    COPRESENCE = open(os.path.join(HERE, PREFIX + '_copresence.log'), 'w')
    COPRESENCE.write('truncation | union N | per variable %s: law present, on its plane front, kept | exact present, '
                     'kept\n' % '/'.join(VARS))

    def fate_of(m):
        fate = FATE.get(id(m))
        if fate is not None and fate['ref']() is not m:
            fate = None
        return fate

    def carrier(cls):
        return 'law' if any(c in ('law', 'law+') for c in cls) else 'other'

    def copresence(T, population, kept):
        cls = [FATE[id(m)]['cls'] for m in population]
        cells = []
        for j, var in enumerate(VARS):
            e = T.keys.index(var)
            law = [i for i, c in enumerate(cls) if c[j] == 'law']
            cells.append('%d,%d,%d' % (len(law), int(any(T.front[e, i] for i in law)),
                                       sum(1 for i in law if id(population[i]) in kept)))
        exact = [i for i, c in enumerate(cls) if c == EXACT]
        COPRESENCE.write('%d | %d | %s | %d,%d\n' % (ALL['truncations'], len(population), ' | '.join(cells),
                                                     len(exact), sum(1 for i in exact if id(population[i]) in kept)))
        COPRESENCE.flush()

    real_truncate = NO.NSGAOptimizer._truncate

    def traced_truncate(self):
        population = list(self.pareto_levels.population)
        level_of = {id(m): k for k, lvl in enumerate(cycle_safe_nds(population)) for m in lvl}
        T = None
        try:
            T = arm_pv.plane_table(population)
        except Exception:
            ALL['plane_errors'] += 1
            if ALL['plane_errors'] == 1:
                COPRESENCE.write('plane_table failed:\n%s\n' % traceback.format_exc())
        for i, m in enumerate(population):
            if fate_of(m) is None:
                cls = classes(m)
                fate = FATE[id(m)] = dict(kind=carrier(cls), cls=cls, level=level_of.get(id(m), -1), survived=0,
                                          wins=0, alive=True, ref=weakref.ref(m), plane_front=None, law_slot=None)
                if T is not None:
                    fate['plane_front'] = bool(T.front_count[i] > 0)
                    fate['law_slot'] = any(c == 'law' and bool(T.front[T.keys.index(v), i])
                                           for v, c in zip(VARS, cls))
                FATES.append(fate)
                ALL['members_seen'] += 1
        real_truncate(self)
        kept = {id(m) for m in self.pareto_levels.population}
        ALL['truncations'] += 1
        for m in population:
            fate = fate_of(m)
            if fate['alive']:
                if id(m) in kept:
                    fate['survived'] += 1
                else:
                    fate['alive'] = False
        if T is not None:
            copresence(T, population, kept)
    NO.NSGAOptimizer._truncate = traced_truncate

    real_select = NOPS.NSGASelection.apply

    def traced_select(self, objective, arguments):
        before = {id(m): m.crossover_times() for m in objective.population}
        out = real_select(self, objective, arguments)
        for m in objective.population:
            won = m.crossover_times() - before.get(id(m), 0)
            if won > 0:
                ALL['wins'] += won
                fate = fate_of(m)
                if fate is not None:
                    fate['wins'] += won
        return out
    NOPS.NSGASelection.apply = traced_select

    def fate_report():
        rows = collections.defaultdict(list)
        for fate in FATES:
            rows[fate['kind']].append(fate)
        opt = OPT.get('self')
        with open(os.path.join(HERE, PREFIX + '_fate.log'), 'w') as out:
            out.write('%d members seen over %d truncations, %d tournament wins | mu %s | epochs run %s (requested %s)'
                      ' | plane_table errors %d\n' % (
                          ALL['members_seen'], ALL['truncations'], ALL['wins'], getattr(opt, '_mu', None),
                          OPT['epochs_run'], OPT.get('epochs_requested'), ALL['plane_errors']))
            for kind in ('law', 'other'):
                group = rows.get(kind, [])
                if not group:
                    continue
                levels = np.array([f['level'] for f in group])
                survived = np.array([f['survived'] for f in group])
                wins = np.array([f['wins'] for f in group])
                planed = [f for f in group if f['plane_front'] is not None]
                share = lambda key: ('%5.1f%%' % (100.0 * np.mean([f[key] for f in planed]))) if planed else '  n/a'
                out.write('%-6s n %6d | entry level median %4.1f, on level 0 %5.1f%% | truncations survived: '
                          'median %4.1f, mean %5.2f, none %5.1f%% | tournament wins: mean %5.2f, none %5.1f%% | '
                          'on a plane front %s | law slot on its plane front %s\n' % (
                              kind, len(group), np.median(levels), 100.0 * np.mean(levels == 0),
                              np.median(survived), survived.mean(), 100.0 * np.mean(survived == 0),
                              wins.mean(), 100.0 * np.mean(wins == 0), share('plane_front'), share('law_slot')))
        COPRESENCE.close()
    atexit.register(fate_report)

if os.environ.get('TRACE_FACTOR_MOVES'):
    # EXPERIMENT (not in EPDE), as lv_trace: one move in six edits one factor of a non-target term -- its power by
    # +-1, a random single factor added to a one-factor term, or one factor of a two-factor term dropped.
    from epde import _loop_stats
    from epde.operators.multiobjective import plain_moves as PM
    from epde.structure.main_structures import Term
    stock_mutation = PM.plain_mutation

    def _power_step(term):
        factor = term.structure[np.random.randint(len(term.structure))]
        for idx, description in factor.params_description.items():
            if description['name'] == 'power':
                lo, hi = description['bounds'][:2]
                new = factor.params[idx] + np.random.choice([-1.0, 1.0])
                if lo <= new <= hi:
                    params = factor.params.copy()
                    params[idx] = new
                    factor.params = params
                    return True
        return False

    def _add_factor(term):
        if len(term.structure) >= 2:
            return False
        extra = Term(term.pool, max_factors_in_term=1).structure[0]
        if extra.label in {f.label for f in term.structure}:
            return False
        term.structure = term.structure + [extra]
        return True

    def _drop_factor(term):
        if len(term.structure) < 2:
            return False
        victim = np.random.randint(len(term.structure))
        term.structure = [f for k, f in enumerate(term.structure) if k != victim]
        return True

    def factor_mutation(equation):
        if np.random.random() >= 1.0 / 6.0:
            return stock_mutation(equation)
        target = PM.gene_term(equation)
        features = [t for t in equation.structure if t is not target]
        if not features:
            return stock_mutation(equation)
        term = features[np.random.randint(len(features))]
        kind = ('power', 'add', 'drop')[np.random.randint(3)]
        original = [f for f in term.structure]
        saved = [(f, f.params.copy()) for f in original]
        changed = {'power': _power_step, 'add': _add_factor, 'drop': _drop_factor}[kind](term)
        term.resetSavedState()
        if changed and PM._duplicated(equation):
            changed = False
        if not changed:
            term.structure = original
            for f, params in saved:
                f.params = params
            term.resetSavedState()
            equation._invalidate_label_cache()
            return stock_mutation(equation)
        _loop_stats.record('PlainMutation.factor_' + kind, 1, 1)
        equation._invalidate_label_cache()
        equation.reset_state(reset_right_part=True)
        return equation
    PM.plain_mutation = factor_mutation

if os.environ.get('TRACE_EPOCHS'):
    from epde.interface import interface as II_EP
    real_search_init = II_EP.EpdeSearch.__init__

    @functools.wraps(real_search_init)
    def capped_search_init(self, *args, **kwargs):
        kwargs.setdefault('training_epochs', int(os.environ['TRACE_EPOCHS']))
        return real_search_init(self, *args, **kwargs)
    II_EP.EpdeSearch.__init__ = capped_search_init

SHADOW = {'rows': [], 'errors': 0, 'prev': None}
SHADOW_KEYS = ('A', 'P', 'Vp', 'Va')
SHADOW_PATIENCE = (5, 10, 20)
if os.environ.get('TRACE_STOP_SHADOW'):
    # As lv_trace (USER Oct 28 "test early stop as 2n obj nondom front", "do we really need archive"): EPDE's early
    # stop never fires; after every epoch the four structure-keyed fronts A (the archive's 2n-objective non-dominated
    # set, EPDE's NSGA-II rule), P (the population's 2n-objective front), Vp / Va (per-variable fronts of the
    # population / archive), each variable's exact law on the archive and in the population, and the exact system.
    import traceback as tb_ss
    from epde.interface import interface as II_SS
    from epde.optimizers.moeadd.supplementary import cycle_safe_nds as nds_ss
    JSONL = open(os.path.join(HERE, PREFIX + '_stop.jsonl'), 'w')

    def plane_front_key(members):
        if not members:
            return frozenset()
        T = arm_pv.plane_table(members)
        return frozenset((T.keys[e], T.labels[e][i]) for e in range(T.n_eq) for i in np.flatnonzero(T.lev[e] == 0))

    def shadow_record(opt, epoch_idx):
        archive = list(opt.reported_front())
        population = list(opt.pareto_levels.population)
        keys = {'A': frozenset(m.structure_labels for m in archive),
                'P': frozenset(m.structure_labels for m in (nds_ss(population)[0] if population else [])),
                'Vp': plane_front_key(population), 'Va': plane_front_key(archive)}
        prev = SHADOW['prev']
        cls_a, cls_p = [classes(m) for m in archive], [classes(m) for m in population]
        row = dict(epoch=int(epoch_idx) + 1, inserts=calls['n'], gen=GEN['n'],
                   wall=round(time.perf_counter() - START, 1), pop=len(population), archive=len(archive),
                   arch_law=[sum(c[j] == 'law' for c in cls_a) for j in range(len(VARS))],
                   pop_law=[sum(c[j] == 'law' for c in cls_p) for j in range(len(VARS))],
                   exact_arch=sum(c == EXACT for c in cls_a), exact_pop=sum(c == EXACT for c in cls_p),
                   va_minus_vp=len(keys['Va'] - keys['Vp']), vp_minus_va=len(keys['Vp'] - keys['Va']))
        for k in SHADOW_KEYS:
            old = prev[k] if prev is not None else None
            row[k] = dict(size=len(keys[k]), same=old is not None and keys[k] == old,
                          added=len(keys[k] - old) if old is not None else len(keys[k]),
                          removed=len(old - keys[k]) if old is not None else 0)
        SHADOW['rows'].append(row)
        SHADOW['prev'] = keys
        JSONL.write(json.dumps(row) + '\n')
        JSONL.flush()

    def shadow_stagnation(self, patience, user_callback=None):
        def callback(snapshot, epoch_idx):
            try:
                shadow_record(self.optimizer, epoch_idx)
            except Exception:
                SHADOW['errors'] += 1
                if SHADOW['errors'] == 1:
                    log.write('\nshadow_record failed (first error):\n%s\n' % tb_ss.format_exc())
                    log.flush()
            return False if user_callback is None else user_callback(snapshot, epoch_idx)
        return callback
    II_SS.EpdeSearch._stagnation_callback = shadow_stagnation

    # Nov 10 (robot rerun on EPDE's U default): plain NSGA-II resolves early_stopping_patience
    # 'auto' to None (every epoch runs, no stagnation callback), which silently dropped this shadow (0 epochs
    # recorded). Any int installs the callback; the shadow never stops (returns False), so every epoch still runs.
    _real_resolve_patience = II_SS.resolve_patience

    def shadow_resolve_patience(patience, rps_mode, engine):
        out = _real_resolve_patience(patience, rps_mode, engine)
        return 10 ** 9 if out is None else out
    II_SS.resolve_patience = shadow_resolve_patience

    real_report = NO.NSGAOptimizer._report

    def capturing_report(self):
        OPT['final_pop'] = list(self.pareto_levels.population)
        return real_report(self)
    NO.NSGAOptimizer._report = capturing_report


def first_insert(test):
    hits = [stamp['insert'] for cls, stamp in FIRST.items() if test(cls)]
    return min(hits) if hits else None


def stop_report():
    rows = SHADOW['rows']
    first = {'exact': first_insert(lambda c: c == EXACT)}
    for j, var in enumerate(VARS):
        first['law-' + var] = first_insert(lambda c, j=j: c[j] == 'law')
    with open(os.path.join(HERE, PREFIX + '_stop.log'), 'w') as out:
        out.write('system %s, variables %s | epochs recorded %d (run %s, requested %s) | mu %s | shadow errors %d\n' % (
            NAME, VARS, len(rows), OPT['epochs_run'], OPT.get('epochs_requested'), OPT.get('mu'), SHADOW['errors']))
        out.write('first evaluated (insert index): %s\n' % ' | '.join('%s %s' % kv for kv in first.items()))
        out.write('keys: A archive 2n-obj ND (EPDE) | P population 2n-obj ND | Vp population per-variable fronts | '
                  'Va archive per-variable fronts\n\n')
        out.write('epoch inserts   gen    wall  | ' + ' | '.join('%-14s' % k for k in SHADOW_KEYS)
                  + ' | archive law %s exact | pop law exact | Va-Vp Vp-Va\n' % '/'.join(VARS))
        for r in rows:
            cells = ['%-14s' % ('%d =' % r[k]['size'] if r[k]['same'] else '%d +%d-%d' % (
                r[k]['size'], r[k]['added'], r[k]['removed'])) for k in SHADOW_KEYS]
            out.write('%5d %7d %5d %7.1f | %s | %s %d | %s %d | %d %d\n' % (
                r['epoch'], r['inserts'], r['gen'], r['wall'], ' | '.join(cells), '/'.join(map(str, r['arch_law'])),
                r['exact_arch'], '/'.join(map(str, r['pop_law'])), r['exact_pop'], r['va_minus_vp'],
                r['vp_minus_va']))
        try:
            final_pop = OPT.get('final_pop') or []
            opt = OPT.get('self')
            union_l0 = list(opt.pareto_levels.levels[0]) if opt is not None and opt.pareto_levels.levels else []
            pop_l0 = nds_ss(final_pop)[0] if final_pop else []
            out.write('\nREPORT WITH vs WITHOUT THE ARCHIVE (system classes %s):\n' % '/'.join(VARS))
            for name, group in (('union L0 (EPDE: archive + last population)', union_l0),
                                ('last population L0 (no archive)', pop_l0),
                                ('last population, every member', final_pop)):
                counts = collections.Counter(classes(m) for m in group)
                out.write('  %-44s %4d systems | %s\n' % (name, len(group), ', '.join(
                    '%s %d' % ('/'.join(c), n) for c, n in counts.most_common())))
            out.write('  per-variable fronts: union %d, last population %d, union minus population %d\n' % (
                len(plane_front_key(union_l0)), len(plane_front_key(final_pop)),
                len(plane_front_key(union_l0) - plane_front_key(final_pop))))
        except Exception:
            out.write('\nREPORT WITH vs WITHOUT THE ARCHIVE failed:\n%s\n' % tb_ss.format_exc())
        out.write('\nSTOP RULES (EPDE semantics: still += 1 when the key equals the previous epoch\'s, stop once '
                  'still >= patience):\n')
        for k in SHADOW_KEYS:
            for p in SHADOW_PATIENCE:
                still, stop = 0, None
                for r in rows:
                    still = still + 1 if r[k]['same'] else 0
                    if still >= p:
                        stop = r
                        break
                if stop is None:
                    out.write('  %-2s p=%-2d never within %d epochs | exact %s\n' % (
                        k, p, len(rows), 'found' if first['exact'] is not None else 'never found'))
                    continue
                found = {name: (idx is not None and idx <= stop['inserts']) for name, idx in first.items()}
                out.write('  %-2s p=%-2d stops at epoch %3d (insert %7d, %7.1f s) | by then: %s | exact on the '
                          'archive %d, in the population %d%s\n' % (
                              k, p, stop['epoch'], stop['inserts'], stop['wall'],
                              ', '.join('%s %s' % (name, 'yes' if ok else 'no') for name, ok in found.items()),
                              stop['exact_arch'], stop['exact_pop'],
                              ' | LOST: exact found later, at insert %d' % first['exact']
                              if first['exact'] is not None and not found['exact'] else ''))


def planes_report():
    from epde.operators.multiobjective.moeadd_specific import equation_planes
    opt = OPT.get('self')
    population = list(opt.pareto_levels.population) if opt is not None else []
    with open(os.path.join(HERE, PREFIX + '_planes.log'), 'w') as out:
        if not population:
            out.write('no final population\n')
            return
        keys = list(population[0].vals.equation_keys)
        planes, labels = equation_planes(population)
        cls = [classes(m) for m in population]
        out.write('final union: %d systems, planes %s, mu %s, epochs run %s\n' % (
            len(population), keys, getattr(opt, '_mu', None), OPT['epochs_run']))
        for e, key in enumerate(keys):
            lev = arm_pv.plane_levels_fast(planes[e], labels[e])
            j = VARS.index(key)
            out.write('\nplane %s: level sizes %s\n' % (key, np.bincount(lev).tolist()))
            for i in np.flatnonzero(lev == 0):
                out.write('  L0 #%d system %s point [%s] class %s\n' % (
                    i, cls[i], ', '.join('%.3g' % x for x in planes[e][i]), cls[i][j]))
            law = [i for i in range(len(population)) if cls[i][j] == 'law']
            out.write('  law-%s equations: %d, plane levels %s (min %s)\n' % (
                key, len(law), sorted(collections.Counter(int(lev[i]) for i in law).items()),
                int(min(lev[i] for i in law)) if law else None))
        both = [i for i, c in enumerate(cls) if c == EXACT]
        out.write('\nexact systems in the union: %d\n' % len(both))


if PROV:
    PT.install_operators()          # after every arm and TRACE_FACTOR_MOVES: wraps the CURRENT class attributes

if os.environ.get('TRACE_RNG'):     # Nov 8 (wire U): the global random stream -> <PREFIX>_rng.log (rng_trace.py)
    import rng_trace
    rng_trace.install(PREFIX, HERE)

try:
    R.main()
finally:
    log.write('\nEVALUATED SYSTEM CLASSES %s: %d calls\n' % ('/'.join(VARS), calls['n']))
    for key, n in stats.most_common(40):
        log.write('  %6d  %s\n' % (n, key))
    log.write('\nEPOCHS RUN: %s (requested %s) | mu %s | wall %.1f s | optimize from %s s to %s s\n' % (
        OPT['epochs_run'], OPT.get('epochs_requested'), OPT.get('mu'), time.perf_counter() - START,
        OPT.get('optimize_start'), OPT.get('optimize_end')))
    log.write('FIRST EVALUATION PER CLASS (insert index | plain fit calls / versions | generation | wall s):\n')
    for cls, stamp in sorted(FIRST.items(), key=lambda item: item[1]['insert']):
        if 'law' in cls:
            log.write('  %-30s #%-7d | %s / %s | gen %d | %.1f s\n' % (
                cls, stamp['insert'], stamp['plain_calls'], stamp['plain_versions'], stamp['generation'],
                stamp['seconds']))
    log.write('NEW STRUCTURE LABELS evaluated: %s (over %d inserts)\n' % (
        ', '.join('%s %d' % (v, len(NEWLAB[v])) for v in VARS), calls['n']))
    try:
        planes_report()
    except Exception:
        import traceback
        log.write('\nplanes_report failed:\n%s\n' % traceback.format_exc())
    if os.environ.get('TRACE_STOP_SHADOW'):
        try:
            stop_report()
        except Exception:
            import traceback
            log.write('\nstop_report failed:\n%s\n' % traceback.format_exc())
    log.close()
