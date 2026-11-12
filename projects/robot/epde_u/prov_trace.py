"""Nov 6 2026 (improvement plan I5b): provenance + per-insert archive log, switched on by TRACE_PROV=1 in sys_trace.py /
lv_trace.py. LOGGING ONLY: no RNG is read or drawn, no search object is changed except the plain attributes
`_trace_ops` / `_trace_prov` on Equation objects (__dict__ attributes: Equation.__deepcopy__ copies __slots__ only, so
a marker lives on the one object an operator touched and a copy never inherits it).

Two files beside the other trace outputs (the existing files keep their format; TRACE_PROV unset = this module is not
even imported):

  <PREFIX>_inserts.jsonl   one JSON object per NSGAOptimizer._archive_insert call, in call order (the stream.tsv row
                           with the same 'i'):
      i      insert index (= stream.tsv column 'insert', = calls['n'])
      ep     epochs completed when the insert happened (0-based epoch index of an insert: ep; epoch ep+1 is the epoch
             whose end it belongs to). gen = truncations completed (stream's generation). ph = 'init' before the first
             mutation / crossover call (the initial population's inserts), else 'evo'.
      ids    per slot the equation id (the eqs.jsonl id of <PREFIX>_eqs.jsonl; -1: no target)
      lb     per slot an interned id of the equation's REAL structure_label (what check_dominance compares:
             (terms_labels, has_constant)); equal lb = the same equation structure, whatever the target
      ob     the objective vector as _archive_insert saw it (obj_fun; slot j owns columns j, j+n, j+2n, ...:
             here [wape_1..wape_n, dyad_1..dyad_n])
      res    1 ENTERED, 0 REJECTED
      why    (rejected only) 'twin' = a member with the same equations_labels was not beaten by value, 'dom' = at
             least one archived member dominates it, 'dom?' = rejected with no dominator found by this log (should not
             happen)
      nb     (rejected only) how many archive members dominate it; blk = the insert indices of the first three
      ev     (entered only, only if it evicted) one object per evicted member:
               i     the evicted member's own insert index        ids / lb / ob   as above
               tw    1 when it is the TWIN replaced (same equations_labels), 0 when dominated
               cmp   per objective column, the NEW system vs this member: '<' better, '=' tie, '>' worse, 'S' the
                     slot holds the same structure in both (skipped by check_dominance)
               slots per slot a letter: 'S' same structure, '+' better in every objective of the slot, '=' tied in
                     every objective, '-' worse in some, 'm' mixed (better and worse)
      arch   archive size after the insert
  <PREFIX>_prov.jsonl     one JSON object per NEW equation id (first appearance in the stream), at its first insert:
      id, var, i (insert index), ep, ph, prov, ops
      prov   the operator that created it: `eq._trace_prov` when an arm set one (e.g. 'be'), else the chain of recorded
             operators ('>' joined, e.g. 'crossover>swap'), else 'initial' (initial population) or 'other'
      ops    the recorded chain as a list: the stock/arm plain move kinds (swap, add, drop, target, nudge, fill,
             factor_power, factor_add, factor_drop, read from loop_stats' 'PlainMutation.<kind>' record that
             plain_mutation and every arm copy of it emit) and 'crossover' (EquationCrossover.apply returning new
             objects). 'fill' = arm_pvfpx / arm_pvfp / arm_pvfpt's fill-and-prune (they record 'PlainMutation.fill').
"""
import atexit
import functools
import json
import os

import numpy as np

STATE = {'on': False, 'errors': 0, 'first_error': None, 'evo': False, 'hist0': None, 'hist_seen': 0}
ARCH = {}                  # id(system) -> record of an archive member (alive while archived: removed when evicted)
_STACK = {'dirty': True, 'recs': [], 'obj': None, 'lab': None}
LAB = {}                   # real structure_label -> small int
CUR = {'kind': None, 'changed': 0}
COUNT = {'inserts': 0, 'entered': 0, 'rejected': 0, 'evictions': 0, 'new_ids': 0, 'prov': {}}


def setup(prefix, here, variables):
    STATE.update(on=True, vars=list(variables), n=len(variables), log_path=os.path.join(here, prefix + '_prov.log'),
                 ins=open(os.path.join(here, prefix + '_inserts.jsonl'), 'w'),
                 prov=open(os.path.join(here, prefix + '_prov.jsonl'), 'w'))
    atexit.register(close)


def close():
    for key in ('ins', 'prov'):
        fh = STATE.get(key)
        if fh is not None and not fh.closed:
            fh.close()
    path = STATE.get('log_path')
    if path and not STATE.get('log_written'):
        STATE['log_written'] = True
        with open(path, 'w') as out:
            out.write('\n'.join(report_lines()) + '\n')


def _err(exc):
    import traceback
    STATE['errors'] += 1
    if STATE['first_error'] is None:
        STATE['first_error'] = traceback.format_exc()


def _lab(label):
    return LAB.setdefault(label, len(LAB))


def _stack():
    if _STACK['dirty']:
        recs = list(ARCH.values())
        _STACK['recs'] = recs
        if recs:
            _STACK['obj'] = np.vstack([r['ob'] for r in recs])
            _STACK['lab'] = np.vstack([r['lbv'] for r in recs])
        else:
            _STACK['obj'] = _STACK['lab'] = None
        _STACK['dirty'] = False
    return _STACK


def begin(opt, system, ids, n_before, insert_idx, gen):
    """Called before the real _archive_insert (after stream_write): what only the pre-insert state can give."""
    try:
        if STATE['hist0'] is None:
            STATE['hist0'] = len(opt._pareto_history)
        labs = [_lab(system.vals[v].structure_label) for v in STATE['vars']]
        return dict(i=insert_idx, ids=[int(x) for x in ids], n0=n_before, lb=labs, gen=gen,
                    ep=len(opt._pareto_history) - STATE['hist0'], ph='evo' if STATE['evo'] else 'init')
    except Exception as exc:  # noqa: BLE001
        _err(exc)
        return None


def _slot_letters(cmp_cols, n):
    out = []
    for j in range(n):
        cols = cmp_cols[j::n]
        if all(c == 'S' for c in cols):
            out.append('S')
        elif all(c == '<' for c in cols):
            out.append('+')
        elif all(c == '=' for c in cols):
            out.append('=')
        elif any(c == '<' for c in cols) and any(c == '>' for c in cols):
            out.append('m')
        elif any(c == '>' for c in cols):
            out.append('-')
        else:                       # a mix of ties and improvements
            out.append('+')
    return ''.join(out)


def end(opt, system, tok, entered):
    """Called after the real _archive_insert returned ``entered``."""
    if tok is None:
        return
    try:
        n = STATE['n']
        ob = np.asarray(system.obj_fun, dtype=float)
        m = len(ob)
        lbv = np.array(tok['lb'], dtype=np.int64)[np.arange(m) % n]
        rec = dict(i=tok['i'], ids=tok['ids'], lb=tok['lb'], ob=ob, lbv=lbv)
        COUNT['inserts'] += 1
        row = dict(i=tok['i'], ep=tok['ep'], gen=tok['gen'], ph=tok['ph'], ids=tok['ids'], lb=tok['lb'],
                   ob=ob.tolist(), res=1 if entered else 0)
        alive = {id(x) for x in opt._archive}
        gone = [k for k in ARCH if k not in alive]
        if entered:
            COUNT['entered'] += 1
            evs = []
            for k in gone:
                old = ARCH[k]
                cmp_cols = []
                for o in range(m):
                    if old['lbv'][o] == lbv[o]:
                        cmp_cols.append('S')
                    elif ob[o] < old['ob'][o]:
                        cmp_cols.append('<')
                    elif ob[o] > old['ob'][o]:
                        cmp_cols.append('>')
                    else:
                        cmp_cols.append('=')
                evs.append(dict(i=old['i'], ids=old['ids'], lb=old['lb'], ob=old['ob'].tolist(),
                                tw=1 if old['lb'] == tok['lb'] else 0, cmp=''.join(cmp_cols),
                                slots=_slot_letters(cmp_cols, n)))
            if evs:
                COUNT['evictions'] += len(evs)
                row['ev'] = evs
            for k in gone:
                del ARCH[k]
            ARCH[id(system)] = rec
            _STACK['dirty'] = True
        else:
            COUNT['rejected'] += 1
            for k in gone:                       # cannot happen (a rejection changes nothing); keep the table true
                del ARCH[k]
                _STACK['dirty'] = True
            st = _stack()
            if st['obj'] is None:
                row['why'], row['nb'] = 'dom?', 0
            else:
                differ = st['lab'] != lbv
                dominating = (np.any((st['obj'] < ob) & differ, axis=1) & ~np.any((st['obj'] > ob) & differ, axis=1))
                twins = np.all(st['lab'] == lbv, axis=1)
                nb = int(dominating.sum())
                row['nb'] = nb
                if twins.any():
                    first = np.flatnonzero(twins)[0]
                    old = st['obj'][first]
                    beats = bool(len(old) == len(ob) and np.all(ob <= old) and np.any(ob < old))
                    row['why'] = 'twin' if not beats else ('dom' if nb else 'dom?')
                else:
                    row['why'] = 'dom' if nb else 'dom?'
                if nb:
                    row['blk'] = [st['recs'][int(k)]['i'] for k in np.flatnonzero(dominating)[:3]]
        row['arch'] = len(alive)
        STATE['ins'].write(json.dumps(row) + '\n')
        if tok['i'] % 2000 == 0:
            STATE['ins'].flush()
            STATE['prov'].flush()
        for j, idx in enumerate(tok['ids']):
            if idx >= tok['n0']:
                if idx < 0:
                    continue
                eq = system.vals[STATE['vars'][j]]
                ops = list(getattr(eq, '_trace_ops', None) or [])
                prov = getattr(eq, '_trace_prov', None) or ('>'.join(ops) if ops else None)
                if prov is None:
                    prov = 'initial' if tok['ph'] == 'init' else 'other'
                COUNT['new_ids'] += 1
                COUNT['prov'][prov] = COUNT['prov'].get(prov, 0) + 1
                STATE['prov'].write(json.dumps(dict(id=idx, var=STATE['vars'][j], i=tok['i'], ep=tok['ep'],
                                                    ph=tok['ph'], prov=prov, ops=ops)) + '\n')
        for v in STATE['vars']:                  # the chain ends with the insert (see install_operators)
            d = getattr(system.vals[v], '__dict__', None)
            if d:
                d.pop('_trace_ops', None)
                d.pop('_trace_prov', None)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


def _add_op(eq, op):
    try:
        ops = getattr(eq, '_trace_ops', None)
        eq._trace_ops = (list(ops) if ops else []) + [op]
    except Exception as exc:  # noqa: BLE001
        _err(exc)


def install_operators():
    """Wrap (after every arm is installed, so the CURRENT class attributes are wrapped) EquationMutation.apply and
    EquationCrossover.apply, and loop_stats.record (read-only tap on the 'PlainMutation.<kind>' records)."""
    from epde import _loop_stats
    from epde.operators.multiobjective import mutations as MUT
    from epde.operators.multiobjective import variation as VAR

    real_record = _loop_stats.record

    @functools.wraps(real_record)
    def tapped_record(site, iters, cap):
        if isinstance(site, str) and site.startswith('PlainMutation.'):
            CUR['kind'], CUR['changed'] = site[len('PlainMutation.'):], int(iters)
        return real_record(site, iters, cap)
    _loop_stats.record = tapped_record

    real_mutation = MUT.EquationMutation.apply

    @functools.wraps(real_mutation)
    def mutation_apply(self, objective, arguments):
        STATE['evo'] = True
        CUR['kind'], CUR['changed'] = None, 0
        out = real_mutation(self, objective, arguments)
        if CUR['kind'] is not None and CUR['changed']:
            _add_op(out if out is not None else objective, CUR['kind'])
        CUR['kind'], CUR['changed'] = None, 0
        return out
    MUT.EquationMutation.apply = mutation_apply

    real_crossover = VAR.EquationCrossover.apply

    @functools.wraps(real_crossover)
    def crossover_apply(self, objective, arguments):
        STATE['evo'] = True
        out = real_crossover(self, objective, arguments)
        try:
            for new, old in zip(out, objective):
                if new is not old:
                    _add_op(new, 'crossover')
        except Exception as exc:  # noqa: BLE001
            _err(exc)
        return out
    VAR.EquationCrossover.apply = crossover_apply

    # plain_fit.apply_plain fits `deepcopy(equation)` per version and puts the copy back as the gene, and
    # Equation.__deepcopy__ copies __slots__ only: a marker set by a move would be lost before the insert. This wrapper
    # carries the two marker attributes over to the copy (attributes only); end() clears them from every equation of an
    # inserted system, so a parent that stays in the population never hands a stale chain to its copies.
    from epde.structure.main_structures import Equation
    real_deepcopy = Equation.__dict__['__deepcopy__']

    @functools.wraps(real_deepcopy)
    def deepcopy_keeping_markers(self, memo=None):
        new = real_deepcopy(self, memo)
        d = getattr(self, '__dict__', None)
        if d:
            if '_trace_ops' in d:
                new._trace_ops = list(d['_trace_ops'])
            if '_trace_prov' in d:
                new._trace_prov = d['_trace_prov']
        return new
    Equation.__deepcopy__ = deepcopy_keeping_markers
    STATE['installed'] = True


def report_lines():
    lines = ['TRACE_PROV: %d inserts logged (%d entered, %d rejected, %d evictions), %d new equation ids, '
             'errors %d' % (COUNT['inserts'], COUNT['entered'], COUNT['rejected'], COUNT['evictions'],
                            COUNT['new_ids'], STATE['errors']),
             'TRACE_PROV provenance of new equation ids: %s' % json.dumps(dict(sorted(COUNT['prov'].items())))]
    if STATE['first_error']:
        lines.append('TRACE_PROV first error:\n' + STATE['first_error'])
    return lines
