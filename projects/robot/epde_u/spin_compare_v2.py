"""Nov 12 (USER "Need a way to avoid unit-circle relation"): spin arms side by side, seed 0, 20 epochs.
  U     Nov 10 strong, pre-fix code (porcelain e4a6)
  U2    strong, snapshot 9bd360e            UW2   weak, same
  UI2   strong + invariant_terms='refuse'   UWI2  weak, same (snapshot 9bd360e_inv = 9bd360e + the rule only)
  V2    per-variable spin (spinvc_ / spinvs_: the other component a value-only data token), strong; VW2 weak
Per robot and variable, on the FINAL L0: credited reference forms any(strict) (robot_systems.truth), L0 members,
members holding a term the unit circle makes redundant (the Groebner leading monomials EPDE's rule builds:
generators s'' > c'' > s' > c' > s > c), and whether the best-WAPE member is a credited law. FULL = both variables
credited (spin_: in one L0 system; V2/VW2: each in its own search). WAPE is compared only within one form.
python spin_compare_v2.py -> stdout + spin_compare_v2.txt"""
import collections
import glob
import json
import os
import re
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import results_u as RU  # noqa: E402

c, s, c1, s1, c2, s2 = sp.symbols('c s c1 s1 c2 s2')
GENS = (s2, c2, s1, c1, s, c)
G = sp.groebner([c ** 2 + s ** 2 - 1, c * c1 + s * s1, c1 ** 2 + c * c2 + s1 ** 2 + s * s2], *GENS, order='grevlex')
LMS = [sp.Poly(g, *GENS).monoms(order='grevlex')[0] for g in G.exprs]
SYM = {('c', 0): c, ('s', 0): s, ('c', 1): c1, ('s', 1): s1, ('c', 2): c2, ('s', 2): s2}
DER = re.compile(r'^d(?:\^(\d+))?([A-Za-z_]+)/dx0(?:\^(\d+))?(?:\^(\d+))?$')
ARMS = [('U', 'spin', 'U', 'strong'), ('U2', 'spin', 'U2', 'strong'), ('UI2', 'spin', 'UI2', 'strong'),
        ('V2', 'spinv', 'U2', 'strong'), ('UW2', 'spin', 'UW2', 'weak'), ('UWI2', 'spin', 'UWI2', 'weak'),
        ('VW2', 'spinv', 'UW2', 'weak')]
ROBOTS = ['N30_PWM10_D41_12', 'N30_PWM10_D41_42', 'N40_PWM10_D41_17', 'N40_PWM10_D41_20', 'N40_PWM20_D41_66',
          'N1_PWM10_1']
OUT = []


def say(x=''):
    OUT.append(x)
    print(x, flush=True)


def factor(f):
    m = DER.match(f)
    if m and m.group(2) in ('c', 's'):
        if m.group(1):
            return m.group(2), int(m.group(1)), int(m.group(4) or 1)
        return m.group(2), 1, int(m.group(3) or 1)
    name, _, p = f.partition('^')
    if name in ('c', 's'):
        return name, 0, int(p or 1)
    return None


def identity(term):
    e = dict.fromkeys(GENS, 0)
    for f in term.split('*'):
        h = factor(f)
        if h:
            e[SYM[h[:2]]] += h[2]
    m = tuple(e[g] for g in GENS)
    return any(all(a >= b for a, b in zip(m, lm)) for lm in LMS)


def load(system, arm):
    path = os.path.join(HERE, 'auto_%s_trace_%s_e20.json' % (system, arm))
    if not os.path.exists(path):
        return None
    rep = json.load(open(path))
    l0 = rep['levels'][0] if rep['levels'] else []
    prefix = '%s_trace_%s_e20' % (system, arm)
    side = path[:-5] + '_form.json'
    side = json.load(open(side)) if os.path.exists(side) else {}
    return dict(sys=system, l0=l0, vars=list(l0[0]['marks']) if l0 else [], prefix=prefix,
                head=(side.get('epde_digest_start') or {}).get('head', '')[:7], eq=side.get('digest_equal'),
                wall=RU.runs_tsv(HERE).get(prefix, {}).get('wall'))


def per_var(run, var, truth_sys):
    rows = []
    for key, d, i, _row in RU.plane_points(run, var):
        terms = [t for t in key[1] if t != '1']
        cr = RU.credit(truth_sys, var, key)
        rows.append(dict(d=d, law=cr, idn=any(identity(t) for t in terms), key=key))
    best = min(rows, key=lambda r: r['d']) if rows else None
    return dict(n=len(rows), law_any=sum(1 for r in rows if r['law']),
                law_strict=sum(1 for r in rows if r['law'] and r['law'][0] == 'law'),
                idn=sum(r['idn'] for r in rows), best_law=bool(best and best['law']), best=best)


tot = collections.defaultdict(collections.Counter)
for rb in ROBOTS:
    say('=' * 118)
    say(rb)
    for label, kind, arm, form in ARMS:
        cells, held = [], {}
        heads = set()
        for var in ('c', 's'):
            system = ('spin_' + rb) if kind == 'spin' else ('spinv%s_%s' % (var, rb))
            run = load(system, arm)
            if run is None or var not in run['vars']:
                cells.append('%s (not run)' % var)
                continue
            heads.add('%s%s' % (run['head'], '' if run['eq'] in (True, None) else '!'))
            pv = per_var(run, var, system)
            held[var] = pv['law_any'] > 0
            b = pv['best']
            cells.append('%s %d(%d) L0 %2d id %2d best %s %.4f' % (
                var, pv['law_any'], pv['law_strict'], pv['n'], pv['idn'], 'LAW ' if pv['best_law'] else 'other',
                b['d'] if b else float('nan')))
            t = tot[label]
            t['vars'] += 1
            t['held'] += pv['law_any'] > 0
            t['held_strict'] += pv['law_strict'] > 0
            t['L0'] += pv['n']
            t['idn'] += pv['idn']
            t['best_law'] += pv['best_law']
        if kind == 'spin':
            run = load('spin_' + rb, arm)
            full = 0
            if run is not None:
                _per, _fs, fa = RU.held(run)
                full = int(fa > 0)
        else:
            full = int(len(held) == 2 and all(held.values()))
        if held:
            tot[label]['robots'] += 1
            tot[label]['full'] += full
        say('  %-5s %-6s FULL %d | %s | %s' % (label, form, full, ' | '.join(cells), ','.join(sorted(heads))))
say('=' * 118)
say('TOTAL: robots with FULL / variables held any(strict) / L0 members, of them holding a unit-circle-redundant term '
    '/ variables whose best-WAPE member is a law')
for label, kind, arm, form in ARMS:
    t = tot[label]
    if t['vars']:
        say('  %-5s %-6s robots %d FULL %d | vars held %d(%d)/%d | L0 %d, identity %d (%.0f%%) | best=law %d/%d' % (
            label, form, t['robots'], t['full'], t['held'], t['held_strict'], t['vars'], t['L0'], t['idn'],
            100.0 * t['idn'] / max(t['L0'], 1), t['best_law'], t['vars']))
open(os.path.join(HERE, 'spin_compare_v2.txt'), 'w', encoding='utf-8').write('\n'.join(OUT) + '\n')
