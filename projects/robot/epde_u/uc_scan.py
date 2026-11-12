"""Nov 12 (USER "Need a way to avoid unit-circle relation"): how the unit-circle relation enters the spin L0 fronts.
The heading data lie on c^2 + s^2 = 1 exactly, so a term space over (c, s, c', s', c'', s'') carries exact identities:
  g0 = c^2 + s^2 - 1,  g1 = g0'/2 = c c' + s s',  g2 = g1' = c'^2 + c c'' + s'^2 + s s''.
Candidate rules, judged on the existing fronts (Nov 10 U strong spin_, Nov 11 spinb U strong / UW weak, Nov 12
smokes):
  R_std  standard monomials: Groebner basis of <g0, g1, g2> (grevlex, s-variables above c-variables); a term whose
         heading monomial is divisible by a leading monomial is refused. Spans the same functions on the circle with no
         two equal columns -> no exact rewrite left, the law (dc/dt = -w s, ds/dt = w c, d2c = -w^2 c) untouched.
  R_xd   per-variable search: the other heading variable enters as a value token only (no derivative of it).
Per run and variable: L0 members, credited laws, members R_std would refuse, members R_xd would refuse, and the best
WAPE member with / without the refused ones. python uc_scan.py -> stdout + uc_scan.txt"""
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
GENS = (s2, s1, s, c2, c1, c)
G = sp.groebner([c**2 + s**2 - 1, c * c1 + s * s1, c1**2 + c * c2 + s1**2 + s * s2], *GENS, order='grevlex')
LMS = [sp.Poly(g, *GENS).monoms(order='grevlex')[0] for g in G.exprs]
SYM = {('c', 0): c, ('s', 0): s, ('c', 1): c1, ('s', 1): s1, ('c', 2): c2, ('s', 2): s2}
DER = re.compile(r'^d(?:\^(\d+))?([A-Za-z_]+)/dx0(?:\^(\d+))?(?:\^(\d+))?$')
OUT = []


def say(x=''):
    OUT.append(x)
    print(x, flush=True)


def factor(f):
    """(variable, derivative order, power) of a heading factor, else None."""
    m = DER.match(f)
    if m and m.group(2) in ('c', 's'):
        if m.group(1):
            return m.group(2), int(m.group(1)), int(m.group(4) or 1)
        return m.group(2), 1, int(m.group(3) or 1)
    name, _, p = f.partition('^')
    if name in ('c', 's'):
        return name, 0, int(p or 1)
    return None


def monom(term):
    e = dict.fromkeys(GENS, 0)
    for f in term.split('*'):
        h = factor(f)
        if h:
            e[SYM[h[:2]]] += h[2]
    return tuple(e[g] for g in GENS)


def reducible(term):
    m = monom(term)
    return any(all(a >= b for a, b in zip(m, lm)) for lm in LMS)


def cross_deriv(term, var):
    other = 's' if var == 'c' else 'c'
    return any((h := factor(f)) and h[0] == other and h[1] > 0 for f in term.split('*'))


say('Groebner basis (grevlex %s):' % (GENS,))
for g in G.exprs:
    say('   %s   leading monomial %s' % (g, sp.Poly(g, *GENS).LM(order='grevlex')))
PAT = re.compile(r'^auto_(?P<sys>spinb?_.+)_trace_(?P<arm>[A-Z0-9]+)_(?P<tag>e\d+|smoke)\.json$')
tot = collections.defaultdict(collections.Counter)
for path in sorted(glob.glob(os.path.join(HERE, 'auto_spin*_trace_*.json'))):
    m = PAT.match(os.path.basename(path))
    if not m:
        continue
    rep = json.load(open(path))
    l0 = rep['levels'][0] if rep['levels'] else []
    if not l0:
        continue
    run = dict(sys=m['sys'], l0=l0, vars=list(l0[0]['marks']))
    sysname = m['sys'] if m['sys'].startswith('spin_') else 'spin_' + m['sys'][len('spinb_'):]
    arm = '%s %s' % (m['sys'].split('_')[0], m['arm'])
    say('=' * 110)
    say('%s  arm %s %s' % (m['sys'], m['arm'], m['tag']))
    for var in run['vars']:
        pts = RU.plane_points(run, var)
        rows = []
        for key, d, i, _row in pts:
            terms = [t for t in key[1] if t != '1']
            cr = RU.credit(sysname, var, key)
            rows.append(dict(terms=terms, d=d, i=i, law=cr is not None, red=any(reducible(t) for t in terms),
                             xd=any(cross_deriv(t, var) for t in terms), target=key[0]))
        n = len(rows)
        nl = sum(r['law'] for r in rows)
        nr = sum(r['red'] for r in rows)
        nx = sum(r['xd'] for r in rows)
        nb = sum(r['red'] or r['xd'] for r in rows)
        best = min(rows, key=lambda r: r['d']) if rows else None
        keep_std = [r for r in rows if not r['red']]
        keep_xd = [r for r in rows if not r['xd']]
        say('  %s: L0 %2d | law %2d | R_std refuses %2d | R_xd refuses %2d | either %2d | law refused by R_std %d' % (
            var, n, nl, nr, nx, nb, sum(r['law'] and r['red'] for r in rows)))
        for lab, sub in (('all', rows), ('R_std', keep_std), ('R_xd', keep_xd)):
            if sub:
                b = min(sub, key=lambda r: r['d'])
                say('      best WAPE %-5s %.4f %s| %s = %s' % (lab, b['d'], '[law] ' if b['law'] else '', b['target'],
                                                             ' + '.join(b['terms'])))
        for k, v in (('L0', n), ('law', nl), ('red', nr), ('xd', nx), ('runs', 1),
                     ('best_law', bool(best and best['law'])),
                     ('best_law_std', bool(keep_std and min(keep_std, key=lambda r: r['d'])['law'])),
                     ('best_law_xd', bool(keep_xd and min(keep_xd, key=lambda r: r['d'])['law']))):
            tot[arm][k] += v
say('=' * 110)
say('TOTAL per arm over variables: L0 members, laws, refused by R_std / R_xd; best-WAPE member is a law (all / after '
    'R_std / after R_xd)')
for arm, t in sorted(tot.items()):
    say('  %-10s vars %3d | L0 %4d law %3d | R_std %4d (%.0f%%) R_xd %4d (%.0f%%) | best=law %d / %d / %d' % (
        arm, t['runs'], t['L0'], t['law'], t['red'], 100 * t['red'] / max(t['L0'], 1), t['xd'],
        100 * t['xd'] / max(t['L0'], 1), t['best_law'], t['best_law_std'], t['best_law_xd']))
open(os.path.join(HERE, 'uc_scan.txt'), 'w', encoding='utf-8').write('\n'.join(OUT) + '\n')
