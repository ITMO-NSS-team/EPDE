"""Nov 10 robot rerun: EPDE's current default U (this directory) vs the Nov 2 arms PV / PVFP / PVFPT (ROBOT_ARMS_DIR,
an older run folder), equal epochs, seed 0, same systems and search space.

Per spin / drift system and arm:
  * held (any credit) on the FINAL first level (auto_*.json L0): per variable, an L0 equation that is a reference form
    ('law': robot_systems.truth, identity forms such as the unit circle included) or a reference form plus only
    even-order derivative absorbers ('abs', USER Nov 2 error model); FULL = one L0 system with every variable credited.
    Strict = 'law' only.
  * first epoch a reference form of each variable sat on the archive, first epoch of the exact system (_stop.jsonl)
  * per variable, the union plane front of all arms (all targets, and plain target only = d<v>/dx0)
  * drift: token groups of the plain-target L0 equations and the best four by WAPE
Fields (fieldv = space orders 4, fieldv2 = orders 2): per field and recording, credits, first law epoch, the L0 front.
python results_u.py -> results_u.txt / results_u.json
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Folder with the Nov 2 arm runs (PV / PVFP / PVFPT); optional, set ROBOT_ARMS_DIR to compare against them.
OLD = os.environ.get('ROBOT_ARMS_DIR') or os.path.join(HERE, '_no_old_arms')
sys.path.insert(0, HERE)
ARMS = ['U', 'PV', 'PVFP', 'PVFPT']
PAT = re.compile(r'^auto_(?P<sys>.+)_trace_(?P<arm>U|PV|PVFP|PVFPT)_e(?P<ep>\d+)\.json$')


def runs_tsv(d):
    out = {}
    path = os.path.join(d, 'runs.tsv')
    if os.path.exists(path):
        for line in open(path, encoding='utf-8-sig'):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 4:
                out[f[0]] = dict(exit=f[1], wall=float(f[2].split()[1]), cpu=float(f[3].split()[1]))
    return out


def stop_rows(d, prefix):
    path = os.path.join(d, prefix + '_stop.jsonl')
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def first_epochs(rows, variables):
    first = {v: None for v in variables}
    full = None
    for r in rows:
        for j, v in enumerate(variables):
            if first[v] is None and r['arch_law'][j] > 0:
                first[v] = r['epoch']
        if full is None and r['exact_arch'] > 0:
            full = r['epoch']
    return first, full


def nondominated(points):
    keep = []
    for i, (_, a, b) in enumerate(points):
        if not any((a2 <= a and b2 <= b) and (a2 < a or b2 < b) for _, a2, b2 in points):
            keep.append(i)
    return keep


def plain_target(var):
    return 'd%s/dx0' % var


TOKEN_GROUPS = [('push', 'push_'), ('wall', 'wall_'), ('np', 'np'), ('vb', 'vb_'), ('mem', 'm'), ('contact', 'n_contact'),
                ('pos', 'pos_'), ('t', 't'), ('trig', ('sin', 'cos')), ('heading', ('c', 's'))]


def token_groups(terms):
    found = set()
    for term in terms:
        for factor in term.split('*'):
            name = factor.split('^')[0].split('~')[0].split('@')[0]
            for g, pre in TOKEN_GROUPS:
                pres = pre if isinstance(pre, tuple) else (pre,)
                if g == 'heading':
                    if name in pres:
                        found.add(g)
                elif g == 'mem':
                    if re.fullmatch(r'm\d_[xy]', name):
                        found.add(g)
                elif g == 't':
                    if name == 't':
                        found.add(g)
                elif g == 'np':
                    if name.startswith('np_'):
                        found.add(g)
                elif any(name.startswith(p) for p in pres):
                    found.add(g)
    return found


def load_runs():
    runs = []
    for d in (HERE, OLD):
        tsv = runs_tsv(d)
        for path in sorted(glob.glob(os.path.join(d, 'auto_*_trace_*.json'))):
            m = PAT.match(os.path.basename(path))
            if not m or (d == OLD and m['arm'] == 'U') or (d == HERE and m['arm'] != 'U'):
                continue
            rep = json.load(open(path))
            prefix = '%s_trace_%s_e%s' % (m['sys'], m['arm'], m['ep'])
            l0 = rep['levels'][0] if rep['levels'] else []
            variables = list(l0[0]['marks']) if l0 else []
            rows = stop_rows(d, prefix)
            first, full = first_epochs(rows, variables)
            runs.append(dict(sys=m['sys'], arm=m['arm'], epochs=int(m['ep']), prefix=prefix, rep=rep, l0=l0,
                             vars=variables, first=first, full=full, epochs_run=len(rows), tsv=tsv.get(prefix, {}),
                             pop=rep['settings'].get('population'), last=rows[-1] if rows else {}))
    return runs


def plane_points(run, var):
    """[(structure key, disc, instab, row)] of variable ``var`` over the run's L0 (distinct structures)."""
    pts, seen = [], set()
    for row in run['l0']:
        if not row.get('obj') or not row.get('keys') or row['keys'].get(var) is None:
            continue
        order = row.get('obj_order') or run['vars']
        n = len(order)
        j = order.index(var)
        k = row['keys'][var]
        key = (k['target'], tuple(k['terms']))
        if key in seen:
            continue
        seen.add(key)
        pts.append((key, row['obj'][j], row['obj'][n + j], row))
    return pts


def kind(system):
    return system.split('_')[0]


_DERIV = re.compile(r'^d\^(\d+)[A-Za-z_]+/dx\d\^\1')


def _nk(s):
    return '*'.join(sorted(x.strip() for x in s.split('*')))


def is_absorber(term):
    for factor in term.split('*'):
        m = _DERIV.match(factor)
        if m and int(m.group(1)) % 2 == 0:
            return True
    return False


_REF = {}


def reference_sets(system):
    if system not in _REF:
        import robot_systems as RS
        _REF[system] = {v: [(k, _nk(t), frozenset(_nk(x) for x in keys if x != '1')) for k, t, keys in forms]
                        for v, forms in RS.truth(system).items()}
    return _REF[system]


def credit(system, var, key):
    """('law', kind) if (target, terms) is exactly a reference form (constant ignored), ('abs', kind) if a reference
    form plus only even-order derivative absorbers, else None."""
    target, terms = key
    terms = frozenset(terms) - {'1'}
    for k, t, ref in reference_sets(system)[var]:
        if target == t and terms == ref:
            return 'law', k
    for k, t, ref in reference_sets(system)[var]:
        if target == t and ref < terms and all(is_absorber(x) for x in terms - ref):
            return 'abs', k
    return None


def row_credit(system, row, var):
    k = (row.get('keys') or {}).get(var)
    if not k:
        return None
    return credit(system, var, (k['target'], tuple(k['terms'])))


def held(run):
    """per variable: (strict, any) credit on L0, and FULL systems (strict, any)."""
    per = {}
    full_strict = full_any = 0
    for v in run['vars']:
        cs = [row_credit(run['sys'], row, v) for row in run['l0']]
        per[v] = dict(strict=sum(1 for c in cs if c and c[0] == 'law'), any=sum(1 for c in cs if c),
                      kinds=sorted({'%s%s' % (c[1], '' if c[0] == 'law' else '+abs') for c in cs if c}))
    for row in run['l0']:
        cs = [row_credit(run['sys'], row, v) for v in run['vars']]
        if all(cs):
            full_any += 1
            if all(c[0] == 'law' for c in cs):
                full_strict += 1
    return per, full_strict, full_any


def spin_drift_section(by_sys, lines, out):
    for system in sorted(by_sys, key=lambda s: (['spin', 'driftx', 'drifty'].index(kind(s)), s)):
        if kind(system) not in ('spin', 'driftx', 'drifty'):
            continue
        arms = by_sys[system]
        variables = next(iter(arms.values()))['vars']
        lines.append('=' * 110)
        lines.append('%s   variables %s' % (system, variables))
        sysout = out[system] = {}
        union = {}
        for var in variables:
            for plain in (False, True):
                pts = []
                for a, r in arms.items():
                    for key, d, i, _row in plane_points(r, var):
                        if plain and key[0] != plain_target(var):
                            continue
                        pts.append(((a,) + key, d, i))
                nd = nondominated([(k, d, i) for k, d, i in pts])
                union[(var, plain)] = (len(nd), dict(collections.Counter(pts[n][0][0] for n in nd)), [pts[n] for n in nd])
        for a in ARMS:
            r = arms.get(a)
            if r is None:
                lines.append('  %-6s (not run)' % a)
                continue
            per, fs, fa = held(r)
            wall = r['tsv'].get('wall')
            lines.append('  %-6s ep %2d/%d pop %s | L0 %3d | HELD any: %s; FULL sys any %d strict %d | first law-epoch %s, '
                         'exact %s | wall %s s' % (
                             a, r['epochs_run'], r['epochs'], r['pop'], len(r['l0']),
                             ' '.join('%s %d(%d)' % (v, per[v]['any'], per[v]['strict']) for v in variables), fa, fs,
                             ' '.join('%s:%s' % (v, r['first'][v]) for v in variables), r['full'],
                             '%.0f' % wall if wall else '?'))
            kinds = '; '.join('%s: %s' % (v, ', '.join(per[v]['kinds'])) for v in variables if per[v]['kinds'])
            if kinds:
                lines.append('         credited kinds on L0: %s' % kinds)
            for var in variables:
                n_all, share_all, _ = union[(var, False)]
                n_pl, share_pl, _ = union[(var, True)]
                lines.append('         %s union front: all targets %d/%d, plain target %d/%d' % (
                    var, share_all.get(a, 0), n_all, share_pl.get(a, 0), n_pl))
            targets = collections.Counter()
            for var in variables:
                for key, d, i, _row in plane_points(r, var):
                    targets['%s<-%s' % (var, 'plain' if key[0] == plain_target(var) else 'other')] += 1
            lines.append('         L0 targets: %s' % ', '.join('%s %d' % kv for kv in sorted(targets.items())))
            if kind(system) in ('driftx', 'drifty'):
                var = variables[0]
                pl = [p for p in plane_points(r, var) if p[0][0] == plain_target(var)]
                groups = collections.Counter(g for p in pl for g in token_groups(p[0][1]))
                lines.append('         plain-target L0 eqs %d; token groups %s' % (
                    len(pl), ', '.join('%s %d' % kv for kv in sorted(groups.items()))))
                for key, d, i, _row in sorted(pl, key=lambda p: p[1])[:4]:
                    lines.append('           WAPE %.4f instab %.3g | %s' % (d, i, ' + '.join(key[1])))
            sysout[a] = dict(epochs_run=r['epochs_run'], l0=len(r['l0']), held=per, full_any=fa, full_strict=fs,
                             first=r['first'], exact_first=r['full'], wall=wall, targets=dict(targets), pop=r['pop'])
        for var in variables:
            n_pl, _share, members = union[(var, True)]
            if members and kind(system) != 'spin':
                lines.append('  union plain-target front of %s (%d):' % (var, n_pl))
                for (a, _t, terms), d, i in sorted(members, key=lambda m: m[1]):
                    lines.append('     %-6s WAPE %.4f instab %.3g | %s' % (a, d, i, ' + '.join(terms)))
        sysout['union'] = {'%s|%s' % (v, 'plain' if p else 'all'): dict(n=u[0], share=u[1]) for (v, p), u in union.items()}


def field_section(runs, lines, out):
    fr = [r for r in runs if kind(r['sys']) in ('fieldv', 'fieldv2')]
    if not fr:
        return
    lines.append('=' * 110)
    lines.append('FIELDS, one search per field (other fields + x/y derivatives + Laplacians as data tokens, rho_bar); '
                 'fieldv = space orders 4 (6x6 interior), fieldv2 = orders 2 (10x10)')
    import robot_systems as RS
    for r in sorted(fr, key=lambda r: (RS.parse(r['sys'])[1], RS.parse(r['sys'])[2], r['sys'])):
        k, tag, var = RS.parse(r['sys'])
        per, fs, fa = held(r)
        pts = plane_points(r, var)
        wall = r['tsv'].get('wall')
        lines.append('-' * 110)
        lines.append('%-8s %-14s %-3s | ep %d pop %s | L0 %d | HELD any %d strict %d %s | first law-epoch %s | wall %s s' % (
            k, tag, var, r['epochs_run'], r['pop'], len(r['l0']), per[var]['any'], per[var]['strict'],
            per[var]['kinds'], r['first'][var], '%.0f' % wall if wall else '?'))
        for key, d, i, _row in sorted(pts, key=lambda p: p[1])[:8]:
            c = credit(r['sys'], var, key)
            lines.append('     WAPE %.4f instab %.3g %-22s| %s = %s' % (
                d, i, '' if not c else '[%s %s]' % c, key[0], ' + '.join(key[1])))
        out[r['sys']] = dict(held=per, epochs_run=r['epochs_run'], first=r['first'], l0=len(r['l0']), wall=wall,
                             front=[dict(target=k2[0], terms=list(k2[1]), wape=d, instab=i)
                                    for k2, d, i, _ in sorted(pts, key=lambda p: p[1])])


def summary(by_sys, lines):
    lines.append('=' * 110)
    lines.append('SUMMARY spin / drift (final L0; held any(strict); FULL = one L0 system credited in every variable)')
    for group in ('spin', 'drift'):
        rows = [s for s in sorted(by_sys) if kind(s).startswith(group)]
        if not rows:
            continue
        for a in ARMS:
            n = nf = nfs = nv = nva = nvs = 0
            walls = []
            for s in rows:
                r = by_sys[s].get(a)
                if r is None:
                    continue
                n += 1
                per, fs, fa = held(r)
                nf += fa > 0
                nfs += fs > 0
                nv += len(r['vars'])
                nva += sum(per[v]['any'] > 0 for v in r['vars'])
                nvs += sum(per[v]['strict'] > 0 for v in r['vars'])
                if r['tsv'].get('wall'):
                    walls.append(r['tsv']['wall'])
            if n:
                lines.append('  %-5s %-6s systems %d | FULL held any %d strict %d | variables held any %d/%d strict %d/%d | '
                             'wall total %.0f s' % (group, a, n, nf, nfs, nva, nv, nvs, nv, sum(walls)))


def main():
    runs = load_runs()
    by_sys = collections.defaultdict(dict)
    for r in runs:
        if kind(r['sys']) in ('spin', 'driftx', 'drifty'):
            by_sys[r['sys']][r['arm']] = r
    lines, out = [], {}
    summary(by_sys, lines)
    spin_drift_section(by_sys, lines, out)
    field_section(runs, lines, out)
    text = '\n'.join(lines)
    open(os.path.join(HERE, 'results_u.txt'), 'w', encoding='utf-8').write(text + '\n')
    json.dump(out, open(os.path.join(HERE, 'results_u.json'), 'w'), indent=1, default=str)
    print(text)


if __name__ == '__main__':
    main()
