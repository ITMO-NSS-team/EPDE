"""Nov 12 robot revision: every run on ONE frozen EPDE snapshot (9bd360e, epde_snap_9bd360e). Arms:
  U2   strong, EPDE default (plain NSGA-II -> universal search U)
  UW2  weak (evaluation_form='weak': stored tokens opaque, noise-model guard, half-power window gate)
  UT2  strong + system_rps_params target_axes 0 (fields only)
  UWT2 weak + target_axes 0 (fields only)
next to the Nov 10 strong U on the pre-fix code (arm U, porcelain e4a6). Seed 0, equal epochs (spin/drift 20, fields 30).
Credit as results_u.py: held = a reference form (identities included) or a reference form plus even-order derivative
absorbers on the FINAL L0, any credit (strict = exact form only); FULL = one L0 system credited in every variable.
WAPE is compared only within one form: weak rows and strong rows are different data.
Every v2 run's _form.json must carry head 9bd360e and digest_equal; the script lists any that do not.
python results_v2.py -> stdout + results_v2.txt / results_v2.json"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import results_u as RU  # noqa: E402
import robot_systems as RS  # noqa: E402

SNAP_HEAD = '9bd360e'
ARMS = ['U', 'U2', 'UW2', 'UT2', 'UWT2']
FORM = {'U': 'strong', 'U2': 'strong', 'UT2': 'strong', 'UW2': 'weak', 'UWT2': 'weak'}
PAT = re.compile(r'^auto_(?P<sys>.+)_trace_(?P<arm>U|U2|UW2|UT2|UWT2)_e(?P<ep>\d+)\.json$')
TSV = RU.runs_tsv(HERE)
LINES, OUT, BAD = [], {}, []


def say(s=''):
    LINES.append(s)


def load():
    runs = []
    for path in sorted(glob.glob(os.path.join(HERE, 'auto_*_trace_*.json'))):
        m = PAT.match(os.path.basename(path))
        if not m or RU.kind(m['sys']) not in ('spin', 'driftx', 'drifty', 'fieldv2'):
            continue
        prefix = '%s_trace_%s_e%s' % (m['sys'], m['arm'], m['ep'])
        rep = json.load(open(path))
        side = path[:-5] + '_form.json'
        side = json.load(open(side)) if os.path.exists(side) else {}
        if m['arm'] != 'U':
            head = (side.get('epde_digest_start') or {}).get('head', '')
            if not head.startswith(SNAP_HEAD) or not side.get('digest_equal') or side.get('form') != FORM[m['arm']]:
                BAD.append('%s head %s digest_equal %s form %s' % (prefix, head[:7], side.get('digest_equal'),
                                                                   side.get('form')))
        l0 = rep['levels'][0] if rep['levels'] else []
        variables = list(l0[0]['marks']) if l0 else []
        rows = RU.stop_rows(HERE, prefix)
        first, full = RU.first_epochs(rows, variables)
        runs.append(dict(sys=m['sys'], arm=m['arm'], epochs=int(m['ep']), prefix=prefix, rep=rep, l0=l0,
                         vars=variables, first=first, full=full, epochs_run=len(rows), tsv=TSV.get(prefix, {}),
                         pop=rep['settings'].get('population'), last=rows[-1] if rows else {}))
    return runs


def targets(r, var):
    c = collections.Counter()
    for key, _d, _i, _row in RU.plane_points(r, var):
        c['time' if key[0] == RU.plain_target(var) else 'other'] += 1
    return c


def spin_drift(by_sys):
    for system in sorted(by_sys, key=lambda s: (['spin', 'driftx', 'drifty'].index(RU.kind(s)), s)):
        arms = by_sys[system]
        variables = next(iter(arms.values()))['vars']
        say('=' * 118)
        say('%s   variables %s' % (system, variables))
        OUT[system] = {}
        for a in ARMS:
            r = arms.get(a)
            if r is None:
                continue
            per, fs, fa = RU.held(r)
            tg = {v: dict(targets(r, v)) for v in variables}
            say('  %-5s %-6s ep %2d/%d | L0 %3d | held any(strict) %s | FULL %d(%d) | first law-epoch %s exact %s | '
                'targets %s | wall %s' % (
                    a, FORM[a], r['epochs_run'], r['epochs'], len(r['l0']),
                    ' '.join('%s %d(%d)' % (v, per[v]['any'], per[v]['strict']) for v in variables), fa, fs,
                    ' '.join('%s:%s' % (v, r['first'][v]) for v in variables), r['full'],
                    ' '.join('%s %s' % (v, tg[v]) for v in variables), r['tsv'].get('wall')))
            kinds = '; '.join('%s: %s' % (v, ', '.join(per[v]['kinds'])) for v in variables if per[v]['kinds'])
            if kinds:
                say('                credited: %s' % kinds)
            if RU.kind(system) in ('driftx', 'drifty'):
                var = variables[0]
                pl = [p for p in RU.plane_points(r, var) if p[0][0] == RU.plain_target(var)]
                groups = collections.Counter(g for p in pl for g in RU.token_groups(p[0][1]))
                say('                plain-target L0 %d; token groups %s' % (
                    len(pl), ', '.join('%s %d' % kv for kv in sorted(groups.items()))))
                for key, d, i, _row in sorted(pl, key=lambda p: p[1])[:4]:
                    c = RU.credit(system, var, key)
                    say('                  WAPE %.4f instab %.3g %s| %s' % (d, i, '' if not c else '[%s %s] ' % c,
                                                                       ' + '.join(key[1])))
            else:
                for var in variables:
                    pts = sorted(RU.plane_points(r, var), key=lambda p: p[1])[:3]
                    for key, d, i, _row in pts:
                        c = RU.credit(system, var, key)
                        say('                  %s WAPE %.4f instab %.3g %s| %s = %s' % (
                            var, d, i, '' if not c else '[%s %s] ' % c, key[0], ' + '.join(key[1])))
            OUT[system][a] = dict(held=per, full_any=fa, full_strict=fs, first=r['first'], exact_first=r['full'],
                                  l0=len(r['l0']), targets=tg, wall=r['tsv'].get('wall'), epochs_run=r['epochs_run'])


def fields(runs):
    fr = collections.defaultdict(dict)
    for r in runs:
        if RU.kind(r['sys']) == 'fieldv2':
            fr[r['sys']][r['arm']] = r
    say('=' * 118)
    say('FIELDS fieldv2 (one search per field; other fields + x/y derivatives + Laplacians as data tokens, rho_bar; '
        '10x10 domain), 30 epochs')
    for system in sorted(fr, key=lambda s: (RS.parse(s)[1], ['rho', 'Px', 'Py', 'Wx', 'Wy'].index(RS.parse(s)[2]))):
        _k, tag, var = RS.parse(system)
        say('-' * 118)
        say('%s %s' % (tag, var))
        OUT[system] = {}
        for a in ARMS:
            r = fr[system].get(a)
            if r is None:
                continue
            per, _fs, _fa = RU.held(r)
            pts = RU.plane_points(r, var)
            tg = dict(targets(r, var))
            say('  %-5s %-6s ep %d | L0 %3d targets %s | held any %d strict %d %s | first law-epoch %s | wall %s' % (
                a, FORM[a], r['epochs_run'], len(pts), tg, per[var]['any'], per[var]['strict'], per[var]['kinds'],
                r['first'][var], r['tsv'].get('wall')))
            tp = sorted([p for p in pts if p[0][0] == RU.plain_target(var)], key=lambda p: p[1])[:3]
            for key, d, i, _row in tp:
                c = RU.credit(system, var, key)
                say('        WAPE %.4f instab %.3g %s| %s = %s' % (d, i, '' if not c else '[%s %s] ' % c, key[0],
                                                                 ' + '.join(key[1])))
            OUT[system][a] = dict(held=per, l0=len(pts), targets=tg, first=r['first'], wall=r['tsv'].get('wall'),
                                  time_front=[dict(terms=list(k[1]), wape=d, instab=i) for k, d, i, _ in tp])


def summary(runs):
    say('=' * 118)
    say('SUMMARY (final L0; any credit (strict))')
    by = collections.defaultdict(list)
    for r in runs:
        g = {'spin': 'spin', 'driftx': 'drift', 'drifty': 'drift', 'fieldv2': 'fields'}[RU.kind(r['sys'])]
        by[(g, r['arm'])].append(r)
    for g in ('spin', 'drift', 'fields'):
        for a in ARMS:
            rs = by.get((g, a))
            if not rs:
                continue
            nf = nfs = nv = nva = nvs = 0
            pervar = collections.Counter()
            time_l0 = 0
            walls = [r['tsv']['wall'] for r in rs if r['tsv'].get('wall')]
            for r in rs:
                per, fs, fa = RU.held(r)
                nf += fa > 0
                nfs += fs > 0
                for v in r['vars']:
                    nv += 1
                    nva += per[v]['any'] > 0
                    nvs += per[v]['strict'] > 0
                    if per[v]['any']:
                        pervar[v] += 1
                    time_l0 += targets(r, v).get('time', 0) > 0
            extra = ''
            if g == 'fields':
                extra = ' | per field %s | runs with a time-target L0 eq %d' % (
                    ' '.join('%s %d' % (v, pervar[v]) for v in ['rho', 'Px', 'Py', 'Wx', 'Wy']), time_l0)
            else:
                extra = ' | FULL any %d strict %d' % (nf, nfs)
            say('  %-6s %-5s %-6s runs %2d | variables held any %d/%d strict %d/%d%s | wall %.0f s' % (
                g, a, FORM[a], len(rs), nva, nv, nvs, nv, extra, sum(walls)))


def main():
    runs = load()
    by_sys = collections.defaultdict(dict)
    for r in runs:
        if RU.kind(r['sys']) in ('spin', 'driftx', 'drifty'):
            by_sys[r['sys']][r['arm']] = r
    summary(runs)
    if BAD:
        say('RUNS NOT ON THE SNAPSHOT / FORM MISMATCH (%d):' % len(BAD))
        for b in BAD:
            say('  ' + b)
    spin_drift(by_sys)
    fields(runs)
    text = '\n'.join(LINES)
    open(os.path.join(HERE, 'results_v2.txt'), 'w', encoding='utf-8').write(text + '\n')
    json.dump(OUT, open(os.path.join(HERE, 'results_v2.json'), 'w'), indent=1, default=str)
    print(text)


if __name__ == '__main__':
    main()
