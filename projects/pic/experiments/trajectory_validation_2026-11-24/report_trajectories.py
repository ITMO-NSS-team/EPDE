"""Summarize the confirmation without pooling repeated noise draws as new systems."""
import csv
from io import StringIO
import json
import re
from statistics import median
from settings import DATASETS, FRESH, ROOT

METHODS = ('original', 'fresh_same', 'fresh_new', 'disjoint', 'buffered')
FRONT_METHODS = {
    'original':'Original recording',
    'fresh_new':'New initial states, averaged WAPE per equation',
    'fresh_same_front':'Same initial state, separate recording objectives',
    'fresh_new_front':'New initial states, separate recording objectives',
}
DATASET_NAMES = {'ode':'Forced oscillator', 'vdp':'Van der Pol', 'duffing':'Duffing', 'lv':'Lotka–Volterra'}


def table(rows):
    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def rows_for(scores):
    rows = []
    for s in scores:
        for method in (*METHODS, 'fresh_same_front', 'fresh_new_front'):
            entry = s['methods'].get(method)
            row = dict(cohort='confirmation' if s['seed'] in FRESH else 'retrospective',
                       dataset=s['dataset'], seed=s['seed'], family=s['family'], method=method,
                       generated=s['generated'], available=int(entry is not None))
            for k in ('front_hit','front_size'):
                row[k] = entry[k] if entry is not None else None
            rows.append(row)
    return rows


def render():
    scores = [json.loads(p.read_text()) for p in sorted((ROOT/'pareto_scores').glob('*/*/*.json'))]
    expected = {(n,s,f) for n in DATASETS for s in range(923701,923704) for f in ('correlated','clean','gaussian')}
    expected.update((n,s,'correlated') for n in DATASETS for s in FRESH)
    assert {(x['dataset'],x['seed'],x['family']) for x in scores} == expected and len(scores) == len(expected)
    rows = rows_for(scores)
    def subset(cohort, family, dataset=None):
        return [x for x in scores if (x['seed'] in FRESH) == (cohort == 'confirmation') and x['family'] == family
                and (dataset is None or x['dataset'] == dataset)]
    def hits(sub, method):
        return sum(x['methods'][method]['front_hit'] for x in sub if method in x['methods'])
    def cell(sub, method):
        n = sum(method in x['methods'] for x in sub)
        return f"{hits(sub,method)}/{n}" if n else 'unavailable'
    fresh = subset('confirmation','correlated')
    for s in scores:
        for old, joint in [('fresh_new','fresh_new_front'),('fresh_same','fresh_same_front')]:
            assert set(s['methods'][old]['front']) <= set(s['methods'][joint]['front'])
    lines = ['# Validating discovered equations on additional trajectories', '',
             'Twelve new full-record searches used 5% correlated measurement noise. The additional recordings '
             'were simulated from the benchmark dynamics. Each method scores the same '
             'candidates from a given run. The original-record comparison reconstructs the ordinary archive front. '
             'The native optimizer can return a different front.', '',
             '## Multisample Pareto front', '',
             'Each recording contributes a separate weak-form equation-fit error (WAPE) for each equation. '
             'Each equation also contributes its training coefficient instability. All objectives are minimized. '
             'The mean-WAPE eligibility rule is shared across comparisons. The full front retains every '
             'nondominated archived system, including ties. Archive order is for lookup. '
             'Held-out errors are available per recording and equation for every front member.', '',
             '| Validation and front construction | Correct law on front | Median front size |',
             '|---|---:|---:|']
    for method,label in FRONT_METHODS.items():
        sizes = [s['methods'][method]['front_size'] for s in fresh]
        lines.append(f"| {label} | {cell(fresh,method)} | {median(sizes):g} |")
    lines += ['', '| Dataset | Generated | On multisample front with new initial states |',
              '|---|---:|---:|']
    for name in DATASETS:
        sub = subset('confirmation','correlated',name)
        lines.append(f"| {DATASET_NAMES[name]} | {sum(s['generated'] for s in sub)}/{len(sub)} | "
                     f"{cell(sub,'fresh_new_front')} |")
    lines += ['', '## Per-equation WAPE and instability fronts', '',
              'Each cell counts runs whose front contains a registered law. The denominator counts evaluable runs.', '',
              '| Dataset | Original recording | New draws, same trajectory | New initial states | Disjoint halves | Buffered halves |',
              '|---|---:|---:|---:|---:|---:|']
    for name in (*DATASETS, 'All four'):
        sub = fresh if name == 'All four' else subset('confirmation','correlated',name)
        lines.append('| ' + ' | '.join([DATASET_NAMES.get(name,name), *[cell(sub,m) for m in METHODS]]) + ' |')
    lines += ['', '## Retrospective controls', '',
              'Each cell counts runs whose front contains a registered law.', '',
              '| Measurements | Original recording | New draws, same trajectory | New initial states, mean WAPE | Buffered halves | New initial states, multisample front |',
              '|---|---:|---:|---:|---:|---:|']
    for f in ('clean','gaussian','correlated'):
        sub = subset('retrospective',f)
        lines.append('| ' + ' | '.join([f, *[cell(sub,m) for m in ('original','fresh_same','fresh_new','buffered','fresh_new_front')]]) + ' |')
    clean_lv = subset('retrospective','clean','lv')
    lines += ['', f"On clean Lotka–Volterra data the original archive front recovered the system in "
              f"{hits(clean_lv,'original')}/3 runs and new-state validation recovered it in "
              f"{hits(clean_lv,'fresh_new')}/3. This limits the case for a global scoring change.",
              '', '## Single-record confirmation', '',
              'Six separate searches saw only the first 153 samples. The adjacent and buffered validation segments '
              'each have 153 samples. The 14-sample buffer also moves the validation segment later in time.', '',
              '| Dataset | Generated on prefix | Training front | Adjacent segment | Buffered segment |',
              '|---|---:|---:|---:|---:|']
    time_runs = [json.loads(p.read_text()) for p in (ROOT/'time_results').glob('*/*/pareto.json')]
    assert {(x['dataset'],x['seed']) for x in time_runs} == {(n,s) for n in ('ode','vdp') for s in FRESH}
    for name in ('ode','vdp'):
        sub = [x for x in time_runs if x['dataset']==name]
        lines.append('| ' + ' | '.join([DATASET_NAMES[name],f"{sum(x['generated'] for x in sub)}/{len(sub)}",
                                       *[cell(sub,m) for m in ('train','adjacent','buffered')]]) + ' |')
    lines += ['', 'Three seeds per dataset share the stored discovery trajectory and the same two validation initial states. '
              'Different noisy recordings are independent draws of measurement error. Real-data use requires additional '
              'measurements on the specified coordinates with compatible forcing. The earlier full-record '
              'disjoint and buffered halves reuse the discovered library and refit coefficients. In the six '
              'prefix searches above, both validation segments score coefficients fitted only on the prefix.', '']
    sim = json.loads((ROOT/'simulator_check.json').read_text())
    lines += ['Reintegrating the catalog models from the stored start gave relative deviations of ' +
              ', '.join(f"{n} {'/'.join(f'{v:.1%}' for v in sim[n]['relative_error'])}" for n in DATASETS) +
              ' on the rest of each original trajectory. For the second-order systems the first two position '
              'samples determine the initial velocity. The two Lotka–Volterra figures correspond to its two fields. '
              '[Simulator check](simulator_check.json)', '',
              '## Forced-KdV dispersion measurement', '',
              'Twelve fresh paired cases per noise family. Each estimator uses the catalog support. The average and '
              'cross-record projection see the same two independently measured fields. The true coefficient is −1. '
              'Errors and coefficients are medians. Held-out WAPE uses a third recording and the same raw weak columns for every arm.', '',
              '| Noise | Method | Relative dispersion-column error | Coefficient | Absolute coefficient error | Clean WAPE | Held-out WAPE |',
              '|---|---|---:|---:|---:|---:|---:|']
    with (ROOT/'kdv_measurement.csv').open(newline='') as f:
        kdv = list(csv.DictReader(f))
    assert len(kdv) == 72
    for family in ('gaussian','correlated'):
        for m in ('single','average','cross_record'):
            r = [x for x in kdv if x['method']==m and x['family']==family]
            values = [f'{median(float(x[k]) for x in r):.4f}' for k in
                      ('relative_column_error','coefficient','absolute_coefficient_error','clean_wape','held_wape')]
            lines.append('| ' + ' | '.join([family,m,*values]) + ' |')
    lines += ['', '### Calibration on an unforced record', '',
              'Each method gets two 81 × 81 recordings. The average uses two forced measurements. '
              'The joint and staged fits use the same forced and unforced measurements. '
              'Staging estimates the two intrinsic coefficients from the unforced record before fitting '
              'the forcing amplitude on the forced record. All medians below use a separate cohort of 12 cases per family.', '',
              '| Noise | Method | Dispersion coefficient | Absolute error | Clean forced WAPE | Noisy forced WAPE | Noisy WAPE at exact coefficients |',
              '|---|---|---:|---:|---:|---:|---:|']
    with (ROOT/'kdv_calibration.csv').open(newline='') as f:
        calibration = list(csv.DictReader(f))
    assert len(calibration) == 72
    for family in ('gaussian','correlated'):
        for method in ('average_forced','joint','staged'):
            sub = [x for x in calibration if x['family']==family and x['method']==method]
            values = [f'{median(float(x[k]) for x in sub):.4f}' for k in
                      ('dispersion','abs_error','clean_wape','held_wape','physical_held_wape')]
            lines.append('| ' + ' | '.join([family,method,*values]) + ' |')
    lines += ['', 'The staged calibration improved the dispersion coefficient under both noise families. '
              'On the noisy forced recording, even the exact physical coefficients gave a large weak residual. '
              'Use the calibration result when comparable unforced measurements and a shared intrinsic law '
              'are available. Identifying the full forced law through search remains an open question.',
              '', '## Navier–Stokes generation', '',
              'Each entry counts archives whose actual system handovers contained the named complete equation. '
              'The last column asks whether the three registered equations could be combined from different '
              'handovers even allowing arbitrary slot reassignment.', '',
              '| Noise | x-momentum | y-momentum | Continuity | Assembled full system | All required donors available |',
              '|---|---:|---:|---:|---:|---:|']
    with (ROOT/'ns_donors.csv').open(newline='') as f:
        donors = list(csv.DictReader(f))
    for family in ('gaussian','correlated'):
        sub = [x for x in donors if x['family']==family]
        assert len(sub)==3
        lines.append('| ' + ' | '.join([family,*[f"{sum(int(x[k]) for x in sub)}/3" for k in
                      ('x_momentum','y_momentum','continuity','assembled','all_donors_any_slot')]]) + ' |')
    lines += ['', 'For the three correlated-noise seeds, candidate generation for x-momentum is the first '
              'bottleneck. A ranking change can only select equations already present in the archive.',
              '', '## CPU use', '']
    loads = []
    pools = set()
    for log in (ROOT/'results').glob('*/*/*/worker.log'):
        for load, pool in re.findall(r'Admitted at ([\d.]+)% host CPU, shared pool (\[[\d, ]+\])', log.read_text()):
            loads.append(float(load))
            pools.add(pool)
    if loads:
        lines += [f"The full-record launch logs recorded host CPU use of {min(loads):.1f}% to {max(loads):.1f}% "
                  f"at admission. The shared logical CPU pool was {', '.join(sorted(pools))}. "
                  "The experiment ran up to two single-threaded search workers at below-normal priority.", '']
    lines += ['[Protocol and reproduction](protocol.md) · [Per-run fronts](pareto_runs.csv) · '
              '[Dispersion measurements](kdv_measurement.csv) · [KdV calibration](kdv_calibration.csv) · '
              '[Navier–Stokes donor audit](ns_donors.csv)']
    for name, contents in [('pareto_runs.csv',table(rows)),('findings.md','\n'.join(lines)+'\n')]:
        path = ROOT/name
        temp = path.with_suffix('.pending')
        temp.write_text(contents, encoding='utf8')
        temp.replace(path)


if __name__ == '__main__':
    render()
