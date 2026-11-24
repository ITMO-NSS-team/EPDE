"""Reconstruct Pareto fronts on original and noisy fields using paired discovery seeds."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import json
from statistics import median
import time
from unittest.mock import patch

from clean_search import directory
from compare import previous, read_archive, structural_matches, system_front, system_objectives, write_csv
from dyad_discrepancy import native_row_scores
from experiment import ROOT, full_directory, live, np, trial, whole_record


def archive_dir(job):
    name, seed, design, noise = job
    if noise == 0:
        return directory(name, seed, design)
    locate = full_directory if design == 'full' else trial.folder
    return locate(name, seed, 'candidate')


def assessment_path(job):
    name, seed, design, noise = job
    return ROOT/'noise_fronts'/str(noise)/design/name/f'{seed}.json'


def assess(job):
    name, seed, design, noise = job
    archive = read_archive(archive_dir(job))
    assert (archive['system'],archive['seed'],archive['arm']) == (name,seed,'candidate')
    assert archive.get('noise_percent',5) == noise
    assert archive['calls'] == live.BUDGET and archive['operator']['full_cost'][-1] >= live.BUDGET
    assert archive['settings']['system_rps_params']['variable_order_seed'] == seed
    problem = live.load_data(name)
    raw = problem.noisy(noise, seed)
    split = whole_record if design == 'full' else live.split
    with patch.object(previous, 'split', split):
        train, held, dyad, valid, error = previous.replay(problem, raw, archive)
    combos = np.asarray(archive['systems'])[np.asarray(archive['operator']['system_cost']) <= live.BUDGET]
    if design == 'full':
        (halves, all_scales), _ = native_row_scores(problem, raw, archive, combos)
        rules = dict(ordinary=train, halves=halves, multiscale=all_scales)
    else:
        rules = dict(contiguous=held)
    _, _, hit = structural_matches(name, problem, archive, combos)
    generated = hit(range(len(combos)))
    rows = []
    for rule, score in rules.items():
        pool, objectives = system_objectives(score, dyad, valid, combos)
        front = pool[system_front(objectives)]
        rows.append(dict(dataset=name, seed=seed, noise_percent=noise, rule=rule, assembled=generated,
                         front_hit=hit(front), front_size=len(front), combinations=len(combos),
                         eligible=len(pool), replay_error=error))
    # Store only verified results so interrupted batches can restart.
    path = assessment_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(rows, allow_nan=False))
    tmp.replace(path)


def check_paired_inputs(jobs):
    clean_hashes = {}
    for name, seed, design, noise in jobs:
        if noise:
            continue
        clean, noisy = (read_archive(archive_dir((name,seed,design,p))) for p in (0,5))
        assert clean['settings'] == noisy['settings'] and clean['variables'] == noisy['variables']
        assert clean['axis'] == noisy['axis'] and clean['calls'] == noisy['calls'] == live.BUDGET
        assert clean['train_hashes'] != noisy['train_hashes']
        key = (name, design)
        assert clean_hashes.setdefault(key, clean['train_hashes']) == clean['train_hashes']


def collect(jobs, workers):
    pending = [job for job in jobs if not assessment_path(job).is_file()]
    active = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for job in pending.copy():
                if len(active) >= workers:
                    break
                if (archive_dir(job)/'archive.json.gz').is_file():
                    active.add(pool.submit(assess, job))
                    pending.remove(job)
            if active:
                completed, active = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
            elif pending:
                time.sleep(5)
    return [r for job in jobs for r in json.loads(assessment_path(job).read_text())]


def report(rows):
    results = {(r['dataset'], r['seed'], r['noise_percent'], r['rule']): r for r in rows}
    assert len(results) == len(rows) == len(live.DATASETS)*len(trial.CONFIRMATION)*8
    for filename, rules in [('dyad_rows_confirmation.csv',
                             {s+'_stability':s for s in ('ordinary','halves','multiscale')}),
                            ('contiguous_confirmation.csv', {'held':'contiguous'})]:
        with (ROOT/filename).open(newline='') as f:
            for old in csv.DictReader(f):
                if old['rule'] in rules:
                    row = results[old['dataset'], int(old['seed']), 5, rules[old['rule']]]
                    for field in ('front_size','front_hit'):
                        assert row[field] == int(old[field]), (row, old)
    write_csv(ROOT/'fronts_by_noise_runs.csv', rows)
    summary = []
    for group in ['suite', *live.DATASETS]:
        names = [n for n in live.DATASETS if n!='ode_simple'] if group=='suite' else [group]
        for rule in ('contiguous','ordinary','halves','multiscale'):
            pairs = [(results[n,s,0,rule], results[n,s,5,rule]) for n in names for s in trial.CONFIRMATION]
            summary.append(dict(dataset=group, rule=rule, paired_runs=len(pairs),
                                clean_generated=sum(a['assembled'] for a,b in pairs),
                                noisy_generated=sum(b['assembled'] for a,b in pairs),
                                clean_front=sum(a['front_hit'] for a,b in pairs),
                                noisy_front=sum(b['front_hit'] for a,b in pairs),
                                both=sum(a['front_hit'] and b['front_hit'] for a,b in pairs),
                                clean_only=sum(a['front_hit'] > b['front_hit'] for a,b in pairs),
                                noisy_only=sum(a['front_hit'] < b['front_hit'] for a,b in pairs),
                                clean_median_size=median(a['front_size'] for a,b in pairs),
                                noisy_median_size=median(b['front_size'] for a,b in pairs)))
    write_csv(ROOT/'fronts_by_noise.csv', summary)
    write_report(summary)
    for row in summary[:4]:
        print(row)


def write_report(summary):
    titles = dict(contiguous='Contiguous holdout', ordinary='Full record, ordinary discrepancy',
                  halves='Full record, two-slab CV', multiscale='Full record, all-scale slab CV')
    lines = [
        '# Complete-system recovery on clean and noisy data',
        '',
        'The endpoint is membership of a correct complete system on the unrestricted first nondominated '
        'front. Each condition covers the same 15 datasets and 12 search seeds, with 3,000 search '
        'fit equivalents per run. The 0% inputs are the stored observations. At 5%, Gaussian noise with '
        'standard deviation 0.05 times each field’s standard deviation is added before discovery. '
        'The physical grid, trajectory and registered equations are shared between conditions. '
        'Search, window selection, candidate frequencies, fitting and validation use the fields from '
        'their own condition. Slab CV adds offline coefficient refits.',
        '',
        '| Discrepancy with coefficient stability | Clean front hits / 180 | 5% noise front hits / 180 | '
        'Median front size, clean | Median front size, 5% noise |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in summary:
        if r['dataset'] == 'suite':
            lines.append(f"| {titles[r['rule']]} | {r['clean_front']} | {r['noisy_front']} | "
                         f"{r['clean_median_size']:g} | {r['noisy_median_size']:g} |")
    lines += ['', '## Full-record discovery and all-scale slab CV by dataset', '',
              'Counts are out of 12 paired search seeds for each dataset. A generated coupled system '
              'counts only if its component laws occurred together in an actual candidate.',
              '', '| Dataset | Generated, clean | On front, clean | Generated, 5% noise | On front, 5% noise |',
              '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['dataset'] != 'suite' and r['rule'] == 'multiscale':
            name = r['dataset'] if r['dataset'] != 'ode_simple' else 'ode_simple (separate control)'
            lines.append(f"| {name} | {r['clean_generated']} | {r['clean_front']} | "
                         f"{r['noisy_generated']} | {r['noisy_front']} |")
    lines += ['', 'The oscillator identity control is excluded from the 180-run totals. Clean seeds vary '
              'search randomness on the same stored fields. Noisy seeds vary both the added noise draw '
              'and search randomness. The original sampling and numerical discretization remain in both '
              'conditions. For full-record CV, discovery saw the full record and adjacent weak windows can '
              'overlap raw observations. [Per-dataset paired counts](fronts_by_noise.csv) · '
              '[Run-level results with noise labels](fronts_by_noise_runs.csv)', '']
    shared_names = {'vdp','lv','lorenz','wave','burgers','kdv','ks','ac','pde_compound','pde_divide','ns'}
    common = [r for r in summary if r['dataset'] in shared_names and r['rule'] == 'multiscale']
    extra = [r for r in summary if r['dataset'] in ('ode','duffing','burgers_inviscid','kdv_cossin')
             and r['rule'] == 'multiscale']
    assert len(common) == 11 and len(extra) == 4
    lines += ['## Comparison with the November 7–8 universal-search benchmark', '',
              'That campaign ran 13 systems at 0% and 5% added noise, with three seeds per condition. '
              'With its full historical credit rule, every system succeeded on at least one clean seed. '
              'Wave was the only system that missed on all three noisy seeds. The original first-front '
              'counts separate the effect of the scoring rules on the same archived fronts.', '',
              '| Historical first-front scoring rule | Clean / 39 | 5% noise / 39 |',
              '|---|---:|---:|', '| Listed structure under the historical strict test | 28 | 25 |',
              '| Also credit algebraic identities | 31 | 28 |',
              '| Also credit registered truncation companions and the NS pressure–Poisson forms | 38 | 30 |', '',
              'Eleven named dynamics occur in both primary suites. The old simple harmonic oscillator is '
              'the current separate control. Solar heat remains cataloged outside the core suite. The '
              'current core includes the forced oscillator, Duffing, inviscid Burgers and forced KdV. '
              'The same 15 names were already present in the October core-15 campaign. '
              '[November universal-search report](../../../robot/epde_u/FINDINGS_RU.md) · '
              '[October core-15 report](../baseline_clean_2026-10-09/README.md)', '',
              f"On the 11 shared dynamics, current all-scale CV returned the registered complete system in "
              f"{sum(r['clean_front'] for r in common)}/132 clean and {sum(r['noisy_front'] for r in common)}/132 noisy runs. "
              f"On the four additional core dynamics, the corresponding counts were "
              f"{sum(r['clean_front'] for r in extra)}/48 and {sum(r['noisy_front'] for r in extra)}/48.", '',
              'The November campaign used 50 epochs for scalar laws and 150 for coupled systems, with '
              'the final full-record search front. The current campaign caps search at 3,000 fit equivalents '
              'and reconstructs the eligible front from actual archived systems. Its truth registry accepts '
              'specific listed supports and equivalents. The old Lorenz run used 100,000 time samples; '
              'the current loader supplies a 1,041-sample attractor segment. The old Navier–Stokes grid '
              'was 50 × 50 × 100 and the current grid is 50 × 20 × 36. These simultaneous changes preclude '
              'a code-regression estimate from the suite totals. [Earlier loaders](../../../robot/epde_u/run_auto.py) · '
              '[Current loaders](../../epde_bench/datasets.py) · [Current scoring](../live_operator_suite_2026-11-23/assess.py)', '']
    lorenz_file = ROOT/'historical_lorenz.csv'
    if lorenz_file.exists():
        with lorenz_file.open(newline='') as stream:
            checks = list(csv.DictReader(stream))
        assert len(checks) == 2*len(trial.CONFIRMATION)
        lines += ['### Lorenz with the original identity checker on the same current fronts', '',
                  '| Scoring of the current all-scale CV front | Clean / 12 | 5% noise / 12 |',
                  '|---|---:|---:|']
        for label, field in [('Current registered supports','catalog_hit'),
                             ('November 7 identity and full-system rank check','history_identity_hit')]:
            hits = [sum(int(r[field]) for r in checks if int(r['noise_percent'])==q) for q in (0,5)]
            lines.append(f'| {label} | {hits[0]} | {hits[1]} |')
        lines += ['', 'Only the evaluation of the already-discovered front members changes in this check. '
                  'The historical checker accepts algebraic rewritings when the three equations span all '
                  'three Lorenz laws. [Paired Lorenz checks](historical_lorenz.csv) · '
                  '[Replay code](historical_lorenz.py)', '']
    (ROOT/'fronts_by_noise.md').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    jobs = [(name, seed, design, noise) for noise in (0,5) for seed in trial.CONFIRMATION
            for name in live.DATASETS for design in ('full','contiguous')]
    rows = collect(jobs, args.workers)
    check_paired_inputs(jobs)
    report(rows)
