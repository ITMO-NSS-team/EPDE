"""Measure the structural patience rule without truncating discovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import patch

from epoch_settings import NOISE, OPTIONS, ROOT, SEEDS, directory, epochs, live, np, trial, whole_record
from compare import read_archive, structural_matches, write_csv
from recovery import lorenz_identity


def summarize(job, archive):
    name, seed, noise = job
    problem = live.load_data(name, **OPTIONS.get(name, {}))
    history = archive['epoch_fronts']
    assert [row['epoch'] for row in history] == list(range(1, epochs(name)+1))
    combos = sorted({tuple(c) for row in history for c in row['members']})
    ids = {c:i for i,c in enumerate(combos)}
    _, _, hit = structural_matches(name, problem, archive, np.asarray(combos, dtype=int))
    identity = lorenz_identity(archive) if name=='lorenz' else None
    # The endpoint is the returned full-system front at the stopping epoch.
    def outcome(front):
        return int(hit([ids[tuple(c)] for c in front]) or (identity is not None and identity(front)))

    final = outcome(archive['native_front'])
    rows = []
    for patience in (3, 5, 10):
        earliest = next((row for row in history if row['unchanged']>=patience), None)
        observed = earliest or history[-1]
        rows.append(dict(dataset=name, seed=seed, noise_percent=noise, patience=patience,
                         trigger_epoch=earliest['epoch'] if earliest else None,
                         epochs_saved=epochs(name)-observed['epoch'], full_budget_hit=final,
                         stopping_epoch_hit=outcome(observed['members']),
                         longest_unchanged=max(row['unchanged'] for row in history)))
    return rows


def assess(job):
    name, seed, noise = job
    dest = directory(*job)
    archive = read_archive(dest)
    if 'epoch_fronts' not in archive:
        # Earlier workers kept just the terminal front. Replay with identical RNG and verify
        # the discovery path before crediting their per-epoch counterfactuals.
        with TemporaryDirectory(prefix='patience_replay_',dir=dest) as temp:
            with patch.object(live,'split',whole_record), patch.object(trial,'folder',return_value=Path(temp)):
                trial.worker(name,seed,'candidate',noise_level=noise,epoch_budget=epochs(name),
                             loader_options=OPTIONS.get(name))
            copy = read_archive(Path(temp))
            for field in ('settings','plan','train_hashes','variables','labels','points','systems','native_front',
                          'system_events','calls','epochs_completed'):
                assert copy[field] == archive[field], (job,field)
            archive['epoch_fronts'] = copy['epoch_fronts']
            del copy
    rows = summarize(job, archive)
    tmp = dest/'patience.pending'
    tmp.write_text(json.dumps(rows), encoding='utf8')
    tmp.replace(dest/'patience.json')
    if name=='lorenz':
        selected = {r['trigger_epoch'] or epochs(name) for r in rows} | {epochs(name)}
        proof = [row for row in archive['epoch_fronts'] if row['epoch'] in selected]
        tmp = dest/'patience_fronts.pending'
        tmp.write_text(json.dumps(proof), encoding='utf8')
        tmp.replace(dest/'patience_fronts.json')
    report()


def report():
    rows = [r for path in (ROOT/'results').rglob('patience.json') for r in json.loads(path.read_text())
            if r['dataset']!='lorenz' or path.with_name('patience_fronts.json').is_file()]
    if not rows:
        return
    write_csv(ROOT/'patience.csv', rows)
    lines = ['# Structural early stopping on full records', '',
             'EPDE compares the set of structures on the reported first Pareto front after each epoch. '
             'For patience 5, the first possible trigger is epoch 6. The search ran to its full budget. '
             'Earlier runs were replayed and their full discovery paths checked against the original archives.', '',
             'The count stopped early excludes a trigger at the budget itself. Epochs saved is the median among '
             'runs that would stop early. Late recoveries had a complete registered system on the final front '
             'but lacked it at the stopping epoch. Early-only recoveries show the reverse.', '',
             '| Group | Noise | Patience | Assessed | Stopped early | Epochs saved | Hits at stop | Hits at full budget | Late recoveries | Early-only recoveries |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('Core', 'Oscillator control'):
        for noise in NOISE:
            for k in (3,5,10):
                selected = [r for r in rows if r['patience']==k and r['noise_percent']==noise
                            and (r['dataset']=='ode_simple')==(group=='Oscillator control')]
                if not selected:
                    continue
                early = [r for r in selected if r['epochs_saved']]
                saved = median(r['epochs_saved'] for r in early) if early else '–'
                stop = sum(r['stopping_epoch_hit'] for r in selected)
                full = sum(r['full_budget_hit'] for r in selected)
                late = sum(r['stopping_epoch_hit'] < r['full_budget_hit'] for r in selected)
                early_only = sum(r['stopping_epoch_hit'] > r['full_budget_hit'] for r in selected)
                lines.append(f'| {group} | {noise}% | {k} | {len(selected)} | {len(early)} | {saved} | '
                             f'{stop} | {full} | {late} | {early_only} |')
    lines += ['', '[Per-run stop epochs and structural recovery](patience.csv)']
    tmp = ROOT/'patience_summary.pending'
    tmp.write_text('\n'.join(lines)+'\n', encoding='utf8')
    tmp.replace(ROOT/'patience_summary.md')


def run_job(job):
    dest = directory(*job)
    with (dest/'patience.log').open('w') as log:
        result = subprocess.run([sys.executable,'-u',__file__,'--worker',*map(str, job)],
                                stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ,PYTHONHASHSEED='0'))
    if result.returncode:
        raise RuntimeError(f'{job}: see {dest/"patience.log"}')
    print('Checked structural patience', *job, flush=True)


def watch():
    jobs = [(n,s,q) for s in SEEDS for q in NOISE for n in live.DATASETS]
    pending = set(jobs)
    active = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        while pending or active:
            for job, future in list(active.items()):
                if future.done():
                    future.result()
                    del active[job]
            for job in jobs:
                if job not in pending:
                    continue
                dest = directory(*job)
                ready = dest/'patience.json'
                if ready.is_file() and (job[0]!='lorenz' or (dest/'patience_fronts.json').is_file()):
                    pending.remove(job)
                elif not active and (dest/'archive.json.gz').is_file():
                    active[job] = pool.submit(run_job, job)
                    pending.remove(job)
            if active or pending:
                sleep(10)
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=3)
    args = parser.parse_args()
    if args.worker:
        n,s,q = args.worker
        assess((n,int(s),int(q)))
    else:
        watch()
