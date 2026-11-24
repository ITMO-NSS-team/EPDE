"""Summarize completed epoch runs and evaluate the historical Lorenz identity endpoint."""
import argparse
import csv
import json
from pathlib import Path
import sys
from time import sleep

from epoch_settings import NOISE, OPTIONS, ROOT, SEEDS, directory, live, np
from compare import read_archive, system_front, system_objectives, write_csv
from dyad_discrepancy import native_row_scores
from historical_lorenz import historical_label
from recovery import catalog_diagnostics, lorenz_identity


def score_lorenz(seed, noise, classifier, records, *, dest=None, observation=None):
    dest = directory('lorenz', seed, noise) if dest is None else dest
    records = catalog_diagnostics(dest, records)
    archive = read_archive(dest)
    problem = live.load_data('lorenz', **OPTIONS['lorenz'])
    combos = np.asarray(archive['systems'], dtype=int)
    if observation is None:
        raw = problem.noisy(noise, seed)
    else:
        raw, measurement = observation(problem, noise, seed)
        assert measurement == archive['measurement']
        assert archive['train_hashes'] == {k: live.R.digest(v) for k, v in raw.items()}
    (_, scores), _ = native_row_scores(problem, raw, archive, combos)
    points = archive['points']
    dyad = np.array([p[4] for p in points])
    valid = (np.array([p[3] for p in points]) < live.LOSS_NAN_VAL) & (dyad < live.LOSS_NAN_VAL)
    pool, values = system_objectives(scores, dyad, valid, combos)
    labels = [historical_label(term) for term in archive['labels']]
    def equation(index):
        _, target, pairs, *_ = points[index]
        terms = [j for j, w in pairs if j != target and w != 0.]
        return labels[target], tuple(sorted(labels[j] for j in terms if j != 0)), 0 in terms

    rows = []
    for method, front in [('native', archive['native_front']), ('multiscale', combos[pool[system_front(values)]])]:
        record = next(r for r in records if r['method'] == method)
        assert len(front) == record['front_size']
        keys = {i: equation(i) for combo in front for i in combo}
        flags = {i: classifier.classify(key)['kind'] for i, key in keys.items()}
        positive = unresolved = 0
        for combo in front:
            positive += classifier.full(tuple(keys[i] for i in combo))['full']
            kinds = {flags[i] for i in combo}
            unresolved += 'unparsed' in kinds and kinds <= {'law', 'identity', 'unparsed'}
        rows.append(dict(seed=seed, noise_percent=noise, method=method, front_size=len(front),
                         catalog_hit=record.get('catalog_front_hit', record['front_hit']), historical_hit=int(positive>0),
                         credited_candidates=positive, unresolved_candidates=unresolved))
    output = dest / 'lorenz_identity.json'
    temp = output.with_suffix('.pending')
    temp.write_text(json.dumps(rows), encoding='utf8')
    temp.replace(output)


def read_prior():
    path = ROOT.parent / 'structural_holdout_2026-11-23' / 'fronts_by_noise_runs.csv'
    with path.open(newline='', encoding='utf8') as f:
        prior = {(r['dataset'], int(r['seed']), int(r['noise_percent']), r['rule']): int(r['front_hit'])
                 for r in csv.DictReader(f) if int(r['seed']) in SEEDS}
    from fronts_by_noise import archive_dir
    with path.with_name('historical_lorenz.csv').open(newline='', encoding='utf8') as f:
        for r in csv.DictReader(f):
            s, q = int(r['seed']), int(r['noise_percent'])
            if s not in SEEDS:
                continue
            key = ('lorenz',s,q,'multiscale')
            assert prior[key] == int(r['catalog_hit'])
            prior[key] |= int(r['history_identity_hit'])
            key = ('lorenz',s,q,'ordinary')
            if prior[key]:
                continue
            archive = read_archive(archive_dir(('lorenz',s,'full',q)))
            fits = np.asarray(archive['operator']['system_cost'])
            combos = np.asarray(archive['systems'])[fits<=live.BUDGET]
            train, dyad = np.array([p[3:5] for p in archive['points']]).T
            valid = (train<live.LOSS_NAN_VAL) & (dyad<live.LOSS_NAN_VAL)
            pool, obj = system_objectives(train,dyad,valid,combos)
            prior[key] = int(lorenz_identity(archive)(combos[pool[system_front(obj)]]))
    return prior


def render(rows, historical, prior):
    native = [r for r in rows if r['method'] == 'native']
    total = len(live.DATASETS) * len(SEEDS) * len(NOISE)
    complete = len(native) == total
    lines = ['# Full-record structural discovery at fixed epochs', '',
             f'{len(native)} of {total} runs have completed discovery and front assessment. '
             'A hit requires one candidate on the first Pareto front containing the complete law or a verified '
             'equivalent system. Accepted identities count once per run. Lorenz identities must together span all three laws. '
             'Each condition has three paired seeds per dataset. The oscillator control is reported separately.', '',
             '| Dataset | Clean native | 5% native | Clean multiscale CV | 5% multiscale CV |',
             '|---|---:|---:|---:|---:|']
    for name in ['Core suite', *live.DATASETS]:
        cells = []
        for method in ('native', 'multiscale'):
            for noise in NOISE:
                selected = [r for r in rows if r['noise_percent'] == noise and r['method'] == method
                            and (r['dataset'] == name or (name == 'Core suite' and r['dataset'] != 'ode_simple'))]
                cells.append(f"{sum(r['front_hit'] for r in selected)}/{len(selected)}" if selected else 'pending')
        lines.append('| ' + ' | '.join([name, *cells]) + ' |')
    if native:
        calls = [r['fitness_calls'] for r in native]
        pops = [r['population_size'] for r in native]
        lines += ['', f'Completed runs used {min(calls):,}–{max(calls):,} fitness calls and started with '
                      f'{min(pops)}–{max(pops)} individuals. All completed scalar and coupled runs passed the '
                      'check for their configured 50 and 150 epochs, respectively.']
    if (ROOT/'clean_misses.json').is_file():
        lines += ['', 'The clean inviscid Burgers fronts contain a verified identity of the measured similarity '
                  'solution on all three seeds. This identity was added after the original assessment. '
                  'The earlier 3,000-fit counts below use the registry from that campaign. '
                  '[Clean-run audit](clean_misses.json)']
    lines += ['', '## Comparison with the earlier 3,000-fit campaign', '',
              'Both campaigns have archive fronts for ordinary discrepancy and multiscale CV. Each entry '
              'compares the same datasets and seeds on those fronts. Lorenz and Navier–Stokes also gained samples '
              'in the epoch campaign. Thirteen core datasets used the same observations at each matched noise level.', '',
              '| Inputs | Noise | Archive objective | Paired runs | Earlier hits | Epoch hits | Gains | Losses |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for group in ('Same samples', 'Expanded samples'):
        for q in NOISE:
            for method in ('ordinary', 'multiscale'):
                pairs = []
                for r in rows:
                    n = r['dataset']
                    if (n == 'ode_simple' or r['method'] != method or r['noise_percent'] != q
                            or ((group == 'Expanded samples') != (n in OPTIONS))):
                        continue
                    pairs.append((prior[n, r['seed'], q, method], r['front_hit']))
                lines.append(f'| {group} | {q}% | {method} | {len(pairs)} | '
                             f'{sum(a for a,b in pairs)} | {sum(b for a,b in pairs)} | '
                             f'{sum(b>a for a,b in pairs)} | {sum(b<a for a,b in pairs)} |')
    if historical:
        lines += ['', '## Historical Lorenz identity credit', '',
                  'These identities are included in the recovery totals above. Literal supports remain available in the per-run records.', '',
                  '| Noise | Front | Assessed | Catalog hits | Verified identity-rule hits | Candidates with unresolved identity |',
                  '|---:|---|---:|---:|---:|---:|']
        for q in NOISE:
            for method in ('native','multiscale'):
                sub = [r for r in historical if r['noise_percent'] == q and r['method'] == method]
                lines.append(f'| {q}% | {method} | {len(sub)} | {sum(r["catalog_hit"] for r in sub)} | '
                             f'{sum(r["historical_hit"] for r in sub)} | {sum(r["unresolved_candidates"] for r in sub)} |')
    lines += ['', '[Experiment protocol](protocol.md)']
    if complete and (ROOT/'runs.csv').is_file():
        lines += ['', '[Per-run results](runs.csv) · [All four front definitions by dataset](by_dataset.csv)']
    tmp = ROOT / 'findings.pending'
    tmp.write_text('\n'.join(lines)+'\n', encoding='utf8')
    tmp.replace(ROOT / 'findings.md')


def main(watch):
    prior = read_prior()
    jobs = [(n,s,q) for n in live.DATASETS for s in SEEDS for q in NOISE]
    classifier = None
    previous_state = None
    while True:
        rows, historical = [], []
        for n,s,q in jobs:
            dest = directory(n,s,q)
            path = dest/'fronts.json'
            if not path.is_file():
                continue
            records = json.loads(path.read_text())
            if 'population_size' not in records[0]:
                continue
            if n == 'lorenz':
                if not (dest/'catalog_matches.json').is_file():
                    continue
                records = catalog_diagnostics(dest, records)
                if not (dest/'lorenz_identity.json').is_file():
                    if classifier is None:
                        sys.path.insert(0, str(live.operators.HISTORICAL))
                        from identity_hits import Classifier
                        classifier = Classifier('lorenz', use_cache=False, progress=False)
                    score_lorenz(s,q,classifier,records)
                audited = json.loads((dest/'lorenz_identity.json').read_text())
                for audit in audited:
                    record = next(r for r in records if r['method']==audit['method'])
                    assert audit['catalog_hit'] == record['catalog_front_hit']
                    assert record['front_hit'] == max(audit['catalog_hit'], audit['historical_hit'])
                historical.extend(audited)
            rows.extend(records)
        final = (ROOT/'by_dataset.csv').is_file() and len(rows) == 4 * len(jobs)
        state = len(rows), len(historical), final
        if previous_state != state:
            render(rows, historical, prior)
            print(f'Updated findings for {len(rows)//4} runs.', flush=True)
            previous_state = state
        if final:
            assert len(historical) == 2 * len(SEEDS) * len(NOISE)
            write_csv(ROOT/'historical_lorenz.csv', historical)
        if final or not watch:
            break
        sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    main(parser.parse_args().watch)
