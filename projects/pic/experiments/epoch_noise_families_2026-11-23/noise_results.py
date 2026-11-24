"""Report verified full-record runs, including matched baseline observations."""
import csv
from functools import lru_cache
from io import StringIO
import json

if __name__ == '__main__':
    import psutil
    from cpu_budget import restrict
    restrict(psutil.Process())

from noise_suite import (FAMILIES, LEVEL, OPTIONS, ROOT, SEEDS, atomic_text, baseline_directory,
                         directory, epochs, jobs, live, read_json)
from compare import read_archive
from recovery import catalog_diagnostics
from measurements import rms


METHODS = ('native', 'ordinary', 'halves', 'multiscale')
CONTROLS = ('clean', 'gaussian')


@lru_cache(maxsize=None)
def replay_control(name, seed, level, archive_stamp):
    dest = baseline_directory(name, seed, level)
    archive = read_archive(dest)
    assert (archive['system'], archive['seed'], archive['noise_percent']) == (name, seed, level)
    assert archive['epochs_completed'] == archive['settings']['training_epochs'] == epochs(name)
    assert archive['loader_options'] == OPTIONS.get(name, {}) and archive['budget_unit'] == 'epochs'
    assert archive['settings']['early_stopping_patience'] is None
    problem = live.load_data(name, **OPTIONS.get(name, {}))
    raw = problem.noisy(level, seed)
    assert archive['input_shape'] == list(problem.shape)
    assert archive['train_hashes'] == {k: live.R.digest(v) for k, v in raw.items()}
    return {k: rms(v - problem.data[k]) for k, v in raw.items()}


def write_table(name, rows):
    if rows:
        out = StringIO(newline='')
        columns = dict.fromkeys(key for row in rows for key in row)
        writer = csv.DictWriter(out, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(ROOT / name, out.getvalue())


def records():
    rows, historical = [], []
    for name in live.DATASETS:
        for seed in SEEDS:
            for family in (*CONTROLS, *FAMILIES):
                level = 0 if family == 'clean' else LEVEL
                is_control = family in CONTROLS
                dest = baseline_directory(name, seed, level) if is_control else directory(name, seed, family)
                ready = dest / ('fronts.json' if is_control else 'verified.json')
                if not ready.exists():
                    continue
                fronts = read_json(dest / 'fronts.json')
                if is_control and 'population_size' not in fronts[0]:
                    continue
                assert len(fronts) == 4 and {r['method'] for r in fronts} == set(METHODS)
                if is_control:
                    stamp = (dest / 'archive.json.gz').stat().st_mtime_ns
                    field_rms = replay_control(name, seed, level, stamp)
                else:
                    field_rms = fronts[0]['field_rms']
                for r in catalog_diagnostics(dest, fronts):
                    assert (r['dataset'], r['seed'], r['noise_percent']) == (name, seed, level)
                    assert r.get('noise_family', family) == family
                    rows.append(dict(r, noise_family=family, field_rms=json.dumps(field_rms, sort_keys=True)))
                if name == 'lorenz' and (dest / 'lorenz_identity.json').exists():
                    identity = read_json(dest / 'lorenz_identity.json')
                    assert len(identity) == 2 and {r['method'] for r in identity} == {'native', 'multiscale'}
                    for r in identity:
                        assert (r['seed'], r['noise_percent']) == (seed, level)
                        historical.append(dict(noise_family=family, **r))
    return rows, historical


def render():
    rows, historical = records()
    native = {(r['dataset'], r['seed'], r['noise_family']): r for r in rows if r['method'] == 'native'}
    assert len(rows) == len(native) * len(METHODS)
    new = sum(family in FAMILIES for _, _, family in native)
    n_controls = len(native) - new
    expected_new = len(jobs())
    expected_controls = len(live.DATASETS) * len(SEEDS) * len(CONTROLS)
    complete = (new == expected_new and n_controls == expected_controls
                and len(historical) == 2 * len(SEEDS) * (len(CONTROLS) + len(FAMILIES)))
    lines = ['# Structural recovery under three measurement noise models', '',
             f'{"Complete" if complete else "In progress"}. {new}/{expected_new} new searches and '
             f'{n_controls}/{expected_controls} controls have verified measurements and four scored fronts.', '',
             'Each cell counts runs with a complete catalog match or verified equivalent system on the final native first front. '
             'The core suite has 45 runs per condition and the oscillator has three.', '',
             '| Dataset | Clean | Gaussian 5% | Spikes 5% | Correlated 5% | Signal-dependent 5% |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('Core suite', *live.DATASETS):
        counts = []
        for family in (*CONTROLS, *FAMILIES):
            group = [r for (n, _, f), r in native.items()
                     if f == family and (n == name or (name == 'Core suite' and n != 'ode_simple'))]
            counts.append(f"{sum(r['front_hit'] for r in group)}/{len(group)}" if group else 'pending')
        lines.append('| ' + ' | '.join((name, *counts)) + ' |')
    lines += ['', '## Generation and retention across fronts', '',
              'Counts use the same completed searches across all four front definitions.', '',
              '| Group | Noise family | Searches | Generated | Native | Ordinary | Two-slab CV | Multiscale CV |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for oscillator, group in [(False, 'Core suite'), (True, 'Oscillator')]:
        for family in (*CONTROLS, *FAMILIES):
            selected = [r for r in rows if (r['dataset'] == 'ode_simple') == oscillator and r['noise_family'] == family]
            native_runs = [r for r in selected if r['method'] == 'native']
            hits = [sum(r['front_hit'] for r in selected if r['method'] == method) for method in METHODS]
            values = [group, family, len(native_runs), sum(r['generated'] for r in native_runs), *hits]
            lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## Catalog recovery on the native front', '',
              'Core counts use the structural registry and exclude historical Lorenz identity credit. '
              'The oscillator retains its original identity rule. Each comparison pairs the same dataset and seed.', '',
              '| Group | Noise family | Catalog hits / runs | Catalog generated | Gains vs clean | Losses vs clean | Gains vs Gaussian | Losses vs Gaussian |',
              '|---|---|---:|---:|---:|---:|---:|']
    for oscillator, group in [(False, 'Core suite'), (True, 'Oscillator')]:
        for family in (*CONTROLS, *FAMILIES):
            selected = [r for (n, _, f), r in native.items() if (n == 'ode_simple') == oscillator and f == family]
            counts = []
            for control in CONTROLS:
                pairs = [(r['catalog_front_hit'], native[r['dataset'], r['seed'], control]['catalog_front_hit'])
                         for r in selected if (r['dataset'], r['seed'], control) in native]
                counts.extend([sum(a > b for a, b in pairs), sum(a < b for a, b in pairs)])
            hits = sum(r['catalog_front_hit'] for r in selected)
            values = [group, family, f'{hits}/{len(selected)}', sum(r['catalog_generated'] for r in selected), *counts]
            lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## Paired changes in native-front recovery', '',
              'A gain means only the new noise family recovered the system. A loss means only the control did. '
              'The archive-hit column counts pairs where the new search generated a registered system at any epoch.',
              '', '| Group | Noise family | Control | Pairs | Gains | Losses | Archive hits |',
              '|---|---|---|---:|---:|---:|---:|']
    for oscillator, group in [(False, 'Core suite'), (True, 'Oscillator')]:
        for family in FAMILIES:
            for control in CONTROLS:
                pairs = [(r, native[n, s, control]) for (n, s, f), r in native.items()
                         if f == family and (n == 'ode_simple') == oscillator and (n, s, control) in native]
                gains = sum(a['front_hit'] > b['front_hit'] for a, b in pairs)
                losses = sum(a['front_hit'] < b['front_hit'] for a, b in pairs)
                generated = sum(a['generated'] for a, _ in pairs)
                lines.append(f'| {group} | {family} | {control} | {len(pairs)} | {gains} | {losses} | {generated} |')
    lines += ['', 'The per-run table includes all four fronts, archive recovery, realized per-field RMS, '
              'and search cost. A catalog match or a verified complete-system identity counts once per run. '
              'The Lorenz audit shows the identity evidence included in those counts.', '',
              '[Per-run results](runs.csv) · [Lorenz identity credit](historical_lorenz.csv) · [Protocol](protocol.md)']
    write_table('runs.csv', rows)
    write_table('historical_lorenz.csv', historical)
    atomic_text(ROOT / 'findings.md', '\n'.join(lines) + '\n')
    return complete


if __name__ == '__main__':
    render()
