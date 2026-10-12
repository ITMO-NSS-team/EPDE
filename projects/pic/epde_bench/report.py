"""Tables and figures from a campaign directory.

``make_report(Path('results/<name>'))`` writes into ``results/<name>/report/``:

* ``runs.csv``           one row per run (status, metrics, time);
* ``summary.md``         the tables below, ready to paste;
* ``success_front.csv``  success rate per data set x variant x noise;
* ``ranking.csv``        variants ranked per problem class and noise level;
* ``fig_success_noise<n>.png``  heat map data set x variant;
* ``fig_time.png``       median fit time per data set and variant.

Success rates come with Wilson 95 % intervals: with 3-5 seeds per cell the
interval is wide, and the tables say so instead of hiding it.
"""

import json
from pathlib import Path

from .metrics import wilson_ci

CLASS_ORDER = ['ode', 'ode_system', 'pde_1d', 'pde_2d', 'pde_3d']
#: class names as printed in summary.md
KIND_NAMES = {'ode': 'ODE', 'ode_system': 'ODE system', 'pde_1d': 'PDE 1-D',
              'pde_2d': 'PDE 2-D', 'pde_3d': 'PDE 3-D'}


def load_runs(campaign: Path):
    import pandas as pd
    from .identity import problem_metadata
    rows = []
    manifest_path = campaign / 'campaign.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    key = lambda rec: (rec.get('dataset'), rec.get('variant'), rec.get('noise'), rec.get('seed'))
    planned = {key(rec): rec for rec in manifest.get('jobs', [])}
    records = {}
    for path in sorted((campaign / 'runs').glob('*/*.json')):
        rec = json.loads(path.read_text(encoding='utf-8'))
        job = key(rec)
        if job in records:
            raise ValueError(f'duplicate run for {job}')
        if planned and job not in planned:
            raise ValueError(f'record outside campaign manifest: {job}')
        if job in planned and planned[job].get('identity') is not None:
            if rec.get('identity') != planned[job]['identity']:
                raise ValueError(f'record identity differs from campaign manifest: {job}')
        records[job] = rec
    for job in sorted(set(planned) | set(records)):
        plan = planned.get(job, {})
        rec = records.get(job, dict(plan, status='pending'))
        m = rec.get('metrics') or {}
        prob = dict(plan.get('problem') or {})
        prob.update(rec.get('problem') or {})
        if not prob.get('kind') or ('truth_known' not in prob and 'truth' not in prob):
            try:
                registered = problem_metadata(rec['dataset'])
                prob = dict(registered, **prob)
            except KeyError:
                pass
        rows.append({
            'dataset': rec.get('dataset'), 'variant': rec.get('variant'),
            'noise': rec.get('noise'), 'seed': rec.get('seed'), 'status': rec.get('status'),
            'kind': prob.get('kind'), 'truth_known': bool(prob.get('truth_known', prob.get('truth'))),
            'fit_seconds': rec.get('fit_seconds'), 'total_seconds': rec.get('total_seconds'),
            'front_size': m.get('front_size'), 'success_front': m.get('success_front'),
            'success_selected': m.get('success_selected'), 'hamming_min': m.get('hamming_min'),
            'hamming_selected': m.get('hamming_selected'), 'coef_error': m.get('coef_error'),
            'selected': ' | '.join(rec.get('selected') or []),
            'error': rec.get('error') or rec.get('reason') or '',
        })
    return pd.DataFrame(rows)


def _rate(series):
    """'k/n (p%)' with the Wilson interval; failed runs count as misses."""
    values = series.eq(True)
    k, n = int(values.sum()), int(values.size)
    if n == 0:
        return ''
    lo, hi = wilson_ci(k, n)
    return f'{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})'


def _md_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join([df.index.name or ''] + cols) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append('| ' + ' | '.join([str(idx)] + ['' if v != v else str(v) for v in row]) + ' |')
    return '\n'.join(lines)


def make_report(campaign: Path, formats: str = 'png') -> Path:
    import pandas as pd
    campaign = Path(campaign)
    out = campaign / 'report'
    out.mkdir(exist_ok=True)
    df = load_runs(campaign)
    if df.empty:
        raise SystemExit(f'no runs under {campaign / "runs"}')
    df.to_csv(out / 'runs.csv', index=False)
    known = df[df.truth_known].copy()
    scored = known[known.status.isin(['ok', 'error', 'timeout'])].copy()

    md = [f'# Campaign `{campaign.name}`', '',
          f'Runs: {len(df)} (' + ', '.join(f'{k}: {v}' for k, v in df.status.value_counts().items()) + ').',
          'Success: the true structure (or an accepted equivalent form) is on the final Pareto '
          'front; "pick": the equation chosen from the front without knowing the truth '
          '(compromise) is correct. Cells: k/n (rate, 95 % Wilson interval). Errors and timeouts '
          'count as misses. pending: planned, not yet run; unsupported: the method cannot represent '
          'the problem. These two statuses are excluded from the denominators.', '']

    tables = {}
    for noise, part in scored.groupby('noise'):
        front = part.pivot_table(index='dataset', columns='variant', values='success_front',
                                 aggfunc=_rate)
        pick = part.pivot_table(index='dataset', columns='variant', values='success_selected',
                                aggfunc=_rate)
        front.index.name = pick.index.name = 'dataset'
        tables[noise] = front
        md += [f'## Noise {noise:g} %', '', '### Truth on the Pareto front', '', _md_table(front), '',
               '### Compromise pick is correct', '', _md_table(pick), '']

    long = []
    def median(values):
        values = pd.to_numeric(values, errors='coerce').dropna()
        return values.median() if len(values) else float('nan')

    for (ds, var, noise), planned_part in known.groupby(['dataset', 'variant', 'noise']):
        part = planned_part[planned_part.status.isin(['ok', 'error', 'timeout'])]
        ok = part.success_front.eq(True)
        long.append({'dataset': ds, 'variant': var, 'noise': noise, 'kind': planned_part.kind.dropna().iloc[0]
                     if planned_part.kind.notna().any() else None, 'runs': len(part),
                     'planned': len(planned_part),
                     'pending': int(planned_part.status.eq('pending').sum()),
                     'unsupported': int(planned_part.status.eq('unsupported').sum()),
                     'success_front': ok.mean(),
                     'success_selected': part.success_selected.eq(True).mean(),
                     'median_fit_s': median(part.fit_seconds),
                     'median_coef_error': median(part.coef_error)})
    long = pd.DataFrame(long, columns=['dataset', 'variant', 'noise', 'kind', 'runs', 'planned',
                                         'pending', 'unsupported', 'success_front',
                                         'success_selected', 'median_fit_s', 'median_coef_error'])
    long.to_csv(out / 'success_front.csv', index=False)

    pd.DataFrame(columns=['kind', 'noise', 'variant', 'success_front',
                          'success_selected', 'median_fit_s']).to_csv(out / 'ranking.csv', index=False)
    if not long.empty:
        # Rankings compare the same supported datasets across all variants.
        # Per-dataset cells retain all planned results and support counts.
        common = long.groupby(['dataset', 'noise']).unsupported.transform('sum').eq(0)
        evaluated = long.groupby(['dataset', 'noise']).runs.transform('min').gt(0)
        comparable = long[common & evaluated & (long.runs > 0)].copy()
        long['ranking_comparable'] = common & evaluated
        long.to_csv(out / 'success_front.csv', index=False)
        rank = (comparable.groupby(['kind', 'noise', 'variant'])
                .agg(success_front=('success_front', 'mean'),
                     success_selected=('success_selected', 'mean'),
                     median_fit_s=('median_fit_s', median))
                .reset_index()
                .sort_values(['kind', 'noise', 'success_front', 'success_selected', 'median_fit_s'],
                             ascending=[True, True, False, False, True]))
        rank.to_csv(out / 'ranking.csv', index=False)
        md += ['## Best variant per problem class', '',
               'Mean over the data sets of the class supported by every variant of the per-set '
               'success rate; a data set is left out of the ranking when any variant is '
               'unsupported on it or has no finished runs yet. Ties are broken by the '
               'compromise pick, then by speed.', '']
        best = rank.groupby(['kind', 'noise']).head(1)
        lines = ['| class | noise, % | best variant | success (front) | success (pick) | median time, s |',
                 '|---|---|---|---|---|---|']
        for _, r in best.sort_values(['kind', 'noise'], key=lambda s: s.map(
                {k: i for i, k in enumerate(CLASS_ORDER)}) if s.name == 'kind' else s).iterrows():
            lines.append(f"| {KIND_NAMES.get(r.kind, r.kind)} | {r.noise:g} | {r.variant} | "
                         f"{r.success_front:.0%} | {r.success_selected:.0%} | {r.median_fit_s:.0f} |")
        md += lines + ['']

        overall = (comparable.groupby(['variant', 'noise']).success_front.mean().unstack('noise')
                   .sort_values(by=min(comparable.noise.unique()), ascending=False)) if not comparable.empty else pd.DataFrame()
        overall.index.name = 'variant'
        md += ['## All data sets together (mean success on the front)', '',
               _md_table(overall.map(lambda v: f'{v:.0%}')), '']

    times = df[(df.status == 'ok') & df.fit_seconds.notna()].pivot_table(index='dataset', columns='variant',
                                              values='fit_seconds', aggfunc='median')
    if not times.empty:
        times.index.name = 'dataset'
        md += ['## Median search time, s', '', _md_table(times.round(1)), '']

    unknown = df[~df.truth_known & (df.status == 'ok')]
    if not unknown.empty:
        md += ['## Data sets without a known law: compromise pick', '']
        for _, r in unknown.sort_values(['dataset', 'variant', 'noise', 'seed']).iterrows():
            md.append(f'- **{r.dataset}** / {r.variant} / noise {r.noise:g} / seed {r.seed}: '
                      f'`{r.selected}`')
        md.append('')

    failed = df[df.status.isin(['error', 'timeout'])]
    if not failed.empty:
        md += ['## Failed runs', '']
        md += [f'- {r.dataset} / {r.variant} / noise {r.noise:g} / seed {r.seed}: {r.status} '
               f'{(r.error or "")[:160]}' for _, r in failed.iterrows()]
        md.append('')

    unavailable = df[df.status.isin(['pending', 'unsupported'])]
    if not unavailable.empty:
        md += ['## Pending and unsupported runs', '']
        md += [f'- {r.dataset} / {r.variant} / noise {r.noise:g} / seed {r.seed}: '
               f'{r.status} {r.error}' for _, r in unavailable.iterrows()]
        md.append('')

    (out / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    _figures(long[long.runs > 0] if not long.empty else long, times, out, formats.split(','))
    print(f'report written to {out}')
    return out


def _figures(long, times, out: Path, formats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for noise, part in (long.groupby('noise') if not long.empty else []):
        table = part.pivot_table(index='dataset', columns='variant', values='success_front')
        fig, ax = plt.subplots(figsize=(1.2 + 1.1 * table.shape[1], 0.9 + 0.38 * table.shape[0]))
        im = ax.imshow(table.values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha='right')
        ax.set_yticks(range(table.shape[0]), table.index)
        for (i, j), v in np.ndenumerate(table.values):
            if v == v:
                ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                        color='black' if v > 0.6 else 'white', fontsize=8)
        ax.set_title(f'Truth on the Pareto front, noise {noise:g} %')
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out / f'fig_success_noise{noise:g}.{fmt}', dpi=150)
        plt.close(fig)

    if not times.empty:
        fig, ax = plt.subplots(figsize=(8, 0.9 + 0.35 * times.shape[0]))
        times.plot.barh(ax=ax, logx=True)
        ax.set_xlabel('median fit time, s (log scale)')
        ax.invert_yaxis()
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out / f'fig_time.{fmt}', dpi=150)
        plt.close(fig)
