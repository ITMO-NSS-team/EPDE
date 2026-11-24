"""Exhaust the robot collision token library and evaluate the frozen fronts on new recordings."""
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from discovery import solve
from physics import CUTOFF, LAG, NAMES, OUT, SEEDS, design, partition, recordings
from epde.interface.search_config import active_config, load_search_config, set_active_config
from epde.operators.common.subset_selection import subset_table
from epde.operators.common.survival import dyad_subset_scores


P = len(NAMES)
CONSTANT = 1 << P
MASKS = np.arange(2 * CONSTANT)
LABELS = [*NAMES, 'intercept']
COLS = [[j for j in range(P + 1) if int(m) & (1 << j)] for m in MASKS]
# The five disjoint band vectors add up to push, including when summed over neighbors.
DEPENDENT = sum(1 << NAMES.index(s) for s in ('push', 'band0', 'band1', 'band2', 'band3', 'band4'))
ALIAS = (MASKS & DEPENDENT) == DEPENDENT
REP = np.where(ALIAS, MASKS ^ (1 << NAMES.index('push')), MASKS)


def configure(width):
    cfg = load_search_config()
    space = replace(cfg.search_space, system_rps_mode='plain', equation_terms_max_number=width,
                    equation_factors_max_number=1)
    set_active_config(replace(cfg, search_space=space))
    return space


def pareto_flags(first, second, allowed):
    """Minimize both axes. Identical nondominated objective pairs are all retained."""
    ids = np.flatnonzero(allowed & np.isfinite(first) & np.isfinite(second))
    order = ids[np.lexsort((second[ids], first[ids]))]
    flags = np.zeros(len(first), bool)
    best = np.inf
    for group in np.split(order, np.flatnonzero(np.diff(first[order]) != 0) + 1):
        if len(group) and second[group[0]] < best:
            best = second[group[0]]
            flags[group[second[group] == best]] = True
    return flags


def search(x, y):
    space = configure(P + 2)  # target + fourteen physical columns + optional intercept
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(y))
    np.testing.assert_allclose(x[:, NAMES.index('push')], x[:, -5:].sum(1), rtol=1e-6, atol=2e-7)
    full = np.column_stack((x, w))
    # This identity is the only dependence. All eligible subsets omit at least one member.
    for name in ('push', 'band0', 'band1', 'band2', 'band3', 'band4'):
        independent = np.delete(full, NAMES.index(name), axis=1)
        scaled = independent / np.linalg.norm(independent, axis=0)
        assert np.linalg.matrix_rank(scaled) == len(LABELS) - 1
    ids = MASKS[~ALIAS]
    disc, amp, _, fits = subset_table(x, y, w, subsets=ids.tolist(), discrepancy='wape', return_coefs=True)
    eligible = (ids % CONSTANT != 0) & (amp <= space.rps_amplification_cap)
    dyad = np.full(len(MASKS), np.nan)
    selected = ids[eligible]
    dyad[selected] = dyad_subset_scores(full, y, w, (len(y),), [COLS[m] for m in selected])
    wape, beta = np.empty(len(MASKS)), np.empty((len(MASKS), P + 1))
    wape[ids], beta[ids] = disc, fits
    # Full-rank representatives provide a forecast for algebraically redundant masks.
    wape, beta = wape[REP], beta[REP]
    allowed = np.zeros(len(MASKS), bool)
    allowed[selected] = np.isfinite(dyad[selected]) & (dyad[selected] < np.finfo(float).max)
    front = pareto_flags(wape, dyad, allowed)
    return beta, wape, dyad, allowed, front


def gram(rows):
    rows = np.asarray(rows, float).reshape(-1, rows.shape[-1])
    return rows.T @ rows / len(rows)


def scoring_data(track, params, pair_errors=False):
    frames, x, base, y, _ = design(track, *params, LAG)
    # The screen intercept is one shared scalar on flattened x and y velocity components.
    target = y.astype(float) - base
    a = np.concatenate((x, np.ones((*target.shape, 1)), target[..., None]), axis=-1)
    g = gram(a)
    if not pair_errors:
        return g
    left, right = np.triu_indices(len(track.ids), 1)
    normal_gram = np.zeros_like(g)
    count = 0
    for start in range(0, len(frames), 48):
        sl = slice(start, start + 48)
        pos = track.pos[frames[sl]]
        d = pos[:, right] - pos[:, left]
        radius = np.linalg.norm(d, axis=-1)
        t, jj = np.nonzero((radius > 0) & (radius < CUTOFF))
        normal = d[t, jj] / radius[t, jj, None]
        relative = a[sl][t, right[jj]] - a[sl][t, left[jj]]
        projected = np.einsum('nc,nck->nk', normal, relative)
        normal_gram += projected.T @ projected
        count += len(t)
    return g, normal_gram / count, float(np.var(y, dtype=float)), count


def rmse(b, g):
    v = np.column_stack((b, -np.ones(len(b))))
    return np.sqrt(np.maximum(0., np.sum((v @ g) * v, axis=1)))


def errors(beta, valid, test):
    g, normal, variance, _ = test
    held = rmse(beta, g)
    return dict(validation_rmse=rmse(beta, valid), heldout_rmse=held,
                heldout_r2=1 - held**2 / variance, heldout_pair_normal_rmse=rmse(beta, normal))


def coefficients(b):
    return ' '.join(f'{v:+.12g}*{n}' for n, v in zip(LABELS, b) if v != 0).lstrip('+') or '0'


def terms(mask):
    return '+'.join(LABELS[j] for j in COLS[mask]) or '(none)'


def physical_refits(g):
    beta = np.zeros((CONSTANT, P + 1))
    for mask in range(CONSTANT):
        if REP[mask] == mask:
            beta[mask, :P] = solve(g, [NAMES[j] for j in COLS[mask]])
    return beta[REP[:CONSTANT]]


def as_dict(metrics, i):
    return {k: float(v[i]) for k, v in metrics.items()}


def original_front(folder, form, x, y, valid, test):
    configure(7)
    rows, vectors, masks = [], [], []
    for seed in SEEDS:
        receipt = json.loads((folder / f'{form}_{seed}.json').read_text())
        assert (receipt['form'], receipt['seed'], receipt['epochs']) == (form, seed, 50)
        for idx, eq in enumerate(receipt['native_front']):
            b = np.array([eq['terms'].get(n, 0.) for n in NAMES] + [eq['intercept']])
            mask = sum(1 << j for j in np.flatnonzero(b))
            masks.append(mask)
            vectors.append(b)
            rows.append(dict(heldout=folder.name, form=form, seed=seed, member=idx + 1, mask=mask,
                             terms=terms(mask), equation=coefficients(b), screen_wape=eq['objectives'][0],
                             screen_dyad=eq['objectives'][1]))
    b = np.asarray(vectors)
    full = np.column_stack((x, np.ones(len(y))))
    native_wape = np.abs(full @ b.T - y[:, None]).sum(0) / np.abs(y.astype(float)).sum()
    np.testing.assert_allclose(native_wape, [r['screen_wape'] for r in rows], rtol=2e-6)
    # Reproduce both native objectives on each seed's original screen and slab width.
    unique = sorted(set(masks))
    ds = dyad_subset_scores(full, y, np.ones(len(y)), (len(y),), [COLS[m] for m in unique])
    replay = dict(zip(unique, ds))
    np.testing.assert_allclose([replay[m] for m in masks], [r['screen_dyad'] for r in rows], rtol=2e-5, atol=1e-8)
    scores = errors(b, valid, test)
    for i, row in enumerate(rows):
        row.update(as_dict(scores, i))
    return rows


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(folder, form, z, valid, test, all_rows, fronts):
    x, y = z[form + '_X'], z[form + '_y']
    fronts['nsga2_fronts'].extend(original_front(folder, form, x, y, valid, test))
    beta, wape, dyad, allowed, front = search(x, y)
    scores = errors(beta, valid, test)
    physical = physical_refits(z[form + '_gram'].mean(0))
    refit = errors(physical, valid, test)
    sizes = np.array([len(c) for c in COLS[:CONSTANT]])
    complexity = pareto_flags(sizes, refit['validation_rmse'], ~ALIAS[:CONSTANT])
    # Match the established forecast on the exact same test examples.
    old = json.loads((folder / 'assessment.json').read_text())['models'][form]
    old_mask = sum(1 << NAMES.index(n) for n in old['coef'])
    np.testing.assert_allclose(refit['heldout_rmse'][old_mask], old['heldout']['rmse'], rtol=2e-6)
    np.testing.assert_allclose(refit['heldout_pair_normal_rmse'][old_mask], old['heldout']['pair_rmse'][0], rtol=2e-6)
    for m in range(len(MASKS)):
        row = dict(heldout=folder.name, form=form, mask=m, representative=REP[m], terms=terms(m),
                   size=len(COLS[m]), native_eligible=int(allowed[m]), native_front=int(front[m]),
                   screen_wape=wape[m], screen_dyad=dyad[m] if allowed[m] else '',
                   equation=coefficients(beta[m]), **as_dict(scores, m))
        if front[m]:
            fronts['native_pareto'].append(dict(row))
        full = {}
        if m < CONSTANT:
            full = as_dict(refit, m)
            if complexity[m] or len(COLS[m]) == 1:
                fitted = dict(heldout=folder.name, form=form, mask=m, size=len(COLS[m]),
                              terms=terms(m), equation=coefficients(physical[m]), **full)
                if complexity[m]:
                    fronts['validation_pareto'].append(fitted)
                if len(COLS[m]) == 1:
                    fronts['single_terms'].append(fitted)
        all_rows.writerow([*row.values(), int(complexity[m]) if m < CONSTANT else '',
                           coefficients(physical[m]) if full else '', *[full.get(k, '') for k in refit]])
    print(folder.name, form, 'front sizes', int(front.sum()), int(complexity.sum()), flush=True)


def run():
    dest = OUT / 'pareto'
    dest.mkdir(exist_ok=True)
    tracks = recordings()
    fronts = {name: [] for name in ('nsga2_fronts', 'native_pareto', 'validation_pareto', 'single_terms')}
    header = ['heldout','form','mask','representative','terms','size','native_eligible','native_front',
              'screen_wape','screen_dyad','equation','validation_rmse','heldout_rmse','heldout_r2',
              'heldout_pair_normal_rmse','validation_front','refit_equation','refit_validation_rmse',
              'refit_heldout_rmse','refit_heldout_r2','refit_heldout_pair_normal_rmse']
    previous = active_config()
    try:
        with TemporaryDirectory(dir=dest) as tmp:
            stage = Path(tmp)
            with gzip.open(stage/'all_candidates.csv.gz', 'wt', compresslevel=1, encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for held, track in tracks.items():
                    folder = OUT / held
                    meta = json.loads((folder/'training.json').read_text())
                    assert meta['held'] == held and set(meta['train']) == set(tracks) - {held}
                    lookup = {tuple(map(int, k.split('/'))): np.array(v) for k, v in meta['lookup'].items()}
                    params = lookup, np.array(meta['common']), {int(k): np.array(v) for k, v in meta['arenas'].items()}
                    _, validation, _ = partition(tracks, held)
                    valid = np.mean([scoring_data(t, params) for t in validation], axis=0)
                    test = scoring_data(track, params, pair_errors=True)
                    with np.load(folder / 'prepared.npz') as z:
                        for form in ('strong', 'weak'):
                            evaluate(folder, form, z, valid, test, writer, fronts)
            for name, rows in fronts.items():
                write_rows(stage / (name + '.csv'), rows)
            from merged_pareto import merged_fronts
            merged_fronts(stage/'all_candidates.csv.gz', stage, tracks)
            from pareto_report import report
            report(stage, fronts)
            for path in stage.iterdir():
                path.replace(dest / path.name)
    finally:
        set_active_config(previous)
    print(dest, flush=True)


if __name__ == '__main__':
    from run import check_inputs
    check_inputs()
    run()
