"""
Aggregator for the thesis Section 4.5 ablation study (2x2x2 factorial).

Walks every ``projects/thesis/results/<system>/<cell>_rep<NN>.json`` file
whose ``pipeline`` field names one of the 8 ablation cells, groups by
(system, cell), and writes a markdown summary plus a JSON snapshot.

If your ablation runs landed under a tag (``--outdir ablation_v2`` ->
``results/ablation_v2/<system>/``), point ``--root`` at that subtree.
Either way the layout is always ``<root>/<system>/*.json``; the 000
(``legacy``) and 111 (``new``) corners are read from the same JSON files
that ``thesis_aggregate.py`` already consumes.

Cell-label semantics (each label lists the NEW components that are ON):

    legacy        000  fitness=L2,   sparsity=LASSO, use_pic=False
    wape          100  fitness=L2LR, sparsity=LASSO, use_pic=False
    instab        010  fitness=L2,   sparsity=LASSO, use_pic=True
    reg           001  fitness=L2,   sparsity=VWSR,  use_pic=False
    wape_instab   110  fitness=L2LR, sparsity=LASSO, use_pic=True
    wape_reg      101  fitness=L2LR, sparsity=VWSR,  use_pic=False
    instab_reg    011  fitness=L2,   sparsity=VWSR,  use_pic=True
    new           111  fitness=L2LR, sparsity=VWSR,  use_pic=True
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from thesis_metrics import (  # noqa: E402
    coefficient_error_best,
    consistency_rate,
    wilson_ci,
)
_CONFIGS_DIR = os.path.join(_THIS_DIR, 'configs')
_TRUTH_CACHE: dict = {}


def _load_truth_eq_alts(system: str):
    """Same shape as :func:`thesis_aggregate._load_truth_eq_alts`: returns
    ``[primary, *alternatives]`` lists of equation strings (or an empty
    list when the system has no YAML truth)."""
    if system in _TRUTH_CACHE:
        return _TRUTH_CACHE[system]
    import yaml
    path = os.path.join(_CONFIGS_DIR, f'{system}.yaml')
    if not os.path.exists(path):
        _TRUTH_CACHE[system] = []
        return []
    with open(path, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    primary = list(cfg.get('truth_equations') or [])
    if not primary:
        _TRUTH_CACHE[system] = []
        return []
    alts = [list(eqs) for eqs in (cfg.get('truth_alternatives') or []) if eqs]
    result = [primary] + alts
    _TRUTH_CACHE[system] = result
    return result


_EQ_PREFIX_RE = re.compile(r'^\s*[/\\|]\s+')


def _rep_coef_error(rec: dict, truth_alts: list):
    """Mirror of :func:`thesis_aggregate._rep_coef_error`: filter the
    trailing hparams string out of ``discovered_text`` and strip the
    coupled-system equation prefix (``/``, ``|``, ``\\``) before calling
    the coefficient-error metric."""
    if not rec.get('structural_success') or not truth_alts:
        return None
    raw = rec.get('discovered_text') or []
    disc_text = [_EQ_PREFIX_RE.sub('', s).strip()
                 for s in raw if isinstance(s, str) and '=' in s]
    if not disc_text:
        return None
    err = coefficient_error_best(disc_text, truth_alts)
    if err != err:
        return None
    return err


# Ordered so the report reads from "all off" to "all on" along each axis.
ABLATION_CELLS = (
    'legacy',
    'wape', 'instab', 'reg',
    'wape_instab', 'wape_reg', 'instab_reg',
    'new',
)

DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


def _unique_history_count(rec: dict, rep_path: str):
    """Count unique candidate-objective vectors in the rep's history
    sidecar; mirrors :func:`thesis_aggregate._unique_history_count`."""
    hist_basename = rec.get('history_path')
    if not hist_basename:
        return None
    hist_path = os.path.join(os.path.dirname(rep_path), hist_basename)
    try:
        with open(hist_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    candidate_history = payload.get('candidate_history') or []
    chunks = []
    import numpy as np
    for snap in candidate_history:
        if not snap:
            continue
        try:
            arr = np.asarray(snap, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2 or arr.size == 0:
            continue
        chunks.append(arr)
    if not chunks:
        return 0
    stacked = np.vstack(chunks)
    rounded = np.unique(np.round(stacked, 6), axis=0)
    return int(rounded.shape[0])


def _load_records(root: str):
    records = defaultdict(lambda: defaultdict(list))  # records[system][cell] -> list
    pattern = os.path.join(root, '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        # Skip ``<rep>.history.json`` sidecars: they carry the same
        # ``pipeline`` / ``system`` fields as their parent rep but hold
        # per-epoch candidate trajectories, not metrics. Counting them
        # doubles every rep that had history written. Same fix as
        # ``thesis_aggregate._load_records``.
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        system = rec.get('system') or os.path.basename(os.path.dirname(path))
        cell = rec.get('pipeline')
        if cell not in ABLATION_CELLS:
            continue
        rec['_unique_history'] = _unique_history_count(rec, path)
        records[system][cell].append(rec)
    return records


def _mean_std(values: list):
    """Mean and sample stdev with NaN-safe fallbacks (std=0 for n=1)."""
    if not values:
        return float('nan'), float('nan')
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


def _summarize_cell(reps: list, system: str = '') -> dict:
    if not reps:
        return {'n': 0}
    truth_alts = _load_truth_eq_alts(system) if system else []
    successes = sum(1 for r in reps if r.get('structural_success'))
    hammings = [r['hamming'] for r in reps if r.get('hamming') is not None]
    runtimes = [r['runtime_sec'] for r in reps if 'runtime_sec' in r]
    # ``unique candidates in history`` matches the figure's deduplicated
    # cloud counts -- one row per distinct objective vector ever
    # explored, not the final Pareto-0 set size.
    n_paretos = [r['_unique_history'] for r in reps
                 if r.get('_unique_history') is not None]
    epochs_success = [r['discovery_epoch'] for r in reps
                      if r.get('structural_success')
                      and r.get('discovery_epoch') is not None]
    coef_errs = [e for e in (_rep_coef_error(r, truth_alts) for r in reps)
                 if e is not None]
    rate = successes / len(reps)
    ci = wilson_ci(successes, len(reps))
    mean_h, std_h = _mean_std(hammings)
    mean_t, std_t = _mean_std(runtimes)
    mean_npar, std_npar = _mean_std(n_paretos)
    mean_ep, std_ep = _mean_std(epochs_success)
    mean_ce, std_ce = _mean_std(coef_errs)
    discovered_tokens = [json.dumps(r.get('discovered_tokens', []), sort_keys=True) for r in reps]
    errors = sum(1 for r in reps if 'error' in r)
    return {
        'n': len(reps),
        'successes': successes,
        'rate': rate,
        'wilson_lo': ci[0],
        'wilson_hi': ci[1],
        'mean_hamming': mean_h,
        'std_hamming': std_h,
        'consistency': consistency_rate(discovered_tokens),
        'mean_runtime_sec': mean_t,
        'std_runtime_sec': std_t,
        'mean_n_pareto': mean_npar,
        'std_n_pareto': std_npar,
        'mean_epoch_identified': mean_ep,
        'std_epoch_identified': std_ep,
        'mean_coef_error': mean_ce,
        'std_coef_error': std_ce,
        'n_coef_error': len(coef_errs),
        'errors': errors,
    }


def _cell_axes(cell: str) -> tuple:
    """Return ``(wape_on, instab_on, reg_on)`` triple for a given cell label."""
    if cell == 'legacy':
        return (False, False, False)
    if cell == 'new':
        return (True, True, True)
    parts = set(cell.split('_'))
    return ('wape' in parts, 'instab' in parts, 'reg' in parts)


def _format_table(summary: dict) -> str:
    header = (
        '| System | Cell | W | I | R | n | Success | H (mean±std) | '
        'coef err (mean±std) | runtime (mean±std) | '
        'unique cands (mean±std) | epoch identified (mean±std) |'
    )
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|'
    rows = [header, sep]

    def _check(b: bool) -> str:
        return 'X' if b else '.'

    def _success(c):
        if c['n'] == 0:
            return '-'
        return f"{c['rate']*100:.0f}% ({c['successes']}/{c['n']})"

    def _num(c, key, fmt):
        if c['n'] == 0:
            return '-'
        v = c.get(key)
        if v is None or (isinstance(v, float) and v != v):
            return '-'
        return fmt.format(v)

    def _ms(c, mean_key, std_key, fmt, suffix=''):
        if c['n'] == 0:
            return '-'
        m = c.get(mean_key)
        s = c.get(std_key)
        if m is None or (isinstance(m, float) and m != m):
            return '-'
        if s is None or (isinstance(s, float) and s != s):
            return f"{fmt.format(m)}{suffix}"
        return f"{fmt.format(m)}±{fmt.format(s)}{suffix}"

    for system in sorted(summary.keys()):
        for cell in ABLATION_CELLS:
            c = summary[system].get(cell, {'n': 0})
            w, i, r = _cell_axes(cell)
            rows.append(
                f"| {system} | {cell} | {_check(w)} | {_check(i)} | {_check(r)} | "
                f"{c['n']} | {_success(c)} | "
                f"{_ms(c, 'mean_hamming', 'std_hamming', '{:.1f}')} | "
                f"{_ms(c, 'mean_coef_error', 'std_coef_error', '{:.3f}')} | "
                f"{_ms(c, 'mean_runtime_sec', 'std_runtime_sec', '{:.1f}', 's')} | "
                f"{_ms(c, 'mean_n_pareto', 'std_n_pareto', '{:.1f}')} | "
                f"{_ms(c, 'mean_epoch_identified', 'std_epoch_identified', '{:.1f}')} |"
            )
    return '\n'.join(rows)


def _format_contributions(summary: dict) -> str:
    """Render the marginal contribution of each axis per system."""
    axes = (
        ('WAPE',   0, [('legacy', 'wape'), ('instab', 'wape_instab'),
                        ('reg', 'wape_reg'), ('instab_reg', 'new')]),
        ('Instab', 1, [('legacy', 'instab'), ('wape', 'wape_instab'),
                        ('reg', 'instab_reg'), ('wape_reg', 'new')]),
        ('Reg',    2, [('legacy', 'reg'), ('wape', 'wape_reg'),
                        ('instab', 'instab_reg'), ('wape_instab', 'new')]),
    )
    rows = ['| System | Axis | mean delta success | mean delta H | n pairs |',
            '|---|---|---|---|---|']
    for system in sorted(summary.keys()):
        for axis_name, _idx, pairs in axes:
            d_rate = []
            d_h = []
            for off_cell, on_cell in pairs:
                off = summary[system].get(off_cell, {'n': 0})
                on = summary[system].get(on_cell, {'n': 0})
                if off['n'] == 0 or on['n'] == 0:
                    continue
                d_rate.append(on['rate'] - off['rate'])
                if (
                    on.get('mean_hamming') is not None
                    and off.get('mean_hamming') is not None
                    and on['mean_hamming'] == on['mean_hamming']
                    and off['mean_hamming'] == off['mean_hamming']
                ):
                    d_h.append(on['mean_hamming'] - off['mean_hamming'])
            if not d_rate:
                rows.append(f"| {system} | {axis_name} | - | - | 0 |")
                continue
            mean_dr = statistics.fmean(d_rate)
            mean_dh = statistics.fmean(d_h) if d_h else float('nan')
            dh_str = f"{mean_dh:+.2f}" if mean_dh == mean_dh else '-'
            rows.append(
                f"| {system} | {axis_name} | {mean_dr*100:+.1f}pp | "
                f"{dh_str} | {len(d_rate)} |"
            )
    return '\n'.join(rows)


def aggregate(root: str = None) -> dict:
    root = root or DEFAULT_RESULTS_DIR
    records = _load_records(root)
    summary = {
        system: {cell: _summarize_cell(reps, system)
                 for cell, reps in by_cell.items()}
        for system, by_cell in records.items()
    }
    return summary


def main(argv=None) -> int:
    # Force UTF-8 stdout so ``±`` survives on Windows PowerShell
    # (default cp1252 mangles it).
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_RESULTS_DIR,
                        help=f"results root to scan (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument('--out', default=None,
                        help="path for the JSON snapshot (default: <thesis>/thesis_ablation_summary.json)")
    args = parser.parse_args(argv)

    summary = aggregate(args.root)
    print('# Thesis Section 4.5 -- Ablation Cells')
    print()
    print(_format_table(summary))
    print()
    print('# Marginal contribution per axis (mean delta across the 4 mutually-exclusive pairs)')
    print()
    print(_format_contributions(summary))
    out_path = args.out or os.path.join(_THIS_DIR, 'thesis_ablation_summary.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
