"""
Aggregator for thesis Section 4.5 smoke / full-run results.

Walks every ``projects/thesis/results/<system>/<pipeline>_rep<NN>.json``
file, groups by (system, pipeline), and writes a markdown summary plus a
JSON snapshot. Metrics per (system, pipeline) cell:

    - structural_success_rate (with Wilson 95% CI)
    - mean Hamming distance
    - consistency_rate (modal-set agreement)
    - mean runtime

Pass ``--root`` to point at a tagged results tree (e.g.
``projects/thesis/results/ablation_v2``) -- the layout is always
``<root>/<system>/*.json``.
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


PIPELINES = ('legacy', 'new')
DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'results')
_CONFIGS_DIR = os.path.join(_THIS_DIR, 'configs')
_TRUTH_CACHE: dict = {}


def _load_truth_eq_alts(system: str):
    """Return ``[primary, *alternatives]`` as a list of equation-string lists.

    Each element is itself a list of equation strings -- the YAML's
    ``truth_equations`` (primary) followed by every ``truth_alternatives``
    entry (each an alternative analytical form). Empty list if the
    config has no ``truth_equations``. Cached per process."""
    if system in _TRUTH_CACHE:
        return _TRUTH_CACHE[system]
    import yaml  # local: only the aggregator needs PyYAML
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


def _unique_history_count(rec: dict, rep_path: str) -> int | None:
    """Count unique candidate-objective vectors seen across the full
    history sidecar (``<rep>.history.json``). Same dedup as the
    candidate-cloud figures: round to 6 sig figs, take ``np.unique``.

    Returns None if the sidecar is missing or unreadable so callers
    can omit this rep from the unique-count aggregation rather than
    treating "no sidecar" as "no candidates"."""
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
    records = defaultdict(lambda: defaultdict(list))  # records[system][pipeline] -> list
    pattern = os.path.join(root, '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        # Skip the sidecar history files (``<rep>.history.json``); they
        # carry the same ``pipeline`` / ``seed`` fields as their parent
        # rep but contain per-epoch candidate trajectories instead of
        # metrics. Counting them would double every rep that had history
        # written.
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        system = rec.get('system') or os.path.basename(os.path.dirname(path))
        pipeline = rec.get('pipeline')
        if pipeline not in PIPELINES:
            continue
        rec['_unique_history'] = _unique_history_count(rec, path)
        records[system][pipeline].append(rec)
    return records


def _mean_std(values: list):
    """Return (mean, std) for ``values`` or (nan, nan) if empty.

    Uses sample stdev (Bessel-corrected) when n >= 2; std is 0.0 for
    singleton cohorts so the M±S formatting still renders cleanly."""
    if not values:
        return float('nan'), float('nan')
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


_EQ_PREFIX_RE = re.compile(r'^\s*[/\\|]\s+')


def _rep_coef_error(rec: dict, truth_alts: list):
    """Per-rep relative coefficient error against the best-matching truth
    alternative (success-only). ``None`` if the rep didn't succeed
    structurally or no usable ``discovered_text`` is stored.

    ``discovered_text`` is N equation strings followed by the rep's
    hparams dict serialised as a trailing string. Coupled systems pre-
    pend each equation with a single box-drawing char (``/``, ``|``,
    ``\\``) which would otherwise be swallowed by the term-coef parser
    as an unparseable prefix and drop the leading term's coefficient
    silently. Filter to eq strings and strip the prefix before invoking
    the metric."""
    if not rec.get('structural_success') or not truth_alts:
        return None
    raw = rec.get('discovered_text') or []
    disc_text = [_EQ_PREFIX_RE.sub('', s).strip()
                 for s in raw if isinstance(s, str) and '=' in s]
    if not disc_text:
        return None
    err = coefficient_error_best(disc_text, truth_alts)
    if err != err:  # nan
        return None
    return err


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
    # ``epoch identified`` only meaningful for successful reps -- on a
    # failed rep discovery_epoch still has a value (the epoch the
    # lowest-Hamming candidate first appeared) but that didn't match
    # the truth and would dilute the average.
    epochs_success = [r['discovery_epoch'] for r in reps
                      if r.get('structural_success')
                      and r.get('discovery_epoch') is not None]
    # Coefficient error: success-only; nonsense on a structurally-wrong
    # rep because there's no canonical partner to compare against.
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


def _format_table(summary: dict) -> str:
    header = (
        '| System | n | Legacy success | Legacy H (mean±std) | Legacy coef err (mean±std) | '
        'NEW success | NEW H (mean±std) | NEW coef err (mean±std) | '
        'runtime L (mean±std) | runtime N (mean±std) |'
    )
    sep = '|---|---|---|---|---|---|---|---|---|---|'
    rows = [header, sep]
    for system in sorted(summary.keys()):
        legacy = summary[system].get('legacy', {'n': 0})
        new = summary[system].get('new', {'n': 0})

        def cell_success(c):
            if c['n'] == 0:
                return '-'
            return f"{c['rate']*100:.0f}% ({c['successes']}/{c['n']})"

        def cell_num(c, key, fmt):
            if c['n'] == 0:
                return '-'
            v = c.get(key)
            if v is None or (isinstance(v, float) and v != v):
                return '-'
            return fmt.format(v)

        def cell_ms(c, mean_key, std_key, fmt, suffix=''):
            if c['n'] == 0:
                return '-'
            m = c.get(mean_key)
            s = c.get(std_key)
            if m is None or (isinstance(m, float) and m != m):
                return '-'
            if s is None or (isinstance(s, float) and s != s):
                return f"{fmt.format(m)}{suffix}"
            return f"{fmt.format(m)}±{fmt.format(s)}{suffix}"

        rows.append(
            f"| {system} | {max(legacy['n'], new['n'])} | "
            f"{cell_success(legacy)} | {cell_ms(legacy, 'mean_hamming', 'std_hamming', '{:.1f}')} | "
            f"{cell_ms(legacy, 'mean_coef_error', 'std_coef_error', '{:.3f}')} | "
            f"{cell_success(new)} | {cell_ms(new, 'mean_hamming', 'std_hamming', '{:.1f}')} | "
            f"{cell_ms(new, 'mean_coef_error', 'std_coef_error', '{:.3f}')} | "
            f"{cell_ms(legacy, 'mean_runtime_sec', 'std_runtime_sec', '{:.1f}', 's')} | "
            f"{cell_ms(new, 'mean_runtime_sec', 'std_runtime_sec', '{:.1f}', 's')} |"
        )
    return '\n'.join(rows)


def _format_dynamics_table(summary: dict) -> str:
    """Per-(system, pipeline) discovery dynamics: unique Pareto-0
    solution count and the epoch at which the truth was identified
    (success-only). Both as mean±std."""
    header = (
        '| System | Legacy unique cands (mean±std) | Legacy epoch identified | '
        'NEW unique cands (mean±std) | NEW epoch identified |'
    )
    sep = '|---|---|---|---|---|'
    rows = [header, sep]

    def cell_ms(c, mean_key, std_key, fmt, suffix=''):
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
        legacy = summary[system].get('legacy', {'n': 0})
        new = summary[system].get('new', {'n': 0})
        rows.append(
            f"| {system} | "
            f"{cell_ms(legacy, 'mean_n_pareto', 'std_n_pareto', '{:.1f}')} | "
            f"{cell_ms(legacy, 'mean_epoch_identified', 'std_epoch_identified', '{:.1f}')} | "
            f"{cell_ms(new, 'mean_n_pareto', 'std_n_pareto', '{:.1f}')} | "
            f"{cell_ms(new, 'mean_epoch_identified', 'std_epoch_identified', '{:.1f}')} |"
        )
    return '\n'.join(rows)


def aggregate(root: str = None) -> dict:
    root = root or DEFAULT_RESULTS_DIR
    records = _load_records(root)
    summary = {
        system: {pipeline: _summarize_cell(reps, system)
                 for pipeline, reps in by_pipeline.items()}
        for system, by_pipeline in records.items()
    }
    return summary


def main(argv=None) -> int:
    # The summary tables contain ``±`` (U+00B1). On Windows
    # PowerShell the default stdout encoding is cp1252 which
    # mangles it; force UTF-8 so the output the user copies
    # into ``new_vs_legacy.md`` keeps the proper glyph.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_RESULTS_DIR,
                        help=f"results root to scan (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument('--out', default=None,
                        help="path for the JSON snapshot (default: <root>/../thesis_summary.json)")
    args = parser.parse_args(argv)

    summary = aggregate(args.root)
    print(_format_table(summary))
    print('\n### Discovery dynamics (mean±std)\n')
    print(_format_dynamics_table(summary))
    out_path = args.out or os.path.join(_THIS_DIR, 'thesis_summary.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
