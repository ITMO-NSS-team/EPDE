"""Solver-free crossed weak discrepancy on fixed archived equations.

Split raw samples into even/odd groups BEFORE any smoothing or term construction.
Fit GCV splines separately, evaluate on the original grid, and rebuild all terms.
Compare paired within-reconstruction and crossed-reconstruction residuals using
identical rows, coefficients, frequency slots and saved dyad scores. No candidate
ODE/PDE is solved. Clean splits and clean full-grid coefficient fits are controls.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import signal_refits as S
import window_profiles as W
import numpy as np
from scipy.interpolate import make_smoothing_spline


def reconstruct(t, observed):
    """Each smoother sees only its own raw observations; no shared tuning."""
    indices = [np.arange(parity, len(t), 2) for parity in (0, 1)]
    assert np.intersect1d(*indices).size == 0
    assert np.union1d(*indices).size == len(t)
    reconstructed = [np.asarray(make_smoothing_spline(t[ix], observed[ix], lam=None)(t)) for ix in indices]
    if not all(np.isfinite(u).all() for u in reconstructed):
        raise ValueError("Nonfinite split reconstruction")
    return reconstructed, indices


def paired_score(item, a, b, index, metric, crossed):
    """Pool both directions; historical cap arm retains its original denominator."""
    coef = np.asarray(item["coefficients"])
    group = np.asarray(item["derivative_group"], dtype=bool)
    yi = index[item["target"]]
    xi = [index[label] for label in item["terms"]]
    pairs = ((a, b), (b, a)) if crossed else ((a, a), (b, b))
    residual_mass = target_mass = net_mass = 0.0
    direction_scores = []
    for target, features in pairs:
        y, x = target[:, yi], features[:, xi]
        r_mass = float(np.abs(y - x @ coef - item["intercept"]).sum())
        y_mass = float(np.abs(y).sum())
        n_mass = float(np.abs(y - x[:, group] @ coef[group]).sum())
        denominator = min(y_mass, n_mass) if metric == "wape_net_cap" else y_mass
        if denominator <= 0:
            raise ValueError("Zero paired denominator")
        direction_scores.append(r_mass / denominator)
        residual_mass += r_mass
        target_mass += y_mass
        net_mass += n_mass
    denominator = min(target_mass, net_mass) if metric == "wape_net_cap" else target_mass
    return dict(discrepancy=residual_mass / denominator, wape=residual_mass / target_mass,
                residual_l1_mean=residual_mass / (2 * len(a)),
                target_l1_mean=target_mass / (2 * len(a)), net_l1_mean=net_mass / (2 * len(a)),
                directional_discrepancies=direction_scores,
                target_split_difference=float(np.abs(a[:, yi] - b[:, yi]).sum()) / target_mass)


def compare(case, fitted_models, full, keep, a, b, index):
    result = {name: dict(models=[]) for name in ("full_baseline", "matched_baseline", "within", "crossed")}
    for template, fitted in zip(case["models"], fitted_models):
        assert template["id"] == fitted["id"]
        item = dict(template, coefficients=fitted["coefficients"], intercept=fitted["intercept"])
        metric = case["metric"]
        original = W.score(item, full, index, metric)
        np.testing.assert_allclose(original["discrepancy"], fitted["objectives"][0], rtol=1e-10, atol=1e-12)
        scores = dict(full_baseline=original, matched_baseline=W.score(item, full[keep], index, metric),
                      within=paired_score(item, a, b, index, metric, False),
                      crossed=paired_score(item, a, b, index, metric, True))
        # A=A must reproduce the established score. Swapping A/B must not
        # change the symmetric score. Within must equal pooled ordinary WAPE.
        np.testing.assert_allclose(paired_score(item, a, a, index, metric, True)["discrepancy"],
                                   W.score(item, a, index, metric)["discrepancy"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scores["crossed"]["discrepancy"],
                                   paired_score(item, b, a, index, metric, True)["discrepancy"], rtol=1e-12)
        np.testing.assert_allclose(scores["within"]["discrepancy"],
                                   W.score(item, np.concatenate([a, b]), index, metric)["discrepancy"], rtol=1e-12)
        for name, score in scores.items():
            result[name]["models"].append(dict(id=item["id"], coefficients=item["coefficients"],
                                               intercept=item["intercept"],
                                               objectives=[score["discrepancy"], fitted["objectives"][1]], **score))
    for value in result.values():
        ref = value["models"][0]
        value["reference_objectives"] = ref["objectives"]
        value["dominators"] = [m["id"] for m in value["models"][1:] if W.dominates(m["objectives"], ref["objectives"])]
    return result


def run_case(case, inputs, output, cache):
    print(case["run"], flush=True)
    S.set_active_config(S.load_search_config(overrides={"equation_terms_max_number": case["width"],
                        "system_rps_mode": "plain", "instability_metric": "dyad"}))
    labels = sorted({label for m in case["models"] for label in [m["target"], *m["terms"]]})
    terms = [W.parse_term(label) for label in labels]
    index = {label: i for i, label in enumerate(labels)}
    reach = W.WF.FD_REACH * max(W.WF.leftover_orders(term.monomials(psi_cap=case["base_plan"]["q"]), 1)[0]
                               for term in terms)
    result = dict(run=case["run"], system=case["system"], metric=case["metric"],
                  target=case["models"][0]["target"], pointwise_derivative_reach=reach, trials=[])
    for trial in case["trials"]:
        path = inputs / f"{case['run']}_seed{trial['seed']}_signal.npz"
        with np.load(path) as saved:
            t, noisy, smoothed, clean = [saved[key].copy() for key in ("t", "noisy", "smoothed", "clean")]
        assert W.digest(noisy) == trial["noise_digest"]
        key = (case["system"], trial["seed"])
        if key not in cache:
            split, training = reconstruct(t, noisy)
            clean_split, _ = reconstruct(t, clean)
            cache[key] = split, training, clean_split
            np.savez_compressed(output / f"{case['system']}_seed{trial['seed']}_splits.npz",
                                t=t, a=split[0], b=split[1], clean_a=clean_split[0], clean_b=clean_split[1],
                                training_a=training[0], training_b=training[1])
        split, training, clean_split = cache[key]
        build = lambda u: W.matrix(terms, W.fields(t, u, trial["frequencies"], terms), case["base_plan"])
        raw, centres = build(noisy)
        smooth, _ = build(smoothed)
        a, _ = build(split[0])
        b, _ = build(split[1])
        # Exclude every weak row whose kernel or pointwise FD stencil touches
        # extrapolated values outside either training grid. Same rows in all arms.
        start = max(ix[0] for ix in training)
        end = min(ix[-1] for ix in training)
        half = case["base_plan"]["ms"][0]
        keep = (centres - half - reach >= start) & (centres + half + reach <= end)
        if keep.sum() < 2:
            raise ValueError("No common interior weak rows")
        record = dict(seed=trial["seed"], frequencies=trial["frequencies"], input_sha256=W.source_hash(path),
                      training_sizes=[len(ix) for ix in training],
                      full_rows=len(centres), matched_rows=int(keep.sum()),
                      matched_centres=centres[keep].tolist(),
                      raw=compare(case, trial["raw"]["models"], raw, keep, a[keep], b[keep], index),
                      smoothed=compare(case, trial["smoothed"]["models"], smooth, keep, a[keep], b[keep], index))
        # These oracle quantities are diagnostics only. They never select
        # smoothing strength, frequencies, coefficients or rows of the main test.
        oracle = S.evaluate(case, t, clean, trial["frequencies"])
        clean_matrix, _ = build(clean)
        ca, _ = build(clean_split[0])
        cb, _ = build(clean_split[1])
        record["clean_control"] = compare(case, oracle["models"], clean_matrix, keep, ca[keep], cb[keep], index)
        record["oracle_field_errors"] = dict(noisy_splits=[S.relative_field_error(u, clean) for u in split],
                                             clean_splits=[S.relative_field_error(u, clean) for u in clean_split])
        for stage in ("raw", "smoothed", "clean_control"):
            counts = [len(record[stage][arm]["dominators"]) for arm in ("full_baseline", "matched_baseline", "within", "crossed")]
            print(f"  seed {trial['seed']} {stage}: full/matched/within/crossed {counts}; rows={keep.sum()}/{len(centres)}", flush=True)
        result["trials"].append(record)
    return result


def write(report, output):
    (output / "cross_reconstruction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Cross-reconstruction weak discrepancy (solver-free)", "",
             "Raw even/odd observations -> independent GCV splines -> original-grid weak columns.",
             "Primary: saved RAW coefficients and dyad. Secondary: saved SMOOTHED coefficients and dyad.",
             "All time/trig terms and per-record frequencies retained. No main-arm coefficient refit.",
             "full = prior full-grid score; matched = same score on the common interior weak rows.",
             "within = pooled (A,A)+(B,B); crossed = pooled (A,B)+(B,A).",
             "Each stage freezes its saved I across these four scores. Clean control uses clean full-grid fits.",
             "The capped historical rational arm preserves its cap; plain WAPE is also stored.", ""]
    for c in report["cases"]:
        lines.append(c["run"])
        for t in c["trials"]:
            lines.append(f"  seed {t['seed']}: {t['matched_rows']}/{t['full_rows']} matched rows, frequencies={t['frequencies']}")
            for stage in ("raw", "smoothed", "clean_control"):
                s = t[stage]
                lines.append(f"    {stage}: dominators " + ", ".join(f"{arm}={len(s[arm]['dominators'])}"
                    for arm in ("full_baseline", "matched_baseline", "within", "crossed")))
                lines.append("      reference D: " + ", ".join(f"{arm}={s[arm]['reference_objectives'][0]:.8g}"
                    for arm in ("full_baseline", "matched_baseline", "within", "crossed")))
                if stage != "clean_control":
                    lines.append(f"      crossed dominators: {s['crossed']['dominators']}")
                if c["system"] == "cauchy":
                    lines.append("      carrier D: " + ", ".join(f"{arm}={s[arm]['models'][1]['objectives'][0]:.8g}"
                        for arm in ("full_baseline", "matched_baseline", "within", "crossed")))
            lines.append(f"    diagnostic relative field errors: {t['oracle_field_errors']}")
        lines.append("")
    lines += ["Limitations:", *["- " + s for s in report["limitations"]]]
    (output / "cross_reconstruction.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    titles = ["Rational", "Kepler", "Rational: capped, no trig (historical control)", "Cauchy"]
    for ax, case, title in zip(axes.flat, report["cases"], titles):
        x = np.arange(len(case["trials"]))
        for offset, arm, color, label in ((-0.27, "matched_baseline", "#777777", "Matched raw baseline"),
                                         (0, "within", "#dea22b", "Within reconstruction"),
                                         (0.27, "crossed", "#2478b4", "Crossed reconstruction")):
            ax.bar(x + offset, [len(t["raw"][arm]["dominators"]) for t in case["trials"]],
                   width=0.25, color=color, label=label)
        ax.set_xticks(x, [str(t["seed"]) for t in case["trials"]])
        ax.set_xlabel("Noise seed")
        ax.set_ylabel("Candidates dominating reference")
        ax.set_title(title)
        ax.yaxis.get_major_locator().set_params(integer=True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Cross the two independently reconstructed records\n"
                 "Saved raw coefficients and dyad frozen; identical weak rows; 5% observation noise")
    fig.savefig(output / "cross_reconstruction_counts.png", dpi=180)
    fig.savefig(output / "cross_reconstruction_counts.svg")
    plt.close(fig)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_dir / "signal_refits.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    report = dict(protocol=__doc__, sources={str(p): W.source_hash(p) for p in [source, Path(__file__),
                  Path(S.__file__), Path(W.__file__), W.REPO / "epde/preprocessing/weak_form.py"]},
                  versions=dict(numpy=np.__version__, scipy=W.scipy.__version__),
                  limitations=["Frozen coefficients and frequencies were selected using the full record: this is a diagnostic, not honest held-out validation.",
                     "Independent raw noise across parity groups is appropriate for these synthetic iid-noise records, not arbitrary correlated measurements.",
                     "Both smoothers have half the original sample density; the within-reconstruction arm controls this preprocessing change.",
                     "Common interior weak rows exclude extrapolation and derivative-stencil boundary effects; original dyad remains frozen.",
                     "No equation solver, evolutionary search, support pruning, or production objective change.",
                     "Clean controls and field-error diagnostics never tune the noisy-data protocol.",
                     "The capped no-trig rational case is a preexisting historical diagnostic arm.",
                     "Only three noise seeds and archived supports; rank gains are not general recovery guarantees."], cases=[])
    cache = {}
    start = time.perf_counter()
    for case in saved["cases"]:
        report["cases"].append(run_case(case, args.input_dir, args.output_dir, cache))
        write(report, args.output_dir)
    report["runtime_seconds"] = time.perf_counter() - start
    write(report, args.output_dir)
    plot(report, args.output_dir)
    print(f"Completed in {report['runtime_seconds']:.1f}s: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
