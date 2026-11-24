"""Fixed-structure refits on clean, raw noisy, and GCV-smoothed records.

The smoother sees only (t, observed u); no equation or clean field is supplied.
All nonlinear terms and derivatives are rebuilt from the same chosen field.
Original weak windows, term supports, and intercept presence are held fixed.
Both discrepancy and production dyad are recomputed after each refit.

Example:
    python -B signal_refits.py --input-dir <rho/h> --output-dir signal_results
"""

import argparse
import copy
import csv
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import window_profiles as W
import numpy as np
from scipy.interpolate import make_smoothing_spline

from epde.interface.search_config import load_search_config, set_active_config
from epde.interface.token_frequencies import _spectrum_peaks, _strongest
from epde.operators.common.subset_selection import subset_coefficients
from epde.operators.common.sparsity import instability_scores


def smoothing(t, observed):
    """No truth/candidate inputs, no hand-selected smoothing parameter."""
    curve = make_smoothing_spline(t, observed, lam=None)
    return np.asarray(curve(t), dtype=float)


def coefficients_and_objectives(item, mat, index, metric):
    x = mat[:, [index[label] for label in item["terms"]]]
    y = mat[:, index[item["target"]]]
    weights = np.ones(len(y))
    support = np.array([True] * x.shape[1] + [item["intercept"] != 0])
    coefficients = subset_coefficients(x, y, weights, support)
    fitted = dict(item, coefficients=coefficients[:-1].tolist(), intercept=float(coefficients[-1]))
    discrepancy = W.score(fitted, mat, index, metric)
    per_term = instability_scores("dyad", x, y, weights, (len(y),), support, x.shape[1])
    instability = float(np.sum(per_term))
    if not np.isfinite(instability):
        raise ValueError(f"Non-finite instability for {item['id']}")
    # Check numerical fit quality without introducing a conditioning penalty.
    design = np.column_stack([x, np.ones(len(y))])[:, support]
    residual = y - design @ coefficients[support]
    stationarity = float(np.linalg.norm(design.T @ residual) /
                         max(np.linalg.norm(design) * np.linalg.norm(y), np.finfo(float).tiny))
    return dict(id=item["id"], coefficients=fitted["coefficients"], intercept=fitted["intercept"],
                objectives=[discrepancy["discrepancy"], instability],
                instability_per_term=per_term.tolist(), normal_equation_relative_residual=stationarity,
                **discrepancy)


def evaluate(case, t, values, frequencies):
    labels = sorted({term for m in case["models"] for term in [m["target"], *m["terms"]]})
    terms = [W.parse_term(label) for label in labels]
    index = {label: i for i, label in enumerate(labels)}
    raw = W.fields(t, values, frequencies, terms)
    mat, centres = W.matrix(terms, raw, case["base_plan"])
    rows = [coefficients_and_objectives(m, mat, index, case["metric"]) for m in case["models"]]
    reference = rows[0]
    dominators = [row["id"] for row in rows[1:] if W.dominates(row["objectives"], reference["objectives"])]
    same_target = {m["id"] for m in case["models"] if m["target"] == case["models"][0]["target"]}
    return dict(reference_objectives=reference["objectives"], dominators=dominators,
                same_target_dominators=[name for name in dominators if name in same_target],
                rows=int(len(centres)), models=rows)


def load_case(input_dir, prefix, law, metric):
    paths = [input_dir / (prefix + "_ref.json"), input_dir / ("auto_" + prefix + ".json"),
             input_dir / ("auto_" + prefix + "_form.json")]
    archive, auto, sidecar = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    reference = [v for c in archive["candidates"] if c["name"] == law
                 for v in c.get("versions", []) if v["constant"] == 0]
    if len(reference) != 1:
        raise ValueError("Reference is ambiguous")
    models = [W.model(reference[0], "reference", "reference")]
    models.extend(W.model(m, f"archive_{i}", "archive") for i, m in enumerate(archive["archive"]))
    return dict(run=prefix, system=archive["system"], metric=metric, models=models,
                original_frequencies=W.frequency_map(auto), base_plan=W.parse_plan(sidecar["weak_windows"]["0/u"]),
                width=auto["settings"]["K"], original_noise_digest=sidecar["noise_digest"],
                provenance={str(p): W.source_hash(p) for p in paths})


def relative_field_error(actual, reference):
    # Only a diagnostic after smoothing; never used to select the smoother.
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference - reference.mean()))


def mapped_frequencies(t, observed, original):
    """Keep both frequency slots, estimated from raw observations per seed.

    For a given seed these exact same frequencies are used in raw/smooth arms.
    The archived term names continue to identify the two ordered slots.
    """
    if not original:
        return {}
    peaks = _spectrum_peaks(observed.reshape(1, -1), float(np.median(np.diff(t))))
    chosen = _strongest(peaks, 2, "angular", None)
    if len(chosen) != len(original):
        raise ValueError("Frequency slot count changed")
    return dict(zip(sorted(original, key=original.get), chosen))


def run_case(case, indices, directory, seeds, oracle_only=False):
    print(f"Running {case['run']}", flush=True)
    set_active_config(load_search_config(overrides={"equation_terms_max_number": case["width"],
                                                   "system_rps_mode": "plain", "instability_metric": "dyad"}))
    t, clean = W.generate(case["system"])
    rng = np.random.default_rng([0, indices[case["system"]]])
    noisy0 = clean + 0.05 * clean.std() * rng.standard_normal(clean.shape)
    if W.digest(noisy0) != case["original_noise_digest"]:
        raise ValueError("Original data digest mismatch")
    original = evaluate(case, t, noisy0, case["original_frequencies"])
    expected = np.asarray([m["original_objectives"] for m in case["models"]])
    actual = np.asarray([m["objectives"] for m in original["models"]])
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-6, atol=1e-9)
    expected_dominators = [m["id"] for m in case["models"][1:]
                           if W.dominates(m["original_objectives"], expected[0])]
    if original["dominators"] != expected_dominators:
        raise ValueError("Original Pareto comparison changed")
    oracle = evaluate(case, t, clean, case["original_frequencies"])
    print(f"  archived/raw reproduced ({len(expected)} equations); dominators "
          f"raw={len(original['dominators'])}, clean-refit={len(oracle['dominators'])}", flush=True)
    result = dict(**case, original_raw=original, clean_oracle=oracle,
                  baseline_max_abs_difference=np.abs(actual - expected).max(axis=0).tolist(), trials=[])
    if oracle_only:
        return result
    clean_smoothed = smoothing(t, clean)
    result["clean_smoothed_control"] = evaluate(case, t, clean_smoothed, case["original_frequencies"])
    result["clean_smoothing_relative_error"] = relative_field_error(clean_smoothed, clean)
    for seed in seeds:
        rng = np.random.default_rng([seed, indices[case["system"]]])
        noisy = clean + 0.05 * clean.std() * rng.standard_normal(clean.shape)
        frequencies = mapped_frequencies(t, noisy, case["original_frequencies"])
        if seed == 0 and frequencies:
            np.testing.assert_allclose(list(frequencies.values()),
                                       [case["original_frequencies"][key] for key in frequencies], rtol=1e-14)
        start = time.perf_counter()
        estimated = smoothing(t, noisy)
        smoothing_seconds = time.perf_counter() - start
        if not np.all(np.isfinite(estimated)):
            raise ValueError("Non-finite signal estimate")
        raw = original if seed == 0 else evaluate(case, t, noisy, frequencies)
        processed = evaluate(case, t, estimated, frequencies)
        # Hold the raw dyad coordinates fixed as a second comparison to isolate D.
        fixed_i_dominators = [s["id"] for r, s in zip(raw["models"][1:], processed["models"][1:])
                              if W.dominates([s["objectives"][0], r["objectives"][1]],
                                             [processed["reference_objectives"][0], raw["reference_objectives"][1]])]
        trial = dict(seed=seed, frequencies=frequencies, noise_digest=W.digest(noisy),
                     smoothing_seconds=smoothing_seconds,
                     raw_relative_field_error=relative_field_error(noisy, clean),
                     smoothed_relative_field_error=relative_field_error(estimated, clean),
                     raw=raw, smoothed=processed,
                     smoothed_D_with_raw_I_dominators=fixed_i_dominators)
        result["trials"].append(trial)
        np.savez_compressed(directory / f"{case['run']}_seed{seed}_signal.npz",
                            t=t, clean=clean, noisy=noisy, smoothed=estimated)
        print(f"  seed {seed}: dominators {len(raw['dominators'])} -> {len(processed['dominators'])}; "
              f"D-only {len(fixed_i_dominators)}; signal error {trial['raw_relative_field_error']:.4g} "
              f"-> {trial['smoothed_relative_field_error']:.4g}", flush=True)
    return result


def write_report(report, directory, suffix):
    path = directory / f"signal_refits{suffix}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Fixed-structure refits: original data, clean oracle, and data-only GCV spline", "",
             "The smoother never receives clean data or equations; clean input is a separate control.",
             "Original weak windows/supports preserved; production coefficient fit and dyad recomputed.",
             "D-only holds each raw-data dyad score fixed while replacing discrepancy.",
             "Seeds reuse archived structural templates; frequency slots are selected on each raw record.",
             "No evolutionary search, pruning or automatic front selection is performed.", ""]
    for c in report["cases"]:
        lines += [c["run"], f"  oracle dominators: {len(c['clean_oracle']['dominators'])}; "
                  f"reference (D,I): {c['clean_oracle']['reference_objectives']}"]
        if "clean_smoothed_control" in c:
            lines.append(f"  smoothed-clean dominators: {len(c['clean_smoothed_control']['dominators'])}; "
                         f"signal relative error: {c['clean_smoothing_relative_error']:.6g}")
        for trial in c["trials"]:
            lines += [f"  seed {trial['seed']}: raw {len(trial['raw']['dominators'])}; "
                      f"smoothed {len(trial['smoothed']['dominators'])}; "
                      f"D-only {len(trial['smoothed_D_with_raw_I_dominators'])}",
                      f"    reference raw {trial['raw']['reference_objectives']}; "
                      f"smoothed {trial['smoothed']['reference_objectives']}",
                      f"    signal error {trial['raw_relative_field_error']:.6g} -> "
                      f"{trial['smoothed_relative_field_error']:.6g}"]
            by_id = {m["id"]: m for m in trial["smoothed"]["models"]}
            for identifier in trial["smoothed"]["dominators"]:
                template = next(m for m in c["models"] if m["id"] == identifier)
                lines.append(f"    survivor {identifier} {by_id[identifier]['objectives']}: "
                             f"{template['target']} = {' + '.join(template['terms'])}"
                             + (" + constant" if template["intercept"] else ""))
        lines.append("")
    (directory / f"signal_refits{suffix}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    titles = ["Rational: plain target", "Kepler: inverse tokens", "Rational: keep, no trig (control)",
              "Euler–Cauchy: implicit"]
    for ax, case, title in zip(axes.flat, report["cases"], titles):
        x = np.arange(len(case["trials"]))
        for shift, stage, color, label in ((-0.2, "raw", "#777777", "Raw"),
                                           (0.2, "smoothed", "#2478b4", "GCV-smoothed")):
            ax.bar(x + shift, [len(t[stage]["dominators"]) for t in case["trials"]],
                   width=0.38, color=color, label=label)
        ax.axhline(len(case["clean_oracle"]["dominators"]), linestyle="--", color="#b12a31", label="Clean oracle")
        ax.set_xticks(x, [str(t["seed"]) for t in case["trials"]])
        ax.set_xlabel("Noise seed")
        ax.set_ylabel("Candidates dominating reference")
        ax.set_title(title)
        ax.yaxis.get_major_locator().set_params(integer=True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Refit every structure after estimating the signal · 5% noise\n"
                 "Same weak windows; discrepancy and dyad recomputed; archived candidate templates")
    fig.savefig(directory / "signal_refit_counts.png", dpi=180)
    fig.savefig(directory / "signal_refit_counts.svg")
    plt.close(fig)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=__doc__, smoother="scipy.interpolate.make_smoothing_spline(t, observed, lam=None)",
                  smoother_documentation="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html",
                  source_hashes={str(p): W.source_hash(p) for p in [Path(__file__), Path(W.__file__),
                                 W.REPO / "epde/operators/common/survival.py",
                                 W.REPO / "epde/operators/common/subset_selection.py"]},
                  versions=dict(python=sys.version, numpy=np.__version__, scipy=W.scipy.__version__),
                  limitations=["No search rerun; limited to previously archived structures and reference laws.",
                               "The no-trig rational arm remains a historical diagnostic control.",
                               "Same original weak windows for raw and smoothed arms to isolate preprocessing.",
                               "GCV optimizes field reconstruction, not derivatives or equation recovery.",
                               "The complete original support is refitted; no support pruning is applied.",
                               "Frequency slots are regenerated per noisy record; exact values saved per trial."], cases=[])
    indices = W.seed_indices(args.input_dir / "run_auto.py")
    for prefix, law, metric in W.RUNS:
        case = load_case(args.input_dir, prefix, law, metric)
        report["cases"].append(run_case(case, indices, args.output_dir, args.seeds, args.oracle_only))
    suffix = "_oracle" if args.oracle_only else ""
    write_report(report, args.output_dir, suffix)
    if not args.oracle_only:
        plot(report, args.output_dir)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
