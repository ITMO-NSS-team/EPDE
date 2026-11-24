"""Conditional noise propagation through archived weak columns (diagnostic only).

Fit the existing AR(1) noise model to each observed record. Add independent draws
to its data-only GCV signal estimate and rebuild every column, before and after
GCV smoothing. Frequencies, windows and equation coefficients remain fixed.
Preserve the joint column errors; do not normalize discrepancy by their variance.
Clean fields are accessed only after propagation, for labelled oracle diagnostics.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import time

import signal_refits as S
import window_profiles as W
import numpy as np


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def ratio(a, b):
    return float(a / b) if b > 0 else None


def distribution(values):
    values = np.asarray(values)
    return dict(zip(("q05", "median", "q95"), np.quantile(values, [0.05, 0.5, 0.95]).tolist()),
                first_half_median=float(np.median(values[:len(values) // 2])),
                second_half_median=float(np.median(values[len(values) // 2:])))


def contrast(item, fitted, index):
    a = np.zeros(len(index))
    a[index[item["target"]]] += 1
    for label, coefficient in zip(item["terms"], fitted["coefficients"]):
        a[index[label]] -= coefficient
    return a


def propagate(t, signal, noise_model, count, seed, system_index):
    """No clean fields, candidate structures, or residuals are inputs."""
    sigma, rhos, _ = noise_model
    rng = np.random.default_rng([20261116, system_index, seed])
    errors, processed = [], []
    for draw in range(count):
        e = W.WF.ar_noise(rng, signal.shape, sigma, rhos)
        errors.append(e)
        processed.append(S.smoothing(t, signal + e))
        if (draw + 1) % 16 == 0:
            print(f"    propagated {draw + 1}/{count}", flush=True)
    return np.stack(errors), np.stack(processed)


def summarize_stage(case, trial, stage, labels, baseline, observed, draws, clean_matrix):
    index = {label: i for i, label in enumerate(labels)}
    delta = draws - baseline
    bias = np.mean(delta, axis=0)
    centered = delta - bias
    # Covariance preserves dependence across columns, with no assumption of
    # independent overlapping weak rows. It is the mean pointwise covariance.
    covariance = np.einsum("brj,brk->jk", centered, centered) / ((len(draws) - 1) * baseline.shape[0])
    noise_rms = np.sqrt(np.maximum(np.diag(covariance), 0))
    columns = []
    for j, label in enumerate(labels):
        signal_rms = float(rms(baseline[:, j]))
        observed_rms = float(rms(observed[:, j]))
        actual_error = float(rms(observed[:, j] - clean_matrix[:, j]))
        columns.append(dict(label=label, plugin_signal_rms=signal_rms, observed_rms=observed_rms,
                            noise_sd_rms=float(noise_rms[j]), noise_bias_rms=float(rms(bias[:, j])),
                            noise_mse_rms=float(rms(delta[:, :, j])),
                            plugin_signal_to_noise=ratio(signal_rms, noise_rms[j]),
                            observed_to_noise=ratio(observed_rms, noise_rms[j]),
                            oracle_clean_rms=float(rms(clean_matrix[:, j])),
                            oracle_realized_error_rms=actual_error,
                            oracle_error_over_estimated_noise=ratio(actual_error, noise_rms[j])))
    by_label = {c["label"]: c for c in columns}
    models = []
    for item, fitted in zip(case["models"], trial[stage]["models"]):
        assert item["id"] == fitted["id"]
        a = contrast(item, fitted, index)
        residual_delta = np.einsum("brj,j->br", delta, a)
        residual_variance = float(a @ covariance @ a)
        # Compare joint projection to direct replicate residuals; cancellation
        # can be strong, so retain the direct nonnegative computation.
        direct_variance = float(np.mean(np.var(residual_delta, axis=0, ddof=1)))
        np.testing.assert_allclose(residual_variance, direct_variance, rtol=1e-7, atol=1e-15)
        joint = float(np.sqrt(direct_variance))
        independent = float(np.sqrt(np.sum(a * a * np.diag(covariance))))
        target_index = index[item["target"]]
        rhs_a = np.eye(len(index))[target_index] - a
        rhs_variance = float(rhs_a @ covariance @ rhs_a)
        covariance_y_rhs = float(covariance[target_index] @ rhs_a)
        denominator = np.sqrt(max(covariance[target_index, target_index] * rhs_variance, 0))
        fixed = dict(item, coefficients=fitted["coefficients"], intercept=fitted["intercept"])
        discrepancies = [W.score(fixed, mat, index, case["metric"])["discrepancy"] for mat in draws]
        observed_score = W.score(fixed, observed, index, case["metric"])
        np.testing.assert_allclose(observed_score["discrepancy"], fitted["objectives"][0], rtol=1e-10, atol=1e-12)
        models.append(dict(id=item["id"], objectives=fitted["objectives"],
                           target=item["target"], target_support=by_label[item["target"]],
                           feature_support=[by_label[label] for label in item["terms"]],
                           residual_noise_sd_rms=joint,
                           residual_noise_sd_if_independent=independent,
                           joint_over_independent=ratio(joint, independent),
                           noise_correlation_target_rhs=ratio(covariance_y_rhs, denominator),
                           residual_noise_bias_rms=float(rms(bias @ a)),
                           observed_residual_rms=float(rms(observed @ a - fitted["intercept"])),
                           baseline_discrepancy=W.score(fixed, baseline, index, case["metric"])["discrepancy"],
                           simulated_discrepancy_fixed_coefficients=distribution(discrepancies)))
    return dict(columns=columns, models=models), covariance


def carrier_control(case, trial, labels, terms, t, errors, noisy, signal, matrices):
    """Independent pure-noise control; no clean input and no coefficient refit."""
    if case["system"] != "cauchy":
        return None
    index = {label: i for i, label in enumerate(labels)}
    frequencies = trial["frequencies"]
    noise_mats = np.stack([W.matrix(terms, W.fields(t, e, frequencies, terms), case["base_plan"])[0]
                           for e in errors])
    target = case["models"][1]["target"]
    omega = frequencies[re.search(r"sin~(.+)@0", target)[1]]
    h = float(t[1] - t[0])
    theta = omega * h
    fd_frequency = float((8 * np.sin(theta) - np.sin(2 * theta)) / (6 * h))
    selected = [index[label] for label in [case["models"][0]["target"], *case["models"][0]["terms"],
                                          case["models"][1]["target"], *case["models"][1]["terms"]]]
    # These columns are linear in u, including polynomial time factors. A
    # matched-noise control checks exact propagation independently of Monte Carlo.
    linearity = matrices["draws_raw"][:, :, selected] - matrices["base_raw"][None, :, selected]
    np.testing.assert_allclose(linearity, noise_mats[:, :, selected], rtol=1e-8, atol=1e-10)
    out = dict(omega=float(omega), fd4_modified_frequency=fd_frequency,
               linearity_max_abs_error=float(np.max(np.abs(linearity - noise_mats[:, :, selected]))), models=[])
    for item in case["models"][:2]:
        fitted = next(m for m in trial["raw"]["models"] if m["id"] == item["id"])
        fixed = dict(item, coefficients=fitted["coefficients"], intercept=fitted["intercept"])
        scores = [W.score(fixed, mat, index, case["metric"])["discrepancy"] for mat in noise_mats]
        row = dict(id=item["id"], fitted_coefficients=fitted["coefficients"],
                   pure_noise_discrepancy=distribution(scores))
        if item["id"] == "archive_0":
            predicted = dict(item, coefficients=[-fd_frequency], intercept=0.0)
            row["predicted_coefficient_relative_difference"] = abs(fitted["coefficients"][0] + fd_frequency) / fd_frequency
            row["pure_noise_discrepancy_fd_coefficient"] = distribution(
                [W.score(predicted, mat, index, case["metric"])["discrepancy"] for mat in noise_mats])
            # Physical pointwise balance is different from its weakly filtered
            # balance. Evaluate the same frozen carrier on each observed field.
            row["pointwise_wape"] = {}
            from epde.preprocessing.deriv_calculators import _central4
            for name, values in (("raw", noisy), ("smoothed", signal)):
                d1 = _central4(values, t, 0)
                d2 = _central4(d1, t, 0)
                y = d2 * np.sin(omega * t)
                x = d1 * np.cos(omega * t)
                coefficient = next(m for m in trial[name]["models"] if m["id"] == item["id"])["coefficients"][0]
                # Exclude both nested stencils' boundary reach.
                interior = slice(4, -4)
                row["pointwise_wape"][name] = float(np.abs(y[interior] - coefficient * x[interior]).sum() /
                                                     np.abs(y[interior]).sum())
        out["models"].append(row)
    return out


def run_case(case, inputs, count, cache, seeds):
    labels = sorted({label for m in case["models"] for label in [m["target"], *m["terms"]]})
    terms = [W.parse_term(label) for label in labels]
    result = dict(run=case["run"], system=case["system"], metric=case["metric"], trials=[])
    for trial in case["trials"]:
        if trial["seed"] not in seeds:
            continue
        print(f"{case['run']} seed {trial['seed']}", flush=True)
        path = inputs / f"{case['run']}_seed{trial['seed']}_signal.npz"
        with np.load(path) as saved:
            t, noisy, signal = [saved[name].copy() for name in ("t", "noisy", "smoothed")]
        assert W.digest(noisy) == trial["noise_digest"]
        noise_model = W.WF.noise_model(noisy)
        if noise_model[0] <= 0:
            raise ValueError("No noise estimate available")
        key = (case["system"], trial["seed"])
        if key not in cache:
            system_index = {"rational": 0, "kepler": 1, "cauchy": 2}[case["system"]]
            cache[key] = propagate(t, signal, noise_model, count, trial["seed"], system_index)
        errors, processed = cache[key]
        build = lambda u: W.matrix(terms, W.fields(t, u, trial["frequencies"], terms), case["base_plan"])[0]
        matrices = dict(base_raw=build(signal), base_smoothed=build(S.smoothing(t, signal)),
                        observed_raw=build(noisy), observed_smoothed=build(signal),
                        draws_raw=np.stack([build(signal + e) for e in errors]),
                        draws_smoothed=np.stack([build(u) for u in processed]))
        # Clean data enter only the labelled diagnostics below, after all
        # artificial fields/columns, hyperparameters and noise estimates exist.
        with np.load(path) as saved:
            clean = saved["clean"].copy()
        clean_matrix = build(clean)
        record = dict(seed=trial["seed"], frequencies=trial["frequencies"],
                      input_sha256=W.source_hash(path), noise_model=dict(sigma=noise_model[0],
                        rho=noise_model[1], difference_order=noise_model[2]),
                      oracle_noise_sigma=0.05 * float(clean.std()), stages={})
        for stage in ("raw", "smoothed"):
            record["stages"][stage], covariance = summarize_stage(
                case, trial, stage, labels, matrices[f"base_{stage}"], matrices[f"observed_{stage}"],
                matrices[f"draws_{stage}"], clean_matrix)
            record["stages"][stage]["covariance"] = covariance.tolist()
            ref = record["stages"][stage]["models"][0]
            carrier = record["stages"][stage]["models"][1]
            print(f"  {stage}: reference target signal/noise "
                  f"{ref['target_support']['plugin_signal_to_noise']:.4g}; archive_0 "
                  f"{carrier['target_support']['plugin_signal_to_noise']:.4g}; "
                  f"joint/independent {carrier['joint_over_independent']:.4g}", flush=True)
        record["carrier_control"] = carrier_control(case, trial, labels, terms, t, errors, noisy, signal, matrices)
        if record["carrier_control"]:
            carrier = record["carrier_control"]["models"][1]
            print(f"  pure-noise carrier WAPE median={carrier['pure_noise_discrepancy']['median']:.5g}; "
                  f"FD coefficient relative difference={carrier['predicted_coefficient_relative_difference']:.5g}", flush=True)
        result["trials"].append(record)
    return result


def write(report, directory):
    (directory / "noise_support.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Conditional column noise support and common-noise cancellation", "",
             f"{report['draws']} artificial draws per record; no refitting, selection, objective change, or search.",
             "Noise estimated from observed data; synthetic records centered on its GCV estimate.",
             "Raw baseline: F(signal estimate). Smoothed baseline: F(GCV(signal estimate)).",
             "Ratios below are RMS signal / noise standard deviation, not SNR power or a calibrated test.",
             "Columns share each noise draw. Frequencies and weak windows stay fixed.", ""]
    for c in report["cases"]:
        lines.append(c["run"])
        for t in c["trials"]:
            nm = t["noise_model"]
            lines.append(f"  seed {t['seed']}: sigma_hat={nm['sigma']:.7g}, "
                         f"known_sigma={t['oracle_noise_sigma']:.7g}, rho={nm['rho']}")
            for stage, result in t["stages"].items():
                # Report reference and archive_0 in detail; all candidates remain in JSON.
                for m in result["models"][:2]:
                    s = m["target_support"]
                    lines.append(f"    {stage} {m['id']}: D={m['objectives'][0]:.7g}, "
                                 f"target signal/noise={s['plugin_signal_to_noise']:.7g}, "
                                 f"target observed/noise={s['observed_to_noise']:.7g}, "
                                 f"joint/independent={m['joint_over_independent']:.7g}, "
                                 f"noise correlation y,RHS={m['noise_correlation_target_rhs']:.9g}")
                    lines.append(f"      target oracle actual error / estimated noise="
                                 f"{s['oracle_error_over_estimated_noise']:.7g}; "
                                 f"frozen-coefficient simulated D {m['simulated_discrepancy_fixed_coefficients']}")
            if t["carrier_control"]:
                control = t["carrier_control"]
                lines.append(f"    omega={control['omega']:.8g}; FD4 response={control['fd4_modified_frequency']:.8g}")
                for m in control["models"]:
                    lines.append(f"    pure-noise {m['id']}: {m['pure_noise_discrepancy']}")
                    if m["id"] == "archive_0":
                        lines.append(f"      fitted coefficient={m['fitted_coefficients']}; "
                                     f"relative FD difference={m['predicted_coefficient_relative_difference']:.7g}; "
                                     f"pointwise WAPE={m['pointwise_wape']}")
        lines.append("")
    lines += ["Limitations:", *["- " + s for s in report["limitations"]]]
    (directory / "noise_support.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, case in zip(axes, [report["cases"][i] for i in (0, 1, 3)]):
        for stage, marker in (("raw", "o"), ("smoothed", "s")):
            for role, color, name in ((0, "#b12a31", "Reference"), (1, "#2478b4", "Archive 0")):
                models = [t["stages"][stage]["models"][role] for t in case["trials"]]
                ax.scatter([m["target_support"]["plugin_signal_to_noise"] for m in models],
                           [m["joint_over_independent"] for m in models], marker=marker, color=color,
                           label=f"{name}, {stage}", s=45, alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1, color="#999999", linestyle="--", linewidth=0.8)
        ax.axhline(1, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_title(case["system"].capitalize())
        ax.set_xlabel("Target: estimated signal RMS / noise SD")
        ax.set_ylabel("Residual noise SD: joint / independent")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Conditional noise propagation: weak columns share the same measurement noise\n"
                 "Lower y means stronger cancellation; artificial replicates, fixed frequencies and coefficients")
    fig.savefig(directory / "noise_support.png", dpi=180)
    fig.savefig(directory / "noise_support.svg")
    plt.close(fig)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.draws < 4:
        parser.error("At least four draws are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_dir / "signal_refits.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    report = dict(protocol=__doc__, draws=args.draws, seeds=args.seeds,
                  source_sha256={str(p): W.source_hash(p) for p in [source, Path(__file__), Path(S.__file__),
                      Path(W.__file__), W.REPO / "epde/preprocessing/weak_form.py",
                      W.REPO / "epde/preprocessing/deriv_calculators.py"]},
                  limitations=["Artificial replicates, not independent experimental records.",
                      "Conditional on an estimated signal and an AR(1) noise model; plug-in bias remains possible.",
                      "GCV is reselected per replicate; frequencies/windows/supports are held fixed.",
                      "Joint covariance is averaged over rows; no independent-row significance test is performed.",
                      "No arbitrary signal/noise cutoff is applied and no score is proposed or selected here.",
                      "Clean oracle diagnostics are not inputs to propagation or ranking.",
                      "Archived templates only; no search rerun or coefficient-refit bootstrap.",
                      "Monte Carlo quantiles describe artificial draws, not calibrated confidence intervals."], cases=[])
    cache = {}
    start = time.perf_counter()
    for case in saved["cases"]:
        report["cases"].append(run_case(case, args.input_dir, args.draws, cache, args.seeds))
        write(report, args.output_dir)
    report["runtime_seconds"] = time.perf_counter() - start
    write(report, args.output_dir)
    plot(report, args.output_dir)
    print(f"Completed in {report['runtime_seconds']:.1f}s: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
