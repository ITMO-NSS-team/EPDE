"""Actual corrected-regression test: whole record AND every production dyad slab.

OLS is compared with covariance-corrected moments (CLS) and mean plus covariance
correction (CLSM). The bootstrap sees only the observed record and its GCV pilot.
Scores use the original weak columns or the existing crossed reconstructions;
the noise estimate is never used as a discrepancy denominator. No equation solve.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import signal_refits as S
import window_profiles as W
import cross_reconstruction as C
import numpy as np
from epde.operators.common import survival as SV


def moments(draws, pilot_matrix):
    """Row-wise mean shifts and joint column covariance, including target noise.

The common pilot is F(s), for both raw F(s+e) and smoothed F(GCV(s+e)). Thus
the latter mean shift includes smoothing bias, not just random perturbation.
No independent-row approximation is needed for the expected Gram correction.
"""
    count = len(draws)
    average = draws.mean(axis=0)
    centered = draws - average
    covariance = np.einsum("brj,brk->rjk", centered, centered) / (count - 1)
    return average - pilot_matrix, covariance, count


def signed_moment_fit(observed, bias, covariance, count, columns, target, rows, method):
    """Solve corrected normal equations, retaining negative eigen-directions.

No PSD clipping, ridge, failure penalty, or OLS fallback is applied. An
indefinite system defines a moment-equation root, not an LS minimum; flag it.
Numerical rank uses production dyad's 1e-12 Gram tolerance. Scaling is from
the observed columns, so a negative corrected diagonal cannot change units.
"""
    values = observed[rows]
    shifted = values - bias[rows] if method == "clsm" else values
    x, y = shifted[:, columns], shifted[:, target]
    g, rhs = x.T @ x, x.T @ y
    scale = np.sqrt(np.sum(values[:, columns] ** 2, axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    if method in ("cls", "clsm"):
        correction = covariance[rows].sum(axis=0)
        multiplier = 1.0 + 1.0 / count if method == "clsm" else 1.0
        g = g - multiplier * correction[np.ix_(columns, columns)]
        rhs = rhs - multiplier * correction[columns, target]
    normalized = (g / scale[:, None]) / scale[None, :]
    normalized = (normalized + normalized.T) / 2
    eigenvalues, vectors = np.linalg.eigh(normalized)
    threshold = SV._DYAD_RCOND * np.max(np.abs(eigenvalues))
    active = np.abs(eigenvalues) > threshold
    inverse = np.divide(1.0, eigenvalues, out=np.zeros_like(eigenvalues), where=active)
    coef = (vectors @ (inverse * (vectors.T @ (rhs / scale)))) / scale
    if not np.isfinite(coef).all():
        raise ValueError("Nonfinite corrected coefficient")
    residual = g @ coef - rhs
    denominator = np.linalg.norm(g) * np.linalg.norm(coef) + np.linalg.norm(rhs)
    return coef, dict(negative_directions=int(np.count_nonzero(eigenvalues < -threshold)),
                       retained_rank=int(active.sum()), width=len(columns),
                       minimum_scaled_eigenvalue=float(eigenvalues.min()),
                       normal_equation_relative_residual=float(np.linalg.norm(residual) / denominator)
                       if denominator > 0 else 0.0)


def fit_equation(item, mat, index, bias, covariance, count, parts, method):
    n, p = mat.shape
    augmented = np.column_stack([mat, np.ones(n)])
    augmented_bias = np.column_stack([bias, np.zeros(n)])
    augmented_covariance = np.zeros((n, p + 1, p + 1))
    augmented_covariance[:, :p, :p] = covariance
    cols = [index[label] for label in item["terms"]]
    intercept = item["intercept"] != 0
    if intercept:
        cols.append(p)
    target = index[item["target"]]
    full, full_diagnostic = signed_moment_fit(augmented, augmented_bias, augmented_covariance,
                                             count, cols, target, slice(None), method)
    total = np.zeros(len(cols))
    slabs = []
    for edges in parts:
        coefficients = []
        for start, end in zip(edges[:-1], edges[1:]):
            fitted, diagnostic = signed_moment_fit(augmented, augmented_bias, augmented_covariance,
                                                   count, cols, target, slice(start, end), method)
            coefficients.append(fitted)
            slabs.append(dict(start=int(start), end=int(end), coefficients=fitted.tolist(), **diagnostic))
        coefficients = np.asarray(coefficients)
        med = SV._median_axis0(coefficients)
        mad = SV._median_axis0(np.abs(coefficients - med))
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = mad / np.abs(med)
        total += np.where(np.isnan(relative), 0.0, relative)
    per_term = np.nan_to_num(total / len(parts))
    instability = float(np.sum(per_term))
    if not np.isfinite(instability):
        raise ValueError("Nonfinite summed dyad score")
    fitted_item = dict(item, coefficients=full[:-1].tolist() if intercept else full.tolist(),
                       intercept=float(full[-1]) if intercept else 0.0)
    score = W.score(fitted_item, mat, index, "wape")
    # Caller supplies the case's original discrepancy metric below.
    return dict(id=item["id"], coefficients=fitted_item["coefficients"], intercept=fitted_item["intercept"],
                objectives=[score["discrepancy"], instability], instability_per_term=per_term.tolist(),
                full_fit=full_diagnostic, slab_fits=slabs)


def method_evaluation(case, mat, index, bias, covariance, count, parts, method, a, b, keep):
    fitted = [fit_equation(item, mat, index, bias, covariance, count, parts, method) for item in case["models"]]
    for item, row in zip(case["models"], fitted):
        row["objectives"][0] = W.score(dict(item, coefficients=row["coefficients"], intercept=row["intercept"]),
                                        mat, index, case["metric"])["discrepancy"]
    scored = C.compare(case, fitted, mat, keep, a[keep], b[keep], index)
    scored["fit_diagnostics"] = fitted
    scored["indefinite_full_fits"] = [m["id"] for m in fitted if m["full_fit"]["negative_directions"]]
    scored["indefinite_slab_fits"] = [m["id"] for m in fitted
                                     if any(s["negative_directions"] for s in m["slab_fits"])]
    reference = fitted[0]
    exact = np.array([W.ANALYTICAL[case["system"]][label] for label in case["models"][0]["terms"]])
    scored["reference_coefficient_relative_error"] = float(np.linalg.norm(np.asarray(reference["coefficients"]) - exact)
                                                            / np.linalg.norm(exact))
    return scored


def run_record(case, trial, tag, cache_dir, splits_dir, expected):
    path = cache_dir / f"{case['system']}_{tag}_bootstrap.npz"
    with np.load(path) as data:
        t, observed, pilot, errors, smoothed_draws = [data[key].copy() for key in
                                                     ("t", "observed", "pilot", "errors", "smoothed_draws")]
        metadata = json.loads(str(data["metadata"]))
        model = dict(sigma=float(data["sigma"]), rho=data["rho"].tolist(),
                     difference_order=data["difference_order"].tolist())
    if metadata["observed_digest"] != W.digest(observed):
        raise ValueError("Bootstrap cache digest mismatch")
    if tag != "clean" and W.digest(observed) != trial["noise_digest"]:
        raise ValueError("Observed field changed")
    labels = sorted({label for item in case["models"] for label in [item["target"], *item["terms"]]})
    index = {label: i for i, label in enumerate(labels)}
    terms = [W.parse_term(label) for label in labels]
    build = lambda u: W.matrix(terms, W.fields(t, u, trial["frequencies"], terms), case["base_plan"])
    pilot_matrix, centres = build(pilot)
    with np.load(splits_dir / f"{case['system']}_seed{trial['seed']}_splits.npz") as split:
        a, _ = build(split["clean_a" if tag == "clean" else "a"])
        b, _ = build(split["clean_b" if tag == "clean" else "b"])
    reach = W.WF.FD_REACH * max(W.WF.leftover_orders(term.monomials(psi_cap=case["base_plan"]["q"]), 1)[0]
                               for term in terms)
    half = case["base_plan"]["ms"][0]
    keep = (centres - half - reach >= 1) & (centres + half + reach <= len(t) - 2)
    parts = SV._dyadic_slab_edges(len(centres), (len(centres),), SV.dyad_t_min_rows(case["width"]))
    result = dict(tag=tag, seed=trial["seed"], frequencies=trial["frequencies"], noise_model=model,
                  cache_sha256=W.source_hash(path), partitions=[edges.tolist() for edges in parts],
                  matched_rows=int(keep.sum()), stages={})
    for stage in ("raw", "smoothed"):
        mat = build(observed)[0] if stage == "raw" else pilot_matrix
        fields = (pilot[None, :] + errors) if stage == "raw" else smoothed_draws
        matrices = np.stack([build(values)[0] for values in fields])
        bias, covariance, count = moments(matrices, pilot_matrix)
        ols = [S.coefficients_and_objectives(item, mat, index, case["metric"]) for item in case["models"]]
        if expected is not None:
            old = np.asarray([m["objectives"] for m in expected[stage]["models"]])
            np.testing.assert_allclose([m["objectives"] for m in ols], old, rtol=1e-7, atol=1e-9)
        block = dict(methods={"ols": C.compare(case, ols, mat, keep, a[keep], b[keep], index)}, sensitivity={})
        # A zero-correction solver control separates numerical normal-equation
        # effects from the estimated mean/covariance corrections.
        zero = method_evaluation(case, mat, index, np.zeros_like(bias), np.zeros_like(covariance), count,
                                 parts, "zero_control", a, b, keep)
        zero_objectives = np.asarray([m["objectives"] for m in zero["fit_diagnostics"]])
        original_objectives = np.asarray([m["objectives"] for m in ols])
        block["zero_correction_max_absolute_objective_difference"] = np.abs(zero_objectives - original_objectives).max(axis=0).tolist()
        block["zero_correction_dominators"] = {arm: zero[arm]["dominators"] for arm in
                                               ("full_baseline", "matched_baseline", "within", "crossed")}
        # Retain the complete numerical control if rank decisions differ.
        block["zero_correction_fit_diagnostics"] = zero["fit_diagnostics"]
        for method in ("cls", "clsm"):
            block["methods"][method] = method_evaluation(case, mat, index, bias, covariance, count, parts,
                                                         method, a, b, keep)
        if tag != "clean":
            for name, selection in (("first_half", slice(None, count // 2)), ("second_half", slice(count // 2, None))):
                mb, mc, mn = moments(matrices[selection], pilot_matrix)
                block["sensitivity"][name] = method_evaluation(case, mat, index, mb, mc, mn, parts, "clsm", a, b, keep)
        result["stages"][stage] = block
        ordinary = [len(block["methods"][m]["matched_baseline"]["dominators"]) for m in ("ols", "cls", "clsm")]
        crossed = [len(block["methods"][m]["crossed"]["dominators"]) for m in ("ols", "cls", "clsm")]
        ref = block["methods"]["clsm"]["full_baseline"]["models"][0]
        print(f"  {tag} {stage}: OLS/CLS/CLSM ordinary={ordinary}, crossed={crossed}; "
              f"reference CLSM (D,I)={ref['objectives']}; "
              f"indefinite full/slabs={len(block['methods']['clsm']['indefinite_full_fits'])}/"
              f"{len(block['methods']['clsm']['indefinite_slab_fits'])}", flush=True)
    return result


def synthetic_check():
    """Exact population-moment check with noise in BOTH x and y, correlated."""
    latent = np.repeat(np.array([-1.0, 0.0, 1.0]), 4)
    ex = np.tile(np.array([-0.5, -0.5, 0.5, 0.5]), 3)
    independent = np.tile(np.array([-0.2, 0.2, -0.2, 0.2]), 3)
    ey = 1.2 * ex + independent
    mat = np.column_stack([latent + ex, 2 * latent + 1 + ey, np.ones(len(latent))])
    covariance = np.zeros((len(latent), 3, 3))
    covariance[:, 0, 0] = 0.25
    covariance[:, 0, 1] = covariance[:, 1, 0] = 0.3
    covariance[:, 1, 1] = 0.4
    fitted, _ = signed_moment_fit(mat, np.zeros_like(mat), covariance, 64, [0, 2], 1, slice(None), "cls")
    np.testing.assert_allclose(fitted, [2, 1], rtol=1e-12, atol=1e-12)
    # Known nonzero column bias and uncertainty of its 64-draw mean estimate.
    bias = np.tile([0.3, -0.2, 0], (len(mat), 1))
    fitted_bias, _ = signed_moment_fit(mat + bias, bias, covariance / (1 + 1 / 64),
                                       64, [0, 2], 1, slice(None), "clsm")
    np.testing.assert_allclose(fitted_bias, [2, 1], rtol=1e-12, atol=1e-12)
    return dict(correlated_EIV_population_coefficients=fitted.tolist(),
                known_mean_shift_coefficients=fitted_bias.tolist())


def write(report, output):
    (output / "corrected_estimator.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Corrected weak regression in the whole-record fit and every dyad slab", "",
             "Actual data-only estimator runs; clean data are separate controls, not oracle substitutions.",
             "OLS = production. CLS = covariance subtraction. CLSM = mean shift + covariance subtraction.",
             "CLSM uses F(pilot) as bootstrap baseline, including smoothing bias in the smoothed pipeline.",
             "G=(X-b_X)'(X-b_X)-(1+1/B)C_XX; rhs=(X-b_X)'(y-b_y)-(1+1/B)C_Xy.",
             "CLS omits the mean shift and the extra 1/B correction. Shared target/feature noise is retained.",
             "Discrepancy is evaluated on observed or split-reconstructed columns, not on corrected moments.",
             "Partitions, MAD/abs(median) aggregation, support, intercept presence and frequency slots retained.",
             "Indefinite corrected systems are flagged and solved as signed moment equations, without clipping.", ""]
    for case in report["cases"]:
        lines.append(case["run"])
        records = case["trials"] + ([case["clean_control"]] if case.get("clean_control") else [])
        for record in records:
            lines.append(f"  {record['tag']}: noise_model={record['noise_model']}; partitions={record['partitions']}")
            for stage, block in record["stages"].items():
                for method, values in block["methods"].items():
                    ref = values["full_baseline"]["models"][0]
                    lines.append(f"    {stage} {method}: ordinary={len(values['matched_baseline']['dominators'])}; "
                                 f"crossed={len(values['crossed']['dominators'])}; reference (D,I)={ref['objectives']}; "
                                 f"coefficients={ref['coefficients']}")
                    if method != "ols":
                        fit = values["fit_diagnostics"][0]
                        lines.append(f"      reference coefficient error={values['reference_coefficient_relative_error']:.6g}; "
                                     f"indefinite full={values['indefinite_full_fits']}; "
                                     f"indefinite in any slab={values['indefinite_slab_fits']}")
                        lines.append(f"      reference slab coefficients={[slab['coefficients'] for slab in fit['slab_fits']]}")
                for half, values in block["sensitivity"].items():
                    lines.append(f"    {stage} CLSM {half}: ordinary={len(values['matched_baseline']['dominators'])}; "
                                 f"crossed={len(values['crossed']['dominators'])}; "
                                 f"reference={values['full_baseline']['reference_objectives']}")
                lines.append(f"    zero-correction max objective difference={block['zero_correction_max_absolute_objective_difference']}")
        lines.append("")
    lines += ["Limitations:", *["- " + line for line in report["limitations"]]]
    (output / "corrected_estimator.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = (len(report["cases"]) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.6 * rows), squeeze=False, constrained_layout=True)
    for ax, case in zip(axes.flat, report["cases"]):
        x = np.arange(len(case["trials"]))
        highest = 1
        for offset, method, arm, color, label in (
                (-0.3, "ols", "matched_baseline", "#777777", "OLS, ordinary"),
                (-0.1, "clsm", "matched_baseline", "#dea22b", "Corrected, ordinary"),
                (0.1, "ols", "crossed", "#81b8db", "OLS, crossed"),
                (0.3, "clsm", "crossed", "#2478b4", "Corrected, crossed")):
            values = [len(t["stages"]["raw"]["methods"][method][arm]["dominators"]) for t in case["trials"]]
            highest = max(highest, *values)
            ax.bar(x + offset, values, width=0.19, color=color, label=label)
        ax.set_xticks(x, [str(t["seed"]) for t in case["trials"]])
        ax.set_xlabel("Noise seed")
        ax.set_ylabel("Candidates dominating reference")
        ax.set_title(case["system"].capitalize() + (": capped/no-trig control" if case["metric"] != "wape" else ""))
        ax.set_ylim(0, 1.15 * highest)
        ax.yaxis.get_major_locator().set_params(integer=True)
    for ax in list(axes.flat)[len(report["cases"]):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Noise-aware coefficient fits: full record and every dyad slab\n"
                 "Raw-data fits; 64 bootstrap draws; unchanged dyad formula; no equation solver")
    fig.savefig(output / "corrected_estimator_counts.png", dpi=180)
    fig.savefig(output / "corrected_estimator_counts.svg")
    plt.close(fig)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_dir / "signal_refits.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    report = dict(protocol=__doc__, synthetic_validation=synthetic_check(),
                  source_hashes={str(p): W.source_hash(p) for p in [Path(__file__), source, Path(S.__file__),
                       Path(W.__file__), Path(C.__file__), Path(SV.__file__),
                       Path(__file__).with_name("estimator_bootstrap.py")]},
                  numerical_gram_rank_tolerance=SV._DYAD_RCOND,
                  limitations=["Parametric bootstrap is conditional on an estimated GCV signal and AR(1) noise model.",
                      "The finite-bootstrap mean-uncertainty correction is approximate because the pilot uses the same observed record.",
                      "Frequencies, supports and window plans were selected from the original full record.",
                      "Indefinite corrected normal matrices have no least-squares minimum; signed moment roots are reported without rescuing them by clipping or ridge.",
                      "Numerical zero-correction controls quantify any differences from production's stabilized slab solver.",
                      "No search rerun or keep-rule implementation change. The same estimator is used in every experimental whole-record/slab fit.",
                      "The dyad formula is unchanged; there is only one admissible two-slab partition at these archived weak-window row counts.",
                      "Covariance is a row-wise expectation for moment correction, not an independent-row confidence model.",
                      "32-draw halves assess Monte Carlo sensitivity; they are not additional observed records.",
                      "Clean records undergo the same data-only noise estimation and bootstrap, separately from noisy records.",
                      "Rational capped/no-trig is a preexisting historical control; no time or trig tokens were removed."], cases=[])
    start = time.perf_counter()
    for case_index in args.cases:
        case = saved["cases"][case_index]
        S.set_active_config(S.load_search_config(overrides={"equation_terms_max_number": case["width"],
                            "system_rps_mode": "plain", "instability_metric": "dyad"}))
        print(case["run"], flush=True)
        entry = dict(run=case["run"], system=case["system"], metric=case["metric"], trials=[], clean_control=None)
        for trial in case["trials"]:
            if trial["seed"] in args.seeds:
                entry["trials"].append(run_record(case, trial, f"seed{trial['seed']}", args.cache_dir, args.splits_dir, trial))
        if 0 in args.seeds:
            entry["clean_control"] = run_record(case, case["trials"][0], "clean", args.cache_dir, args.splits_dir, None)
        report["cases"].append(entry)
        write(report, args.output_dir)
    report["runtime_seconds_excluding_bootstrap"] = time.perf_counter() - start
    write(report, args.output_dir)
    plot(report, args.output_dir)
    print(f"Completed estimator evaluation in {report['runtime_seconds_excluding_bootstrap']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
