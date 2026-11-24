"""Explain the surviving Kepler structures without solving candidate equations.

Canonicalize exact power/inverse aliases, inspect archived fits, evaluate
pointwise residuals using the known benchmark's analytic orbit, and quantify
the production weak kernel and first-order propagation of iid observation noise.
Analytic orbit quantities are labelled oracle diagnostics, never ranking inputs.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path

import window_profiles as W
import numpy as np


def canonical(label):
    term = W.parse_term(label)
    powers = [0, 0, 0, term.coords[0]]
    for _, orders in term.data:
        powers[orders[0]] += 1
    trig = []
    for token in term.opaque:
        if token.startswith("uinv"):
            powers[0] -= int(token[4:] or 1)
        else:
            trig.append(token)
    return (*powers, tuple(sorted(trig)))


def key_text(key):
    u, d1, d2, t, trig = key
    pieces = []
    for name, power in (("u", u), ("u'", d1), ("u''", d2), ("t", t)):
        if power:
            pieces.append(name if power == 1 else f"{name}^{power}")
    return "*".join(pieces + list(trig)) or "1"


def orbit(t):
    """Analytic Kepler benchmark; algebraic anomaly inversion, no ODE integration.

Initial data r=1, r'=0.5, angular-momentum squared=0.6, gravitational mu=1.
Specific energy=-0.575, a=1/1.15, e=sqrt(0.31), n=a^(-3/2).
"""
    a, e = 1 / 1.15, np.sqrt(0.31)
    n = a ** -1.5
    initial = np.arccos((1 - 1 / a) / e)
    mean = n * t + initial - e * np.sin(initial)
    eccentric = mean.copy()
    for _ in range(12):
        eccentric -= (eccentric - e * np.sin(eccentric) - mean) / (1 - e * np.cos(eccentric))
    np.testing.assert_allclose(eccentric - e * np.sin(eccentric), mean, atol=2e-14, rtol=1e-14)
    u = a * (1 - e * np.cos(eccentric))
    velocity = np.sqrt(a) * e * np.sin(eccentric) / u
    acceleration = -u ** -2 + 0.6 * u ** -3
    return u, velocity, acceleration, dict(semimajor=a, eccentricity=float(e), frequency=n,
                                           period=float(2 * np.pi / n), initial_anomaly=float(initial))


def state_derivative(key, t, u, frequencies):
    p, d1, d2, tp, trig = key
    if d1 or d2:
        raise ValueError("The surviving RHSs should have no derivative factors")
    result = p * u ** (p - 1) * t ** tp if p else np.zeros_like(u)
    for label in trig:
        function, frequency = label.split("~")
        result = result * getattr(np, function)(frequencies[frequency.split("@")[0]] * t)
    return result


def point_matrix(t, u, velocity, acceleration, frequencies, terms):
    fields = W.fields(t, u, frequencies, terms)
    fields._cache[("u", (1,))] = velocity
    fields._cache[("u", (2,))] = acceleration
    return np.column_stack([fields.point(term.data, term.coords, term.opaque) for term in terms])


def fit_model(template, fit):
    return dict(template, coefficients=fit["coefficients"], intercept=fit["intercept"])


def noise_analysis(model, t, u, frequencies, plan, terms, index, sigma):
    """First-order error of the SAME weak pipeline, including common noise.

All surviving RHS columns depend only on u and t, while targets are plain
u' or u''. These allow an exact analytic measurement Jacobian. This is an
oracle diagnostic of noise sensitivity, not a proposed objective normalization.
"""
    order = canonical(model["target"])[1:3]
    derivative_order = 1 if order == (1, 0) else 2 if order == (0, 1) else None
    if derivative_order is None:
        raise ValueError("Unexpected target")
    mat, centres = W.matrix(terms, W.fields(t, u, frequencies, terms), plan)
    m = plan["ms"][0]
    kernel = W.WF.window_kernels(plan["ms"], (float(t[1] - t[0]),), plan["q"], (2,))[0]
    grad = sum(coef * state_derivative(canonical(label), t, u, frequencies)
               for coef, label in zip(model["coefficients"], model["terms"]))
    j_target = np.zeros((len(centres), len(t)))
    j_rhs = np.zeros_like(j_target)
    for row, centre in enumerate(centres):
        sl = slice(centre - m, centre + m + 1)
        j_target[row, sl] = (-1) ** derivative_order * kernel[derivative_order]
        j_rhs[row, sl] = kernel[0] * grad[sl]
    jacobian = j_target - j_rhs
    target = mat[:, index[model["target"]]]
    # Validate against a directional finite difference of the whole weak map.
    direction = np.random.default_rng(2357).standard_normal(len(t))
    epsilon = 1e-6 * u.std()
    def residual(values):
        mm, _ = W.matrix(terms, W.fields(t, values, frequencies, terms), plan)
        return mm[:, index[model["target"]]] - mm[:, [index[v] for v in model["terms"]]] @ np.asarray(model["coefficients"]) - model["intercept"]
    numerical = (residual(u + epsilon * direction) - residual(u - epsilon * direction)) / (2 * epsilon)
    predicted = jacobian @ direction
    relative = float(np.linalg.norm(numerical - predicted) / np.linalg.norm(predicted))
    if relative > 1e-6:
        raise ValueError(f"Noise Jacobian check failed: {relative}")
    std = sigma * np.linalg.norm(jacobian, axis=1)
    return dict(expected_noise_wape_first_order=float(np.sqrt(2 / np.pi) * std.sum() / np.abs(target).sum()),
                residual_noise_rms=float(np.sqrt(np.mean(std ** 2))),
                target_noise_rms=float(sigma * np.sqrt(np.mean(np.sum(j_target ** 2, axis=1)))),
                rhs_noise_rms=float(sigma * np.sqrt(np.mean(np.sum(j_rhs ** 2, axis=1)))),
                target_rms=float(np.sqrt(np.mean(target ** 2))),
                rhs_sensitivity_rms=float(np.sqrt(np.mean(grad ** 2))),
                jacobian_validation_relative_error=relative)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root
    output = root / "structure_results"
    output.mkdir(parents=True, exist_ok=True)
    sources = [root / "signal_results/signal_refits.json", root / "estimator_results/corrected_estimator.json"]
    source = json.loads(sources[0].read_text(encoding="utf-8"))["cases"][1]
    corrected = json.loads(sources[1].read_text(encoding="utf-8"))["cases"][1]
    survivors = corrected["trials"][0]["stages"]["raw"]["methods"]["clsm"]["crossed"]["dominators"]
    groups = {}
    for item in source["models"]:
        if item["id"] not in survivors:
            continue
        signature = (canonical(item["target"]), tuple(sorted(canonical(label) for label in item["terms"])))
        groups.setdefault(signature, []).append(item["id"])
    representatives = [ids[0] for ids in groups.values()]
    assert representatives == ["archive_0", "archive_1", "archive_2", "archive_6"]
    labels = sorted({label for item in source["models"] for label in [item["target"], *item["terms"]]})
    terms = [W.parse_term(label) for label in labels]
    index = {label: i for i, label in enumerate(labels)}
    data_path = root / "signal_results/kepler_trace_PI_e30_n5_weak_kf_seed0_signal.npz"
    with np.load(data_path) as data:
        t, saved_clean = data["t"].copy(), data["clean"].copy()
    clean, velocity, acceleration, orbital = orbit(t)
    np.testing.assert_allclose(clean, saved_clean, rtol=1e-10, atol=1e-11)
    plan = source["base_plan"]
    frequencies = source["original_frequencies"]
    pointwise = point_matrix(t, clean, velocity, acceleration, frequencies, terms)
    weak, centres = W.matrix(terms, W.fields(t, clean, frequencies, terms), plan)
    sigma = 0.05 * float(saved_clean.std())
    report = dict(protocol=__doc__, source_hashes={str(p): W.source_hash(p) for p in [*sources, data_path, Path(__file__),
                      Path(W.__file__), W.REPO / "epde/preprocessing/weak_form.py"]},
                  orbital=orbital, reference_radius_max_abs_difference=float(np.max(np.abs(clean - saved_clean))),
                  weak_plan=plan, known_noise_sigma=sigma, survivors=survivors,
                  groups=[], comparisons=[], validations={},
                  limitations=["Oracle derivatives/orbit are explanatory diagnostics, not a proposed scoring method.",
                      "No candidate ODE/PDE is integrated and no second experimental record is introduced.",
                      "First-order iid-noise propagation omits nonlinear bias, coefficient uncertainty and frequency selection.",
                      "Aliasing reductions are exact algebraic equivalences, not evidence that their four remaining families are exact laws.",
                      "Dynamic sensitivity of one fitted form is explanatory; positive growth is not proposed as a universal rejection rule.",
                      "Analysis covers the archived Kepler competitors, not every candidate encountered during search."])
    for (signature, ids), family in zip(groups.items(), "ABCD"):
        report["groups"].append(dict(family=family, representative=ids[0], members=ids,
                                     target=key_text(signature[0]), terms=[key_text(k) for k in signature[1]]))
    # Verify the groupings on all noisy/smoothed fits and scores, with actual frequencies.
    max_duplicate_score_difference = 0.0
    for trial in corrected["trials"]:
        for stage in ("raw", "smoothed"):
            method = trial["stages"][stage]["methods"]["clsm"]
            assert set(method["crossed"]["dominators"]) == set(survivors)
            by_id = {m["id"]: m for m in method["fit_diagnostics"]}
            for group in report["groups"]:
                representative = by_id[group["representative"]]
                for identifier in group["members"]:
                    current = by_id[identifier]
                    delta = np.max(np.abs(np.array(current["objectives"]) - representative["objectives"]))
                    max_duplicate_score_difference = max(max_duplicate_score_difference, float(delta))
                    np.testing.assert_allclose(current["objectives"], representative["objectives"], rtol=1e-8, atol=1e-11)
                    assert not current["full_fit"]["negative_directions"]
                    assert all(not slab["negative_directions"] for slab in current["slab_fits"])
            report["comparisons"].append(dict(seed=trial["seed"], stage=stage,
                                               models=[by_id[identifier] for identifier in ["reference", *representatives]]))
    report["validations"]["maximum_duplicate_objective_difference"] = max_duplicate_score_difference
    report["validations"]["all_surviving_corrected_systems_positive_semidefinite"] = True
    report["clean_structure_checks"] = []
    for identifier in ["reference", *representatives]:
        template = next(m for m in source["models"] if m["id"] == identifier)
        clean_fit = next(m for m in source["clean_oracle"]["models"] if m["id"] == identifier)
        corrected_fit = next(m for m in report["comparisons"][0]["models"] if m["id"] == identifier)
        fitted = fit_model(template, clean_fit)
        strong_score = W.score(fitted, pointwise, index, "wape")
        weak_score = W.score(fitted, weak, index, "wape")
        np.testing.assert_allclose(weak_score["discrepancy"], clean_fit["objectives"][0], rtol=1e-6, atol=1e-10)
        x = pointwise[:, [index[label] for label in template["terms"]]]
        y = pointwise[:, index[template["target"]]]
        point_coef = np.linalg.lstsq(x, y, rcond=None)[0]
        point_fit = dict(fitted, coefficients=point_coef.tolist(), intercept=0.0)
        residual = y - x @ np.asarray(fitted["coefficients"])
        kernel = W.WF.window_kernels(plan["ms"], (float(t[1]-t[0]),), plan["q"], (0,))
        filtered = W.WF.wint_s(residual, (0,), plan["ms"], plan["ss"], kernel)
        all_contributions = weak[:, [index[label] for label in template["terms"]]] * np.asarray(corrected_fit["coefficients"])
        target_mass = np.abs(weak[:, index[template["target"]]]).sum()
        report["clean_structure_checks"].append(dict(id=identifier, weak_oracle_coefficients=clean_fit["coefficients"],
            pointwise_wape=strong_score["discrepancy"], weak_wape=weak_score["discrepancy"],
            pointwise_ols_refit_wape=W.score(point_fit, pointwise, index, "wape")["discrepancy"],
            pointwise_ols_refit_relative_l2=float(np.linalg.norm(y - x @ point_coef) / np.linalg.norm(y)),
            residual_rms_filter_gain=float(np.sqrt(np.mean(filtered ** 2) / np.mean(residual ** 2))),
            clean_dyad=clean_fit["objectives"][1],
            contribution_mass_over_target=float(np.abs(all_contributions).sum() / target_mass),
            noise=noise_analysis(fit_model(template, corrected_fit), t, clean, frequencies, plan, terms, index, sigma)))
    m, h = plan["ms"][0], float(t[1] - t[0])
    k0 = W.WF.window_kernels(plan["ms"], (h,), plan["q"], (0,))[0][0]
    offsets = np.arange(-m, m + 1) * h
    report["window_filter"] = dict(full_support=2*m*h, support_in_periods=2*m*h/orbital["period"],
        harmonic_gains=[float(abs(k0 @ np.exp(1j * k * orbital["frequency"] * offsets))) for k in range(1, 7)])
    q = np.column_stack([np.ones(len(t)), np.sin(frequencies["1.233"] * t), np.cos(frequencies["1.233"] * t)])
    z = clean ** 3
    zcoef = np.linalg.lstsq(q, z, rcond=None)[0]
    report["cube_radius_sinusoid"] = dict(coefficients=zcoef.tolist(),
        centered_relative_rms_error=float(np.linalg.norm(z-q@zcoef)/np.linalg.norm(z-z.mean())))
    d_fit = next(m for m in report["comparisons"][0]["models"] if m["id"] == "archive_6")
    d_template = next(m for m in source["models"] if m["id"] == "archive_6")
    dcoef = dict(zip(d_template["terms"], d_fit["coefficients"]))
    growth = 3 * dcoef["u"]
    report["family_D_transformation"] = dict(form="z=u^3; z'=3*A*cos(omega*t)+3*B+3*C*z",
        cosine=3*dcoef["cos~1.233@0*uinv2"], constant=3*dcoef["uinv2"], growth_rate=growth,
        deviation_efolding_time=1/growth, deviation_growth_over_record=float(np.exp(growth * (t[-1]-t[0]))))
    edges = corrected["trials"][0]["partitions"][0]
    slabs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        slabs.append(dict(centre_times=[float(t[centres[lo]]),float(t[centres[hi-1]])],
            centre_span_periods=float((t[centres[hi-1]]-t[centres[lo]])/orbital["period"]),
            raw_support_times=[float(t[centres[lo]-m]),float(t[centres[hi-1]+m])]))
    report["dyad_slabs"] = slabs
    report["overlap_time"] = slabs[0]["raw_support_times"][1] - slabs[1]["raw_support_times"][0]
    (output / "kepler_structures.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_text(report, output)
    plot(report, output)
    print(f"Verified {len(survivors)} surviving spellings = {len(groups)} families; saved {output}", flush=True)


def write_text(report, output):
    lines = ["Kepler surviving-structure analysis", "",
             "All ten corrected-fit dominators in all seeds/stages are four distinct algebraic forms.",
             "They all target u' and use the orbital fundamental; the reference targets u''.", ""]
    for group in report["groups"]:
        lines.append(f"{group['family']}: {group['target']} <- {group['terms']}; members={group['members']}")
    lines += ["", f"Orbit: {report['orbital']}", f"Weak kernel: {report['window_filter']}",
              f"Cube-radius sinusoid: {report['cube_radius_sinusoid']}",
              f"Dyad slabs: {report['dyad_slabs']}; overlap={report['overlap_time']}", ""]
    for row in report["comparisons"]:
        lines.append(f"seed {row['seed']} {row['stage']}")
        for item in row["models"]:
            lines.append(f"  {item['id']}: (D,I)={item['objectives']}; coefficients={item['coefficients']}; "
                         f"per-term I={item['instability_per_term']}; slabs={[s['coefficients'] for s in item['slab_fits']]}")
    lines.append("\nClean oracle checks (same weak-fitted coefficients for strong vs weak comparison)")
    for row in report["clean_structure_checks"]:
        lines.append(f"  {row['id']}: strong D={row['pointwise_wape']:.7g}; weak D={row['weak_wape']:.7g}; "
                     f"strong-refit relative L2={row['pointwise_ols_refit_relative_l2']:.7g}; "
                     f"noise expected WAPE={row['noise']['expected_noise_wape_first_order']:.7g}; "
                     f"sum contribution/target={row['contribution_mass_over_target']:.7g}")
        lines.append(f"    noise diagnostic: {row['noise']}")
    lines += ["", f"Family D transformed equation: {report['family_D_transformation']}",
              f"Validation: {report['validations']}", "", "Limitations:", *report["limitations"]]
    (output / "kepler_structures.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines[:14]))


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ["Reference", "A", "B", "C", "D"]
    colors = ["#b12a31", "#1976b9", "#d78c21", "#588d42", "#7954aa"]
    ax = axes[0, 0]
    for name, color, row in zip(labels, colors, report["comparisons"][0]["models"]):
        d, i = row["objectives"]
        ax.scatter(d, i, color=color, s=60)
        ax.annotate(name, (d, i), xytext=(5, 5), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="Discrepancy", ylabel="Dyad",
           title="Seed 0: corrected raw fits (lower is better)")
    ax.grid(alpha=.2)
    ax = axes[0, 1]
    gain = report["window_filter"]["harmonic_gains"]
    ax.bar(np.arange(1, 7), gain, color="#1976b9")
    ax.set(yscale="log", xlabel="Multiple of orbital frequency", ylabel="Amplitude retained",
           title="Production weak window suppresses higher harmonics")
    ax.set_xticks(np.arange(1, 7))
    for k, g in enumerate(gain, 1):
        ax.text(k, g*1.1, f"{100*g:.2g}%", ha="center", fontsize=8)
    ax.set_ylim(3e-4, 2)
    ax = axes[1, 0]
    x = np.arange(5)
    checks = report["clean_structure_checks"]
    ax.bar(x-.18, [r["pointwise_wape"] for r in checks], .35, label="Pointwise, analytic derivatives", color="#777777")
    ax.bar(x+.18, [r["weak_wape"] for r in checks], .35, label="Weak", color="#1976b9")
    ax.set(yscale="log", ylabel="WAPE on clean data", title="Same clean weak-fit coefficients, two evaluations")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(x-.18, [m["objectives"][0] for m in report["comparisons"][0]["models"]], .35,
           label="Observed D, corrected seed-0 fit", color="#777777")
    ax.bar(x+.18, [r["noise"]["expected_noise_wape_first_order"] for r in checks], .35,
           label="First-order iid-noise estimate", color="#1976b9")
    ax.set(ylabel="Relative L1 residual", title="Reference residual is more sensitive to observation noise")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)
    fig.suptitle("Why Kepler's four surrogate families dominate on noisy weak rows\n"
                 "Structural analysis and oracle diagnostics; no candidate-equation integration")
    fig.savefig(output / "kepler_structure_diagnostics.png", dpi=180)
    fig.savefig(output / "kepler_structure_diagnostics.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
