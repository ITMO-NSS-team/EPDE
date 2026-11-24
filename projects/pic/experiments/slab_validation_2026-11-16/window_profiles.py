"""Re-evaluate archived equations at weak-window widths with frozen coefficients.

Run with the original EPDE_3_14 environment, one BLAS thread:
    python -B window_profiles.py --input-dir <Claude rho/h> --output-dir <folder>

No search or coefficient fitting occurs. Every archived equation is checked
against its original WAPE before comparing admissible windows on common centres.
The clean field and analytical coefficients are diagnostic controls only.
"""

import argparse
import ast
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from epde.preprocessing import weak_form as WF
from epde.operators.common.survival import row_floor


RUNS = (
    ("rational_trace_PW_e30_n5_weak_kf", "exact", "wape"),
    ("kepler_trace_PI_e30_n5_weak_kf", "exact_inv", "wape"),
    ("rational_trace_PK_e30_n5_weak_cap_kf_notrig", "exact", "wape_net_cap"),
    ("cauchy_trace_FW_e30_n5_weak_proff_kf", "exact", "wape"),
)
ANALYTICAL = {
    "rational": {"du/dx0*u^2": -1.0, "u": -1.0},
    "kepler": {"uinv2": -1.0, "uinv3": 0.6},
    "cauchy": {"du/dx0*t@0": -1.0, "u": -16.0},
}


def digest(values):
    h = hashlib.blake2b(digest_size=12)
    h.update(b"u")
    h.update(np.ascontiguousarray(values, dtype=float).tobytes())
    return h.hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(name):
    """Exact copy of the three generator formulas/settings in run_auto._load."""
    if name == "rational":
        t = np.linspace(0.0, 8.0, 800)
        sol = solve_ivp(lambda _, y: [-y[0] / (1.0 + y[0] ** 2)],
                        (t[0], t[-1]), [3.0], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        return t, sol.y[0]
    if name == "kepler":
        t = np.linspace(0.0, 30.0, 1500)
        sol = solve_ivp(lambda _, y: [y[1], -y[0] ** -2.0 + 0.6 * y[0] ** -3.0],
                        (t[0], t[-1]), [1.0, 0.5], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        return t, sol.y[0]
    if name == "cauchy":
        t = np.linspace(1.0, 20.0, 1000)
        return t, np.cos(4 * np.log(t)) + 0.5 * np.sin(4 * np.log(t))
    raise ValueError(name)


def seed_indices(harness):
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    truth = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "TRUTH" for t in node.targets))
    return {ast.literal_eval(key): i for i, key in enumerate(truth.keys)}


def frequency_map(auto):
    """Restore full precision from equation text; rounded labels are not enough."""
    frequencies = {float(v) for level in auto["levels"] for record in level
                   for v in re.findall(r"freq: ([0-9.eE+\-]+)", record["text"])}
    out = {}
    for frequency in sorted(frequencies):
        label = "1" if abs(frequency - 1.0) < 0.01 else f"{frequency:.4g}"
        if label in out and out[label] != frequency:
            raise ValueError(f"Ambiguous rounded frequency {label}")
        out[label] = frequency
    return out


def parse_term(label):
    data, coords, opaque = [], [0], []
    for factor in label.split("*"):
        if match := re.fullmatch(r"u(?:\^(\d+))?", factor):
            data.extend([("u", (0,))] * int(match[1] or 1))
        elif match := re.fullmatch(r"du/dx0(?:\^(\d+))?", factor):
            data.extend([("u", (1,))] * int(match[1] or 1))
        elif match := re.fullmatch(r"d\^2u/dx0\^2(?:\^(\d+))?", factor):
            data.extend([("u", (2,))] * int(match[1] or 1))
        elif match := re.fullmatch(r"t(?:\^(\d+))?@0", factor):
            coords[0] += int(match[1] or 1)
        elif re.fullmatch(r"(?:sin|cos)~[0-9.eE+\-]+@0|uinv[23]?", factor):
            opaque.append(factor)
        else:
            raise ValueError(f"Unsupported factor: {factor}")
    return WF.WeakTerm(data, coords, opaque)


def fields(t, u, frequencies, terms):
    opaque = {}
    for term in terms:
        for label in term.opaque:
            if label.startswith("uinv"):
                power = int(label[4:] or 1)
                opaque[label] = 1.0 / u if power == 1 else u ** -float(power)
            else:
                match = re.fullmatch(r"(sin|cos)~(.+)@0", label)
                opaque[label] = getattr(np, match[1])(frequencies[match[2]] * t)
    return WF.Fields({"u": u}, [t], opaque)


def parse_plan(text):
    def value(name):
        return ast.literal_eval(re.search(rf"\b{name}=(\([^)]*\)|\d+)", text)[1])
    return dict(j=value("j"), q=value("q"), ms=value("m"), ss=value("s"), cut=value("cut"),
                shape=value("rows"))


def model(record, identifier, role):
    entries = record["diag"]["terms"]
    if set(e[0] for e in entries) != set(record["terms"]):
        raise ValueError("Saved coefficient/term mismatch")
    return dict(id=identifier, role=role, target=record["target"],
                terms=[e[0] for e in entries], coefficients=[e[1] for e in entries],
                derivative_group=[e[3] for e in entries], intercept=record["constant"],
                original_objectives=record.get("obj", record.get("values")),
                saved_wape=record["diag"]["wape"])


def matrix(terms, raw, plan):
    monos = [term.monomials(psi_cap=plan["q"]) for term in terms]
    mat, shape = WF.weak_matrix(monos, [t.opaque for t in terms], raw,
                               plan["cut"], plan["ms"], plan["ss"], plan["q"])
    centres = plan["cut"][0] + plan["ms"][0] + np.arange(shape[0]) * plan["ss"][0]
    if not np.all(np.isfinite(mat)):
        raise ValueError("Non-finite weak columns")
    return mat, centres


def score(model, mat, index, metric):
    y = mat[:, index[model["target"]]]
    x = mat[:, [index[t] for t in model["terms"]]]
    coefficients = np.asarray(model["coefficients"])
    residual = y - x @ coefficients - model["intercept"]
    numerator = float(np.abs(residual).sum())
    target_mass = float(np.abs(y).sum())
    group = np.asarray(model["derivative_group"], dtype=bool)
    net_mass = float(np.abs(y - x[:, group] @ coefficients[group]).sum())
    denominator = min(target_mass, net_mass) if metric == "wape_net_cap" else target_mass
    if denominator <= 0:
        raise ValueError("Zero discrepancy denominator")
    return dict(discrepancy=numerator / denominator, wape=numerator / target_mass,
                residual_l1_mean=numerator / len(y), target_l1_mean=target_mass / len(y),
                net_l1_mean=net_mass / len(y))


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def audit_case(input_dir, output_dir, prefix, law_name, metric, indices):
    print(f"Reconstructing {prefix}", flush=True)
    paths = [input_dir / (prefix + "_ref.json"), input_dir / ("auto_" + prefix + ".json"),
             input_dir / ("auto_" + prefix + "_form.json")]
    archive, auto, sidecar = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    name = archive["system"]
    t, clean = generate(name)
    rng = np.random.default_rng([sidecar["seed"] or 0, indices[name]])
    noisy = clean + sidecar["noise"] * clean.std() * rng.standard_normal(clean.shape)
    if digest(noisy) != sidecar["noise_digest"]:
        raise ValueError(f"Noise/data digest mismatch for {name}: {digest(noisy)} != {sidecar['noise_digest']}")
    frequencies = frequency_map(auto)
    base_plan = parse_plan(sidecar["weak_windows"]["0/u"])
    reference = [version for cand in archive["candidates"] if cand["name"] == law_name
                 for version in cand.get("versions", []) if version["constant"] == 0]
    if len(reference) != 1:
        raise ValueError("Expected one reference law without a fitted intercept")
    models = [model(reference[0], "reference", "reference")]
    models.extend(model(row, f"archive_{i}", "archive") for i, row in enumerate(archive["archive"]))
    analytic = dict(models[0], id="analytical", role="analytical_control",
                    coefficients=[ANALYTICAL[name][term] for term in models[0]["terms"]])
    models.append(analytic)
    labels = sorted({term for m in models for term in [m["target"], *m["terms"]]})
    terms = [parse_term(label) for label in labels]
    index = {label: i for i, label in enumerate(labels)}
    raw = fields(t, noisy, frequencies, terms)
    clean_raw = fields(t, clean, frequencies, terms)
    base_matrix, base_centres = matrix(terms, raw, base_plan)
    reproduction = []
    for item in models[:-1]:
        value = score(item, base_matrix, index, metric)
        np.testing.assert_allclose(value["wape"], item["saved_wape"], rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(value["discrepancy"], item["original_objectives"][0], rtol=1e-9, atol=1e-11)
        reproduction.append(abs(value["wape"] - item["saved_wape"]))

    # The actual one-factor selection library, in pool order; no frequency reselection.
    library = [parse_term(label) for label in ("u", "du/dx0", "d^2u/dx0^2", "t@0")]
    width = auto["settings"]["K"]
    plan = WF.choose_window(WF.Fields({"u": noisy}, [t]), library, [1, 2],
                            np.random.default_rng([0, 0]),
                            rows_ok=lambda shape: row_floor("dyad", shape, width) is None,
                            gate=WF.HALF_POWER)
    for key in ("j", "q", "ms", "ss", "cut", "shape"):
        if getattr(plan, key) != base_plan[key]:
            raise ValueError(f"Reconstructed window differs: {key}: {getattr(plan, key)} != {base_plan[key]}")
    # Change width only, keeping test-function smoothness fixed at the original value.
    candidates = []
    for j, ms, ss, shape, selection_score in plan.table:
        if j != plan.j:
            continue
        share = plan.shares.get((j, ms))
        admissible = bool(np.isfinite(selection_score) and share is not None and share >= WF.HALF_POWER)
        candidates.append(dict(j=j, q=plan.q, ms=ms, ss=ss, cut=plan.cut, shape=shape,
                               selection_score=float(selection_score) if np.isfinite(selection_score) else None,
                               kept_share=share, admissible=admissible))
    eligible = sorted([p for p in candidates if p["admissible"]], key=lambda p: p["ms"])
    if len(eligible) < 2:
        raise ValueError(f"Only {len(eligible)} admissible widths for {name}")
    matrices = [matrix(terms, raw, p) for p in eligible]
    common = base_centres.copy()
    for _, centres in matrices:
        common = np.intersect1d(common, centres)
    if common.size < 2:
        raise ValueError("No common evaluation centres")
    baseline_dominators = [m["id"] for m in models if m["role"] == "archive"
                           and dominates(m["original_objectives"], models[0]["original_objectives"])]
    evaluations = {m["id"]: [] for m in models}
    windows = []
    for p, (mat, centres) in zip(eligible, matrices):
        positions = np.searchsorted(centres, common)
        np.testing.assert_array_equal(centres[positions], common)
        mat = mat[positions]
        cm, cc = matrix(terms, clean_raw, p)
        np.testing.assert_array_equal(cc, centres)
        cm = cm[positions]
        # The weak constant stays one at every width (intercepts must not be rescaled).
        ones, _ = WF.weak_matrix([WF.WeakTerm([], (0,)).monomials(psi_cap=p["q"])], [()],
                                 raw, p["cut"], p["ms"], p["ss"], p["q"])
        np.testing.assert_allclose(ones, 1.0, rtol=0, atol=1e-12)
        for item in models:
            evaluations[item["id"]].append(dict(
                half_width_samples=p["ms"][0], noisy=score(item, mat, index, metric),
                clean=score(item, cm, index, metric)))
        law_d = evaluations["reference"][-1]["noisy"]["discrepancy"]
        dominators = [item["id"] for item in models if item["role"] == "archive" and dominates(
            [evaluations[item["id"]][-1]["noisy"]["discrepancy"], item["original_objectives"][1]],
            [law_d, models[0]["original_objectives"][1]])]
        windows.append(dict(**p, half_width_physical=p["ms"][0] * float(t[1] - t[0]),
                            common_rows=int(common.size), dominators=dominators,
                            baseline_dominators_remaining=[x for x in baseline_dominators if x in dominators]))
    all_scale_dominators = [m["id"] for m in models if m["role"] == "archive" and dominates(
        [v["noisy"]["discrepancy"] for v in evaluations[m["id"]]] + [m["original_objectives"][1]],
        [v["noisy"]["discrepancy"] for v in evaluations["reference"]] + [models[0]["original_objectives"][1]])]
    np.savez_compressed(output_dir / (prefix + "_inputs.npz"), t=t, clean=clean, noisy=noisy,
                        common_centres=common)
    result = dict(run=prefix, system=name, metric=metric, noise_digest=digest(noisy),
                  frequencies=frequencies, base_plan=base_plan, reconstructed_plan=repr(plan),
                  provenance={str(p): source_hash(p) for p in paths},
                  max_baseline_wape_abs_error=max(reproduction),
                  reconstructed_equations=len(reproduction), candidate_windows=candidates, windows=windows,
                  common_centre_range=[float(t[common[0]]), float(t[common[-1]])],
                  models=models, evaluations=evaluations, baseline_dominators=baseline_dominators,
                  all_scale_dominators=all_scale_dominators)
    print(f"  Verified {len(reproduction)} equations; max WAPE delta {max(reproduction):.3g}; "
          f"widths {[p['ms'][0] for p in eligible]}; common centres {common.size}; "
          f"dominators {len(baseline_dominators)} -> {len(all_scale_dominators)} across all widths", flush=True)
    return result


def outputs(report, directory):
    (directory / "window_profile_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Fixed-coefficient discrepancy across weak-window widths", "",
             "One record, 5% noise, seed 0. No refitting, no search rerun; original dyad scores retained.",
             "Same admissible window widths for all equations in a run; common evaluation centres.",
             "Clean data and analytical coefficients are diagnostic controls, not ranking inputs.", ""]
    for case in report["cases"]:
        lines += [case["run"], f"  metric: {case['metric']}; base window: {case['base_plan']}",
                  f"  baseline dominators: {len(case['baseline_dominators'])}; "
                  f"dominate on EVERY width + dyad: {len(case['all_scale_dominators'])}",
                  "  width | reference D | analytical D (noisy) | analytical D (clean) | dominators"]
        for i, window in enumerate(case["windows"]):
            law = case["evaluations"]["reference"][i]
            analytic = case["evaluations"]["analytical"][i]
            lines.append(f"  {window['ms'][0]:5d} | {law['noisy']['discrepancy']:.8g} | "
                         f"{analytic['noisy']['discrepancy']:.8g} | "
                         f"{analytic['clean']['discrepancy']:.8g} | {len(window['dominators'])}")
        for identifier in case["all_scale_dominators"]:
            item = next(m for m in case["models"] if m["id"] == identifier)
            lines.append(f"  surviving {identifier}: {item['target']} = " + " + ".join(
                f"({c:.8g})*{term}" for c, term in zip(item["coefficients"], item["terms"]))
                + f" + ({item['intercept']:.8g}); I={item['original_objectives'][1]:.8g}")
        lines.append("")
    (directory / "window_profile_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (directory / "window_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "model", "role", "width", "fixed_instability", "noisy_D", "clean_D",
                         "noisy_residual_l1_mean", "noisy_target_l1_mean"])
        for case in report["cases"]:
            for item in case["models"]:
                for row in case["evaluations"][item["id"]]:
                    writer.writerow([case["run"], item["id"], item["role"], row["half_width_samples"],
                                     item["original_objectives"][1] if item["role"] != "analytical_control" else "",
                                     row["noisy"]["discrepancy"], row["clean"]["discrepancy"],
                                     row["noisy"]["residual_l1_mean"], row["noisy"]["target_l1_mean"]])


def plot(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ["Rational: plain target", "Kepler: inverse tokens", "Rational: keep, no trig (control)",
             "Euler–Cauchy: implicit form"]
    for ax, case, title in zip(axes.flat, report["cases"], names):
        widths = [w["ms"][0] for w in case["windows"]]
        for item in case["models"]:
            if item["role"] != "archive":
                continue
            values = [v["noisy"]["discrepancy"] for v in case["evaluations"][item["id"]]]
            baseline = item["id"] in case["baseline_dominators"]
            ax.plot(widths, values, color="#1976b9" if baseline else "#c9c9c9", alpha=0.8 if baseline else 0.45,
                    linewidth=1.4 if baseline else 0.8)
        for identifier, label, color, style in (("reference", "Reference: saved fitted coefficients", "#b12a31", "-"),
                                               ("analytical", "Reference: analytical coefficients", "#202020", "--")):
            ax.plot(widths, [v["noisy"]["discrepancy"] for v in case["evaluations"][identifier]],
                    style, color=color, linewidth=2, marker="o", markersize=3, label=label)
        ax.plot([], [], color="#1976b9", label="Original dominators (fixed coefficients)")
        ax.axvline(case["base_plan"]["ms"][0], color="#555555", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(widths, labels=[str(w) for w in widths])
        ax.set_xlabel("Weak-window half-width (samples)")
        ax.set_ylabel("Discrepancy on the noisy record")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Frozen equations across weak-window widths · same evaluation centres · 5% noise\n"
                 "Dotted line: original width. Grey: other archive members. Dyad unchanged.", fontsize=12)
    fig.savefig(directory / "window_profiles.png", dpi=180)
    fig.savefig(directory / "window_profiles.svg")
    plt.close(fig)


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    harness = args.input_dir / "run_auto.py"
    indices = seed_indices(harness)
    report = dict(
        protocol="Fixed archived coefficients and frequencies; vary only weak-window width at original smoothness. "
                 "Use original chooser feasibility and half-power gate, restrict all widths to common centres. "
                 "Compare per-scale discrepancy with original dyad, without refitting or averaging scales.",
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        source_hashes={str(p): source_hash(p) for p in [Path(__file__), Path(WF.__file__), harness,
                                                       REPO / "epde/preprocessing/deriv_calculators.py"]},
        limitations=["One noise seed and one record per system; archived candidates only.",
                     "Widths are correlated transformations of the same data, not independent validation sets.",
                     "No retuning of the coefficients; this measures transfer of the fitted equations.",
                     "Target-dependent normalization and the original instability are retained.",
                     "Keep/no-trig rational arm is an existing diagnostic control, not a proposed library restriction.",
                     "Analytical coefficients and clean data are used only for diagnostic controls."],
        cases=[])
    for prefix, law, metric in RUNS:
        report["cases"].append(audit_case(args.input_dir, args.output_dir, prefix, law, metric, indices))
    outputs(report, args.output_dir)
    plot(report, args.output_dir)
    print(f"Reports and plots saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
