"""Truth-only localization of weak-column errors; never a selection criterion.

Evaluate analytical reference coefficients to distinguish column error from
coefficient-estimation error. Also show frozen observed coefficients on clean
columns and the common-noise change in their residuals.
"""

import argparse
import json
from pathlib import Path

import signal_refits as S
import window_profiles as W
import numpy as np


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def run(case, trial, inputs):
    labels = sorted({v for item in case["models"] for v in [item["target"], *item["terms"]]})
    terms = [W.parse_term(label) for label in labels]
    index = {label: i for i, label in enumerate(labels)}
    with np.load(inputs / f"{case['run']}_seed{trial['seed']}_signal.npz") as data:
        t = data["t"]
        mats = {stage: W.matrix(terms, W.fields(t, data[key], trial["frequencies"], terms), case["base_plan"])[0]
                for stage, key in (("clean", "clean"), ("raw", "noisy"), ("smoothed", "smoothed"))}
    reference = case["models"][0]
    exact = dict(reference, coefficients=[W.ANALYTICAL[case["system"]][label] for label in reference["terms"]],
                 intercept=0)
    out = dict(seed=trial["seed"], reference_target=reference["target"], analytical_coefficients=exact["coefficients"],
               analytical_clean=W.score(exact, mats["clean"], index, case["metric"]), stages={})
    for stage in ("raw", "smoothed"):
        clean_y = mats["clean"][:, index[reference["target"]]]
        target_error = mats[stage][:, index[reference["target"]]] - clean_y
        parts = []
        rhs_error = np.zeros_like(target_error)
        for term, coefficient in zip(reference["terms"], exact["coefficients"]):
            error = coefficient * (mats[stage][:, index[term]] - mats["clean"][:, index[term]])
            rhs_error += error
            parts.append(dict(term=term, coefficient=coefficient, contribution_error_rms=rms(error),
                              error_over_clean_target_rms=rms(error) / rms(clean_y)))
        models = []
        for item, fitted in zip(case["models"], trial[stage]["models"]):
            frozen = dict(item, coefficients=fitted["coefficients"], intercept=fitted["intercept"])
            clean_score = W.score(frozen, mats["clean"], index, case["metric"])
            models.append(dict(id=item["id"], observed_D=fitted["objectives"][0],
                               clean_D_with_observed_coefficients=clean_score["discrepancy"],
                               fitted_coefficients=fitted["coefficients"]))
        out["stages"][stage] = dict(analytical_reference=W.score(exact, mats[stage], index, case["metric"]),
                                     target_error_over_clean_target_rms=rms(target_error) / rms(clean_y),
                                     rhs_error_over_clean_target_rms=rms(rhs_error) / rms(clean_y),
                                     residual_error_over_clean_target_rms=rms(target_error - rhs_error) / rms(clean_y),
                                     parts=parts, models=models)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_dir / "signal_refits.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    report = dict(protocol=__doc__, sources={str(p): W.source_hash(p) for p in [source, Path(__file__), Path(W.__file__)]},
                  cases=[dict(run=c["run"], system=c["system"], trials=[run(c, t, args.input_dir) for t in c["trials"]])
                         for c in data["cases"]])
    (args.output_dir / "oracle_column_errors.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["Oracle error localization (truth supplied; not a data-only scoring method)", ""]
    for case in report["cases"]:
        lines.append(case["run"])
        for trial in case["trials"]:
            for stage, item in trial["stages"].items():
                lines.append(f"  seed {trial['seed']} {stage}: analytical law D={item['analytical_reference']['discrepancy']:.7g}; "
                             f"target error={item['target_error_over_clean_target_rms']:.7g}; "
                             f"combined RHS error={item['rhs_error_over_clean_target_rms']:.7g}; "
                             f"residual error={item['residual_error_over_clean_target_rms']:.7g} "
                             "(RMS errors relative to clean target)")
                for m in item["models"][:2]:
                    lines.append(f"    {m['id']}: observed D={m['observed_D']:.7g}; "
                                 f"clean D with same coefficients={m['clean_D_with_observed_coefficients']:.7g}; "
                                 f"coefficients={m['fitted_coefficients']}")
    (args.output_dir / "oracle_column_errors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
