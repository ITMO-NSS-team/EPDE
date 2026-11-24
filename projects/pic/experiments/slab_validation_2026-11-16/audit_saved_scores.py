"""Compare all saved slab-partition scores without rerunning the search.

Usage:
    python audit_saved_scores.py --input-dir PATH --output report.json

Inputs are the four November 14 archives from the Claude scratch harness.
Each saved k-fold score already pools EVERY held slab at that partition size.
This audit combines partition sizes, not individual slab models or raw errors.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


RUNS = (
    ("rational_trace_PW_e30_n5_weak_kf", "exact"),
    ("kepler_trace_PI_e30_n5_weak_kf", "exact_inv"),
    ("rational_trace_PK_e30_n5_weak_cap_kf_notrig", "exact"),
    ("cauchy_trace_FW_e30_n5_weak_proff_kf", "exact"),
)
COUNTS = ("2", "4", "8")


def dominates(a, b):
    """The original report's strict Pareto comparison, with shape validation."""
    if len(a) != len(b):
        raise ValueError("Different objective dimensions")
    if not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError("Non-finite objective")
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def objectives(record):
    original = record["obj"] if "obj" in record else record["values"]
    if len(original) != 2:
        raise ValueError("Expected discrepancy and instability for one equation")
    diag = record["diag"]
    folds = diag["slab_kfold"]
    out = [folds[k]["out"] for k in COUNTS]
    one = [folds[k]["one"] for k in COUNTS]
    # At k=2, fitting one slab and fitting all but the held slab are identical.
    if not math.isclose(out[0], one[0], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Two-slab protocols disagree")
    instability = original[1]
    result = {
        "original": original,
        "mean_out_2_4_8": [sum(out) / 3, instability],
        "max_out_2_4_8": [max(out), instability],
        "vector_out_2_4_8": out + [instability],
        "mean_one_2_4_8": [sum(one) / 3, instability],
        "vector_out_and_one_2_4_8": out + one + [instability],
        "vector_two_halves": [diag["holdout"][k] for k in ("ab", "ba")] + [instability],
    }
    for k, value in zip(COUNTS, out):
        result[f"out_{k}"] = [value, instability]
    return result


def describe(record):
    return {key: record[key] for key in ("target", "terms", "constant")}


def audit(input_dir):
    rows = []
    for prefix, law_name in RUNS:
        path = input_dir / (prefix + "_ref.json")
        content = path.read_bytes()
        source = json.loads(content)
        archive = source["archive"]
        member_scores = [objectives(member) for member in archive]
        laws = [version for candidate in source["candidates"]
                if candidate["name"] == law_name
                for version in candidate.get("versions", [])
                if version["constant"] == 0]
        if len(laws) != 1:
            raise ValueError(f"{prefix}: expected exactly one reference without an intercept")
        law = laws[0]
        metrics = {}
        for name, law_score in objectives(law).items():
            witnesses = [dict(archive_index=i, **describe(member), objectives=scores[name])
                         for i, (member, scores) in enumerate(zip(archive, member_scores))
                         if dominates(scores[name], law_score)]
            metrics[name] = {
                "reference_objectives": law_score,
                "dominator_count": len(witnesses),
                "dominators": witnesses,
            }
        rows.append({
            "run": prefix,
            "source_path": str(path.resolve()),
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "archive_size": len(archive),
            "reference": dict(name=law_name, **describe(law)),
            "metrics": metrics,
        })
    return {
        "scope": "Offline rescoring of four saved archives; no evolutionary search rerun.",
        "conditions": "One record, 5% noise, weak form, 30 epochs, seed 0.",
        "definitions": {
            "out": "Fit on k-1 slabs; predict each held slab; pool all residual magnitudes / sum(abs(target)).",
            "one": "Fit on each single slab; score its complement; pool residual magnitudes / ((k-1)*sum(abs(target))).",
            "mean_out_2_4_8": "Equal-weight average of the three already-pooled partition scores, with original instability.",
            "max_out_2_4_8": "Maximum of the three partition scores, with original instability.",
            "vector_out_2_4_8": "Three separate pooled partition scores and original instability (four objectives).",
            "vector_out_and_one_2_4_8": "Six pooled protocol/partition scores and original instability (seven objectives; k=2 duplicated).",
            "vector_two_halves": "Separate ab and ba validation errors and original instability.",
        },
        "limits": [
            "Saved files contain pooled errors, not individual 4-/8-slab errors or fitted slab coefficients.",
            "These are comparisons against existing archive candidates, not results of a search with new objectives.",
            "Reference-law structures were refitted to noisy data; reported scores do not fix analytical coefficients.",
            "The rational keep-arm archive excludes trig tokens; it is only a historical diagnostic control.",
            "The rational plain-target arm cannot generate the reference's target-related right-hand term.",
            "More objective dimensions can remove dominance without identifying the physical law.",
            "Comparisons retain each equation's original target normalization and instability.",
        ],
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ("original", "mean_out_2_4_8", "vector_out_2_4_8", "vector_out_and_one_2_4_8")
    print("Run | original | mean of 2/4/8 | three scores separate | six scores separate")
    for row in result["runs"]:
        counts = " | ".join(str(row["metrics"][key]["dominator_count"]) for key in columns)
        print(f"{row['run']} | {counts}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
