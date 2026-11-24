"""Retrospective single-trajectory ablation on both completed noise cohorts."""
import json
from compare import pair, summarize, save_csv
from holdouts import ROOT

ARMS = {k: (design, False, "forced_only") for k, design in
        (("baseline", "original"), ("wide", "wide"), ("projected", "projected"))}


if __name__ == "__main__":
    out = {}
    for source, start in ((ROOT, 788001), (ROOT / "projection_check", 790001)):
        dest = source / "forced_only"
        dest.mkdir(exist_ok=True)
        rows = []
        for i in range(24):
            folds, _, _, _ = pair(start+i, start+1000+i, directory=source, arms=ARMS)
            rows.extend(folds)
        save_csv("folds.csv", rows, dest)
        by = summarize(rows, range(start, start+24), dest, ARMS)
        out[start] = by
        print(start, [(k, v["extra_wins"], v["genuine_retained"], v["coefficients"]) for k, v in by.items()])
    (ROOT / "forced_only.json").write_text(json.dumps(out, indent=2, allow_nan=False))
