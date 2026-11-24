"""Separate wide weak windows from forcing-matched projection on fresh fields."""
import csv
import json
from compare import run
from holdouts import np
from prepare_check import DEST

ARMS = {
    "baseline": ("original", False, False), "wide": ("wide", False, False),
    "projected": ("projected", False, False), "staged_baseline": ("original", False, True),
    "staged_wide": ("wide", False, True), "staged_projected": ("projected", False, True),
    "staged_both": ("projected", True, True),
}

if __name__ == "__main__":
    run(DEST, ARMS, 790001, 791001)
    with (DEST / "fields.csv").open(newline="") as f:
        table = {(int(r["seed"]), r["arm"]): r for r in csv.DictReader(f)}
    out = {}
    for first, second in (("wide", "projected"), ("staged_wide", "staged_projected")):
        score_change, count_change = [], []
        for seed in range(790001, 790025):
            a, b = table[seed, first], table[seed, second]
            score_change.append(float(b["score"]) - float(a["score"]))
            count_change.append(sum(int(float(b[f"add_{i}"]) < 0) - int(float(a[f"add_{i}"]) < 0) for i in range(8)))
        delta = np.array(count_change)
        out[second] = dict(comparator=first, fewer_same_more=[int(sum(delta < 0)), int(sum(delta == 0)), int(sum(delta > 0))],
                           median_score_change=float(np.median(score_change)),
                           n_lower_score=int(sum(d < 0 for d in score_change)))
    (DEST / "projection_contrasts.json").write_text(json.dumps(out, indent=2))
    print(out)
