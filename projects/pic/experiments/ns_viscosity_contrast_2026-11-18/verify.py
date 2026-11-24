"""Independently recompute every reported held-out contrast from stored fits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root/"summary.json").read_text())
for name,key in (("paired.csv","paired_sha256"),("fits.json","fits_sha256")):
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==summary[key]
rows = json.loads((root/"fits.json").read_text())
with (root/"paired.csv").open() as f:
    paired = list(csv.DictReader(f))
scores = {}
max_error = 0.
for seed in sorted({r["seed"] for r in rows}):
    with np.load(root/"holdouts"/f"{seed}.npz") as data:
        ids = data["ids"].tolist()
        for r in (r for r in rows if r["seed"]==seed):
            g = r["fold"]
            v = np.zeros(len(ids))
            v[[ids.index(k) for k in r["ids"]]] = r["coefficients"]
            cross = data[f"left{g}"].T @ data[f"right{g}"]
            d = float((v @ cross @ v)/np.dot(v*v,data[f"scale{g}"]**2))
            max_error = max(abs(d-r["D"]),max_error)
            np.testing.assert_allclose(d,r["D"],rtol=2e-9,atol=5e-14)
            scores[(seed,r["bank"],r["fit"],r["candidate"],g)] = d
for row in paired:
    args = (int(row["seed"]),row["bank"],row["fit"])
    full,drop = [np.mean([scores[(*args,name,g)] for g in range(3)])
                 for name in ("e0:reference","e0:drop_3")]
    np.testing.assert_allclose([full,drop,drop-full],
        [float(row[k]) for k in ("full","drop","drop_minus_full")],atol=5e-14,rtol=2e-9)
for row in summary["table"]:
    cases = [r for r in paired if (r["cohort"],r["bank"],r["fit"])==
             (row["cohort"],row["bank"],row["fit"])]
    vals = [float(r["drop_minus_full"]) for r in cases]
    assert len(vals)==row["total"] and sum(d>0 for d in vals)==row["wins"]
    np.testing.assert_allclose(np.median(vals),row["median_delta"])
    np.testing.assert_allclose(np.median([float(r["median_viscosity"]) for r in cases]),row["median_viscosity"])
assert (len(rows),len(scores),len(paired),len(summary["table"]))==(1170,1170,195,26)
print(f"Verified {len(rows)} fold scores, {len(paired)} paired contrasts and 26 summary rows. Maximum score error {max_error:.3g}.")
