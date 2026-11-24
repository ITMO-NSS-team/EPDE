"""Replay each held-out 0.4 s forecast with the arena wall alone and with pair exclusion."""
import csv
import json

import numpy as np

from physics import CAP, DT, LAG, NAMES, OUT, design, recordings
from swarm_sim import project


def score(track):
    folder = OUT / track.tag
    training = json.loads((folder / "training.json").read_text())
    assessment = json.loads((folder / "assessment.json").read_text())
    method = assessment["selected_search"]
    model = assessment["models"][method]
    lookup = {tuple(map(int, k.split("/"))): np.array(v) for k, v in training["lookup"].items()}
    arenas = {int(k): np.array(v) for k, v in training["arenas"].items()}
    rows, features, base, velocity, _ = design(track, lookup, np.array(training["common"]), arenas, LAG)
    coef = np.array([model["coef"].get(name, 0.) for name in NAMES])
    forecast = base + features @ coef
    r2 = 1 - np.sum((forecast - velocity)**2) / np.sum((velocity - velocity.mean())**2)
    np.testing.assert_allclose(r2, model["heldout"]["r2"], atol=1e-7)

    xc, yc, radius = arenas[int(track.tag.rsplit("D", 1)[1])]
    container = dict(xc=xc, yc=yc, r_max=radius)
    proposal = track.pos[rows] + DT * forecast
    wall_only = np.stack([project(p, container, core=0.) for p in proposal])
    with_contact = np.stack([project(p, container, core=CAP) for p in proposal])
    observed = track.pos[rows + LAG]
    mm_per_px = 48.7 / CAP
    correction = np.linalg.norm(with_contact - wall_only, axis=-1)
    changed = correction > 1.  # one camera pixel, as in the overlap diagnostic
    error_before = mm_per_px * np.linalg.norm(wall_only - observed, axis=-1)
    error_after = mm_per_px * np.linalg.norm(with_contact - observed, axis=-1)
    return dict(recording=track.tag, model=method, frames=len(rows), robot_steps=changed.size,
                changed_over_one_pixel=int(changed.sum()), changed_fraction=float(changed.mean()),
                median_correction_mm_when_changed=float(np.median(correction[changed]) * mm_per_px)
                if changed.any() else None,
                closer_fraction_when_changed=float(np.mean(error_after[changed] < error_before[changed]))
                if changed.any() else None,
                position_rmse_wall_only_mm=float(np.sqrt(np.mean(error_before**2))),
                position_rmse_with_contact_mm=float(np.sqrt(np.mean(error_after**2))))


if __name__ == "__main__":
    results = [score(track) for track in recordings().values()]
    output = OUT / "contact_step.csv"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(output)
    print(output)
