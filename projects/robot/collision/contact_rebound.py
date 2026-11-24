"""Test excess pair separation at fixed times after an inward cap crossing.

The baseline is the frozen drift forecast with whole-swarm disk and wall projection.
The response is relative displacement along the entry normal. A positive multiplier
adds incoming closing speed times elapsed time to this scalar forecast.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from impacts import episodes, pairs, slope
from physics import (CAP, CUTOFF, FPS, LAG, NAMES, OUT, input_at, motion_inputs,
                     partition, recordings, robot_parameters)
from swarm_sim import project


WIDTHS = (3, 5)
HORIZONS = (5, 10)
EARLY_HORIZONS = (1, 2, 3)
ENTRY_OFFSETS_PX = (-2, 0, 2)
CLEARANCES = {"neighborhood": CUTOFF, "physical_clearance": 1.03 * CAP}
MM_PER_PX = 48.7 / CAP
EVENT = np.dtype([("frame", int), ("i", int), ("j", int), ("release", int),
                  ("closing", float), ("strict", bool)])


def entries(track, arena, *, offset=0, geometry=None):
    """Vary the entry threshold alone. All horizons use the original ten-frame follow-up."""
    left, right, distances, neighbors, wall = pairs(track, arena) if geometry is None else geometry
    tables = {width: [] for width in WIDTHS}
    for k, (i, j) in enumerate(zip(left, right)):
        distance = distances[:, k]
        frame, release = episodes(distance, CAP + offset, 1.03 * CAP)
        available = (frame >= LAG) & (frame + max(HORIZONS) < len(distance)) & (distance[frame] > 0)
        frame, release = frame[available], release[available]
        if not len(frame):
            continue
        relative = track.pos[:, j] - track.pos[:, i]
        normal = relative[frame] / distance[frame, None]
        for width in WIDTHS:
            history = frame[:, None] - np.arange(width + 1)
            closing = -np.sum(slope(relative, frame - width, width) * normal, axis=1)
            keep = (closing > 0) & ~(wall[history, i] | wall[history, j]).any(axis=1)
            isolated = {}
            for label, limit in CLEARANCES.items():
                counts = neighbors[limit]
                partner = (distance[history] < limit).astype(int)
                isolated[label] = (np.maximum(counts[history, i], counts[history, j])
                                   <= partner).all(axis=1)
            keep &= isolated["physical_clearance"]
            if keep.any():
                batch = np.empty(keep.sum(), dtype=EVENT)
                batch["frame"], batch["release"] = frame[keep], release[keep]
                batch["i"], batch["j"] = i, j
                batch["closing"], batch["strict"] = closing[keep], isolated["neighborhood"][keep]
                tables[width].append(batch)
    return {w: np.concatenate(rows) if rows else np.empty(0, dtype=EVENT)
            for w, rows in tables.items()}


@dataclass
class Outcomes:
    incoming_travel: np.ndarray  # mm, closing speed times horizon
    residual: np.ndarray         # mm, observed minus baseline along entry normal
    released: np.ndarray         # reached the release threshold at least once by horizon
    outward_mm: np.ndarray | None = None  # observed relative displacement along entry normal


def forecasts(track, frames, params, common, arena, coef, horizons=HORIZONS):
    container = dict(xc=arena[0], yc=arena[1], r_max=arena[2])
    positions = {h: np.empty((len(frames), len(params), 2)) for h in horizons}
    for start in range(0, len(frames), 48):
        chunk = slice(start, start + 48)
        state = input_at(track, frames[chunk])
        features, base, _ = motion_inputs(*state, params, common, arena)
        velocity = base + features @ coef
        for h in horizons:
            endpoints = state[0] + velocity * (h / FPS)
            positions[h][chunk] = [project(p, container, core=CAP) for p in endpoints]
    return positions


def measure(track, tables, frames, predicted):
    result = {}
    for width, events in tables.items():
        frame, i, j = events["frame"], events["i"], events["j"]
        initial = track.pos[frame, j] - track.pos[frame, i]
        normal = initial / np.linalg.norm(initial, axis=1, keepdims=True)
        index = np.searchsorted(frames, frame)
        for h, projected in predicted.items():
            observed = track.pos[frame + h, j] - track.pos[frame + h, i]
            baseline = projected[index, j] - projected[index, i]
            residual = np.sum((observed - baseline) * normal, axis=1) * MM_PER_PX
            outward = np.sum((observed - initial) * normal, axis=1) * MM_PER_PX
            travel = events["closing"] * (h / FPS) * MM_PER_PX
            released = events["release"] <= frame + h
            for label in CLEARANCES:
                use = events["strict"] if label == "neighborhood" else slice(None)
                result[label, width, h] = Outcomes(travel[use], residual[use], released[use], outward[use])
    return result


def threshold_outcomes(track, lookup, common, arenas, coef, offsets, horizons):
    arena = np.array(arenas[track.tag.rsplit("D", 1)[1]])
    geometry = pairs(track, arena)
    tables = {k: entries(track, arena, offset=k, geometry=geometry) for k in offsets}
    frames = np.unique(np.concatenate([t["frame"] for widths in tables.values() for t in widths.values()]))
    predicted = forecasts(track, frames, robot_parameters(track, lookup), common, arena, coef, horizons)
    return {k: measure(track, widths, frames, predicted) for k, widths in tables.items()}


def outcomes(track, lookup, common, arenas, coef):
    return threshold_outcomes(track, lookup, common, arenas, coef, (0,), HORIZONS)[0]


def fit(blocks):
    """One weight per represented training recording for each competing correction."""
    xy = np.mean([np.mean(b.incoming_travel * b.residual) for b in blocks])
    xx = np.mean([np.mean(b.incoming_travel ** 2) for b in blocks])
    multiplier = float(np.clip(xy / xx, 0, 1))
    offset = max(0., float(np.mean([b.residual.mean() for b in blocks])))
    return multiplier, offset


def rmse(errors):
    return float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None


def early_scores(train, valid, test):
    """Compare a train-calibrated bias against the same model with an incoming-speed term."""
    # Each represented training recording has unit weight, including in the centering.
    bias = float(np.mean([b.residual.mean() for b in train])) if train else 0.
    mean_travel = float(np.mean([b.incoming_travel.mean() for b in train])) if train else 0.
    multiplier = 0.
    if train:
        variance = np.mean([np.mean((b.incoming_travel - mean_travel)**2) for b in train])
        if variance > 0:
            covariance = np.mean([np.mean((b.incoming_travel - mean_travel) * (b.residual - bias))
                                  for b in train])
            multiplier = float(np.clip(covariance / variance, 0., 1.))

    def improvement(b):
        error = b.residual - bias
        fitted = error - multiplier * (b.incoming_travel - mean_travel)
        return np.mean(error**2 - fitted**2)

    selected = multiplier if valid and np.mean([improvement(b) for b in valid]) > 0 else 0.
    error = test.residual - bias
    centered = test.incoming_travel - mean_travel
    return dict(outward_events=int(np.sum(test.outward_mm > 0)),
                outward_over_one_px=int(np.sum(test.outward_mm > MM_PER_PX)),
                mean_outward_displacement_mm=float(test.outward_mm.mean()) if len(error) else None,
                calibrated_offset_mm=bias, training_mean_incoming_travel_mm=mean_travel,
                offset_speed_multiplier=multiplier, selected_offset_speed_multiplier=selected,
                calibrated_offset_rmse_mm=rmse(error), offset_speed_rmse_mm=rmse(error - multiplier * centered),
                selected_offset_speed_rmse_mm=rmse(error - selected * centered))


def assess(track_tag, key, train, valid, test, *, early=False):
    label, width, h = key
    train = [items[key] for items in train if len(items[key].residual)]
    valid = [items[key] for items in valid if len(items[key].residual)]
    multiplier, offset = fit(train) if train else (None, None)
    selected = 0.
    if valid and multiplier is not None:
        deltas = [np.mean((b.residual - multiplier * b.incoming_travel) ** 2 - b.residual ** 2)
                  for b in valid]
        if np.mean(deltas) < 0:
            selected = multiplier
    error = test.residual
    fitted_error = error - multiplier * test.incoming_travel if multiplier is not None else None
    selected_error = error - selected * test.incoming_travel
    row = dict(recording=track_tag, isolation=label, third_clearance_px=CLEARANCES[label],
               pre_frames=width, horizon_s=h / FPS, train_recordings=len(train),
               train_events=sum(len(b.residual) for b in train), valid_recordings=len(valid),
               valid_events=sum(len(b.residual) for b in valid), test_events=len(error),
               released_by_horizon=int(test.released.sum()),
               train_multiplier=multiplier, selected_multiplier=selected, train_offset_mm=offset,
               baseline_rmse_mm=rmse(error), fitted_rmse_mm=rmse(fitted_error) if train else None,
               selected_rmse_mm=rmse(selected_error), offset_rmse_mm=rmse(error - offset) if train else None)
    for name, mask in (("released", test.released), ("unreleased", ~test.released)):
        row[name + "_baseline_rmse_mm"] = rmse(error[mask])
        row[name + "_selected_rmse_mm"] = rmse(selected_error[mask])
    if early:
        row.update(early_scores(train, valid, test))
    return row


def score_rebound(*, early=False):
    tracks = recordings()
    offsets = ENTRY_OFFSETS_PX if early else (0,)
    horizons = EARLY_HORIZONS if early else HORIZONS
    rows = []
    for tag in tracks:
        training = json.loads((OUT / tag / "training.json").read_text())
        assessment = json.loads((OUT / tag / "assessment.json").read_text())
        assert assessment["held"] == training["held"] == tag
        assert set(training["train"]) == set(tracks) - {tag}
        model = assessment["models"][assessment["selected_search"]]
        coef = np.array([model["coef"].get(n, 0.) for n in NAMES])
        lookup = {tuple(map(int, k.split("/"))): np.array(v) for k, v in training["lookup"].items()}
        params = lookup, np.array(training["common"]), training["arenas"], coef, offsets, horizons
        train, valid, test = partition(tracks, tag)
        training_data = [threshold_outcomes(t, *params) for t in train]
        validation_data = [threshold_outcomes(t, *params) for t in valid]
        test_data = threshold_outcomes(test, *params)
        for offset, tables in test_data.items():
            train_blocks = [t[offset] for t in training_data]
            valid_blocks = [v[offset] for v in validation_data]
            for key, table in tables.items():
                row = assess(tag, key, train_blocks, valid_blocks, table, early=early)
                if early:
                    row["entry_offset_px"] = offset
                rows.append(row)
        print("Scored", tag, flush=True)
    path = OUT / ("early_contact.csv" if early else "contact_rebound.csv")
    with NamedTemporaryFile("w", dir=OUT, newline="", encoding="utf-8", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    Path(f.name).replace(path)
    print(path)


if __name__ == "__main__":
    from run import check_inputs
    check_inputs()
    score_rebound()
