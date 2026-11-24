"""Fit the preregistered joint and staged arms and score common forced holdouts."""
import csv
import json
from collections import defaultdict
from holdouts import ROOT, PREVIOUS, E, corrected_fit, np, records, score
from evaluate import matrices
from scipy.linalg import eigh


ARMS = {
    "baseline": ("original", False, False), "space_only": ("original", True, False),
    "forcing_only": ("projected", False, False), "both": ("projected", True, False),
    "staged_baseline": ("original", False, True), "staged_both": ("projected", True, True),
}


def fitted(fgram, fmass, ugram, umass, positions, staged):
    columns = len(fmass)
    ids = np.ix_(positions, positions)
    basis = np.eye(columns)
    gram = fgram.copy()
    mass = fmass.copy()
    if staged is True:
        intrinsic, _, status = corrected_fit(ugram, np.sqrt(umass))
        assert status == "ok"
        shared = np.zeros(columns)
        shared[positions] = intrinsic
        free = np.delete(np.eye(columns), positions, axis=1)
        basis = np.column_stack((shared, free))
    elif staged is False:
        gram[ids] += ugram
        mass[positions] += umass
    elif staged != "forced_only":
        raise ValueError(f"Unknown fit mode {staged}")
    reduced = basis.T @ gram @ basis
    weight = (basis.T * mass) @ basis
    np.testing.assert_allclose(weight, np.diag(np.diag(weight)), atol=1e-13)
    v, _, status = corrected_fit(reduced, np.sqrt(np.diag(weight)))
    assert status == "ok"
    alternative = eigh((reduced + reduced.T) / 2, weight, driver="gvx", subset_by_index=(0, 0))[1][:, 0]
    return basis @ v, basis @ alternative


def blocks(tr, fold, tags):
    array, gram = matrices(tr, fold, tags)
    return gram, np.sum(abs(array) ** 2, axis=0)


def source(directory, stem):
    path = directory / (stem + ".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert records.sha(path) == meta["archive"]
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, meta


def pair(fseed, useed, replay=False, directory=ROOT, arms=ARMS):
    fstem, ustem = f"kdv_cossin_{fseed}", f"kdv_{useed}"
    old_root = PREVIOUS / "kdv_joint"
    obs, meta = source((old_root if replay else directory) / "records", fstem)
    ft, fm = source(directory / "training", fstem)
    ut, um = source(directory / "training", ustem)
    assert fm["raw"] == meta["raw_hash"] and fm["protocol"] == um["protocol"] == records.sha(directory / "protocol.txt")
    cases = [c for c in meta["candidates"] if c["label"] == "reference" or c["label"].startswith(("add_", "drop_"))]
    ref = cases[0]
    assert ref["label"] == "reference" and len(cases) == 12
    _, ulabels = E.R.panel(E.R.datasets.load("kdv"))
    forced_labels = [json.dumps(label, sort_keys=True) for label in meta["labels"]]
    shared = {forced_labels.index(json.dumps(ulabels[k], sort_keys=True)): j for j, k in enumerate(um["columns"])}
    assert len(shared) == 3 and fm["columns"] == list(range(len(meta["labels"])))
    checks = defaultdict(float)
    rows = []
    if replay:
        historical, = [r for r in json.loads((old_root / "folds.json").read_text()) if r["forced_seed"] == fseed]
        for stem, new, cols in ((fstem, ft, fm["columns"]), (ustem, ut, um["columns"])):
            with np.load(old_root / "training" / (stem + ".npz")) as old:
                for k, m in new.items():
                    if not any(t in k for t in ("_local_", "_time_")):
                        continue
                    expected = old[k][:, cols] if k.endswith("_point") else old[k][np.ix_(cols, cols)]
                    np.testing.assert_allclose(m, expected, rtol=2e-10, atol=3e-12)
    for fold in range(3):
        forced = {name: blocks(ft, fold, tags) for name, tags in
                  (("original", ("local", "time")), ("projected", ("spatial", "forcing")))}
        if any(design == "wide" for design, _, _ in arms.values()):
            forced["wide"] = tuple(2 * a for a in blocks(ft, fold, ("spatial",)))
        unforced = [blocks(ut, fold, tags) for tags in (("local", "time"), ("local", "spatial"))]
        for c in cases:
            ids = sorted([c["yi"], *c["xi"]])
            pos = [j for j, k in enumerate(ids) if k in shared]
            cols = [shared[ids[j]] for j in pos]
            l, r, sc = [obs[f"{name}{fold}"][..., ids] for name in ("left", "right", "scale")]
            for arm, (forced_design, use_space, staged) in arms.items():
                fg, fn = forced[forced_design]
                ug, un = unforced[use_space]
                f, m = fg[np.ix_(ids, ids)], fn[ids]
                u, n = ug[np.ix_(cols, cols)], un[cols]
                v, alternative = fitted(f, m, u, n, pos, staged)
                d = score(l, r, sc, v)
                v2d = alternative @ (l.T @ r) @ alternative / np.sum((sc * alternative) ** 2)
                checks["generalized_eigen_score_error"] = max(checks["generalized_eigen_score_error"], abs(d-v2d))
                np.testing.assert_allclose(d, v2d, rtol=1e-7, atol=3e-10)
                # Simultaneously change physical units and reverse the candidate ordering.
                units = (-1.) ** np.arange(len(ids)) * np.geomspace(.12, 6., len(ids))
                perm = np.arange(len(ids))[::-1]
                ui = units[pos]
                p2 = [list(perm).index(i) for i in pos]
                mapped, _ = fitted((f * np.outer(units, units))[np.ix_(perm, perm)], (m * units**2)[perm],
                                   u * np.outer(ui, ui), n * ui**2, p2, staged)
                d2 = score(l, r, sc, mapped[np.argsort(perm)] * units)
                checks["column_change_score_error"] = max(checks["column_change_score_error"], abs(d-d2))
                np.testing.assert_allclose(d, d2, rtol=1e-7, atol=3e-10)
                row = dict(seed=fseed, fold=fold, arm=arm, case=c["label"], score=d)
                if c is ref:
                    co = [-v[ids.index(k)] / v[ids.index(ref["yi"])] for k in ref["xi"]]
                    exact = np.zeros(len(ids))
                    exact[ids.index(ref["yi"])] = 1.
                    for k, b in zip(ref["xi"], ref["exact"]):
                        exact[ids.index(k)] = -b
                    row.update(zip(("nonlinearity", "dispersion", "forcing"), co))
                    row["fixed"] = score(l, r, sc, exact)
                if replay and arm == "baseline" and c["id"] in historical["scores"]["joint"]:
                    delta = abs(d-historical["scores"]["joint"][c["id"]][fold])
                    assert delta < 2e-11
                    checks["archived_score_error"] = max(checks["archived_score_error"], delta)
                rows.append(row)
    return rows, dict(checks), fm["raw"]["u"], um["raw"]["u"]


def save_csv(name, rows, directory=ROOT):
    with (directory / name).open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, fields, directory=ROOT, arms=ARMS):
    keys = ("nonlinearity", "dispersion", "forcing")
    truth = np.array([-6., -1., 1.])
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["seed"], r["arm"], r["case"]].append(r)
    field_rows = []
    for seed in fields:
        for arm in arms:
            ref = grouped[seed, arm, "reference"]
            base = np.mean([r["score"] for r in ref])
            fixed = np.mean([r["fixed"] for r in ref])
            row = dict(seed=seed, arm=arm, score=base, fixed=fixed)
            row.update({k: np.median([r[k] for r in ref]) for k in keys})
            for suffix, count in (("add", 8), ("drop", 3)):
                for j in range(count):
                    case = f"{suffix}_{j}"
                    row[case] = np.mean([r["score"] for r in grouped[seed, arm, case]]) - base
            field_rows.append(row)
    save_csv("fields.csv", field_rows, directory)
    result = {}
    for arm in arms:
        rr = [r for r in field_rows if r["arm"] == arm]
        control = [r for r in field_rows if r["arm"] == "baseline"]
        assert [r["seed"] for r in rr] == [r["seed"] for r in control]
        err = abs(np.array([[r[k] for k in keys] for r in rr]) - truth)
        cerr = abs(np.array([[r[k] for k in keys] for r in control]) - truth)
        wins = np.array([[r[f"add_{j}"] < 0 for j in range(8)] for r in rr])
        base_wins = np.array([[r[f"add_{j}"] < 0 for j in range(8)] for r in control])
        deletions = np.array([[r[f"drop_{j}"] > 0 for j in range(3)] for r in rr])
        fixed_wins = sum(r[f"add_{j}"] + r["score"] < r["fixed"] for r in rr for j in range(8))
        field_change = wins.sum(axis=1) - base_wins.sum(axis=1)
        result[arm] = dict(
            coefficients=np.median([[r[k] for k in keys] for r in rr], axis=0).tolist(),
            median_absolute_errors=np.median(err, axis=0).tolist(),
            fields_closer_than_baseline=(err < cerr).sum(axis=0).tolist(),
            extra_wins=int(wins.sum()), by_addition=wins.sum(axis=0).tolist(),
            fields_beating_all_eleven=int(np.sum((~wins.any(axis=1)) & deletions.all(axis=1))),
            fewer_equal_more_extra_wins=[int(np.sum(sign * field_change > 0)) if sign else int(np.sum(field_change == 0))
                                        for sign in (-1, 0, 1)],
            beats_fixed=int(fixed_wins),
            genuine_retained=deletions.sum(axis=0).tolist(),
            median_score=float(np.median([r["score"] for r in rr])),
            improved_reference=sum(int(a["score"] < b["score"]) for a, b in zip(rr, control)))
    return result


def run(directory, arms, forced_start, unforced_start, replay=None):
    rows, checks, hashes = [], [], []
    for i in range(24):
        fold_rows, errs, fhash, uhash = pair(forced_start+i, unforced_start+i, directory=directory, arms=arms)
        rows.extend(fold_rows)
        checks.append(errs)
        hashes.append((fhash, uhash))
    assert len(set(v for pair_hash in hashes for v in pair_hash)) == 48
    save_csv("folds.csv", rows, directory)
    results = summarize(rows, list(range(forced_start, forced_start+24)), directory, arms)
    output = dict(arms=results, folds_checked=len(rows), baseline_replay=replay,
                  max_errors={k: max(c[k] for c in checks) for k in checks[0]}, hashes=hashes,
                  protocol=records.sha(directory / "protocol.txt"))
    (directory / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    for arm, result in results.items():
        print(arm, result["coefficients"], result["extra_wins"], result["genuine_retained"],
              result["fields_beating_all_eleven"])


if __name__ == "__main__":
    _, replay, _, _ = pair(786001, 787001, replay=True)
    run(ROOT, ARMS, 788001, 789001, replay)
