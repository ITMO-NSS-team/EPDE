"""Paired weak training measurements on each trajectory, with shared redraws."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from holdouts import ROOT, E, TrainingMoments, np, records


@lru_cache(maxsize=2)
def geometry(system):
    problem = E.R.datasets.load(system)
    axes = E.R.axes_of(problem)
    cases, labels = E.R.panel(problem)
    reference = next(c for c in cases if c["label"] == "reference")
    columns = ([reference["yi"], *reference["xi"]] if system == "kdv" else list(range(len(labels))))
    all_terms, opaque = E.R.symbols(problem, labels)
    terms = [all_terms[i] for i in columns]
    baseline_path, base = E.F.baseline(system, 0)
    local_path = E.F.ROOT / "fit_results" / f"{system}_seed0.json"
    validation_path = E.M.ROOT / "results" / f"{system}_seed0.json"
    local, validation = [json.loads(p.read_text()) for p in (local_path, validation_path)]
    common = slice(*validation["segment"])
    tax = [axes[0][common], *axes[1:]]
    top = {k: a[common] for k, a in opaque.items()}
    folds = []
    for g, group in enumerate(base["groups"]):
        idx = np.asarray(group["indices"])
        seg, ax, op, plan = E.Q.mapped_plan(base, g, local["plans"][g]["plan"], axes, opaque)
        local_bank = E.Q.Bank(ax, terms, op, plan=plan)
        key = validation["folds"][g]["selection"]["q0"]["key"]
        tplan = validation["banks"][key]["plan"]
        w = np.pi / (tplan["ms"][0] * (tax[0][1] - tax[0][0]))
        time = E.Q.Bank(tax, terms, top, plan=tplan,
                        frequencies=[w * np.r_[0:5, -1:-5:-1], np.zeros(1)])
        if system == "kdv":
            xplan = plan
            modes = np.array([1., -1.])
        else:
            assert len(ax[1]) == 81 and plan["cut"][1] == 0
            xplan = dict(q=(plan["q"][0], 3), cut=plan["cut"], ms=(plan["ms"][0], 30),
                         ss=(plan["ss"][0], 10), shape=(plan["shape"][0], 3))
            modes = np.array([0., 1., -1.])
        w = np.pi / (xplan["ms"][1] * (ax[1][1] - ax[1][0]))
        spatial = E.Q.Bank(ax, terms, op, plan=xplan, frequencies=[np.zeros(1), w * modes])
        direction = None
        if system == "kdv_cossin":
            forcing, = [i for i, term in enumerate(labels) if "cos(t)sin(x)" in str(term) and len(term) == 1]
            # The measured forcing is deterministic, so use the zero field to get its weak moments.
            mat = spatial.matrix({"u": np.zeros((len(ax[0]), len(ax[1])))})
            force = mat[:, forcing].reshape(spatial.shape)
            direction = force / np.linalg.norm(force, axis=1, keepdims=True)
        folds.append((idx, seg, local_bank, time, spatial, direction))
    sets = [set(f[0]) for f in folds]
    assert len(set.union(*sets)) == sum(map(len, sets)) == len(axes[0])
    inputs = {p.name: records.sha(p) for p in (baseline_path, local_path)}
    inputs["validation"] = records.sha(validation_path)
    return problem, axes, columns, common, folds, inputs


def train(job, directory=ROOT):
    system, seed = job
    path = directory / "training" / f"{system}_{seed}.npz"
    if path.with_suffix(".json").exists():
        return path.stem
    problem, axes, columns, common, folds, inputs = geometry(system)
    raw = problem.noisy(5, seed)
    saved = {}
    for g, (idx, seg, local, time, spatial, direction) in enumerate(folds):
        obs = {k: a[idx] for k, a in raw.items()}
        noise = {k: E.R.WF.noise_model(a) for k, a in obs.items()}
        pilot, _ = E.R.reconstruct(obs, axes, idx)

        def blocks(field):
            part = {k: a[seg] for k, a in field.items()}
            x = spatial.matrix(part)
            out = dict(local=local.matrix(part), time=time.matrix({k: a[common] for k, a in field.items()}),
                       spatial=x)
            if direction is not None:
                out["forcing"] = np.einsum("ij,ijk->ik", direction.conj(), x.reshape(*spatial.shape, -1))
            return out

        stats = {k: TrainingMoments(point) for k, point in blocks(pilot).items()}
        rng = np.random.default_rng(np.random.SeedSequence([1171, seed, g]))
        for _ in range(128):
            draw = {k: pilot[k][idx] + E.R.WF.ar_noise(rng, v.shape, *noise[k][:2]) for k, v in obs.items()}
            reconstructed, _ = E.R.reconstruct(draw, axes, idx)
            for tag, m in blocks(reconstructed).items():
                stats[tag].update(m)
        for tag, st in stats.items():
            saved[f"{g}_{tag}_point"] = st.point
            saved[f"{g}_{tag}_corrected"] = st.corrected() / len(st.point)
    np.savez_compressed(path, **saved)
    meta = dict(columns=columns, raw={k: E.R.digest(a) for k, a in raw.items()}, archive=records.sha(path),
                inputs=inputs, protocol=records.sha(directory / "protocol.txt"))
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path.stem


if __name__ == "__main__":
    (ROOT / "training").mkdir(exist_ok=True)
    jobs = [(system, start + i) for i in range(24)
            for system, start in (("kdv_cossin", 788001), ("kdv", 789001))]
    jobs[:0] = [("kdv_cossin", 786001), ("kdv", 787001)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(train, job) for job in jobs]):
            print(f.result(), flush=True)
