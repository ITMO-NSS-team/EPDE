"""Freeze the common held-out measurements before comparing training methods."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "implicit_objective_2026-11-18"
sys.path.insert(0, str(ROOT.parent / "uncertainty_refit_2026-11-16"))
import experiment as E
import numpy as np

REPLAY = (6111, 6112, 6113)
FRESH = tuple(range(72011, 72023))
NAMES = ("e0:reference", "e0:drop_3")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holdouts(seed):
    dest = ROOT / "holdouts" / f"{seed}.npz"
    if seed in REPLAY:
        path = PRIOR / "records" / f"ns_{seed}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        assert digest(path) == meta["archive"]
        candidates = meta["candidates"]
        pos = [next(i for i,c in enumerate(candidates) if c["id"] == s) for s in NAMES]
        ids = sorted([candidates[pos[0]]["yi"], *candidates[pos[0]]["xi"]])
        with np.load(path) as z:
            out = {f"{key}{g}":z[f"{key}{g}"][...,ids] for g in range(3)
                   for key in ("left","right","scale")}
            out["anchors"] = z["old_coefficients"][:,ids,:][:,:,pos]
        source = meta["archive"]
    else:
        exp = E.Experiment("ns",seed)
        pos = [next(i for i,c in enumerate(exp.cs) if c["id"] == s) for s in NAMES]
        ids = sorted([exp.cs[pos[0]]["yi"], *exp.cs[pos[0]]["xi"]])
        assert all(exp.choices[g][i]=="broad0_clsm" for i in pos for g in range(3))
        matrices = []
        for idx, state in zip(exp.indices,exp.states):
            obs = {v:y[idx] for v,y in exp.raw.items()}
            mapped = E.O.map_values(state["map"], obs)
            matrices.append({key:exp.basis[key].convert(E.O.matrix(bank,mapped,state["cov"]))
                             for key,bank in exp.vbanks.items()})
        out = {"anchors":exp.anchor_vectors[:,ids,:][:,:,pos]}
        for g, (idx,state) in enumerate(zip(exp.indices,exp.states)):
            obs = {v:y[idx] for v,y in exp.raw.items()}
            pilot,_ = E.R.reconstruct(obs,exp.axes,idx)
            key = exp.keys[g]
            vp = exp.vbanks[key].matrix({v:y[exp.vseg] for v,y in pilot.items()})
            scale = np.linalg.norm(state["white"] @ exp.basis[key].convert(vp),axis=0)
            np.testing.assert_allclose(scale[exp.cs[pos[0]]["yi"]]**2,state["scale"][pos[0]])
            out[f"scale{g}"] = scale[ids]
            a,b = [h for h in range(3) if h!=g]
            for side,h in (("left",a),("right",b)):
                out[f"{side}{g}"] = (state["white"] @ matrices[h][key])[:,ids]
        source = {v:E.R.digest(y) for v,y in exp.raw.items()}
    out["ids"] = ids
    np.savez_compressed(dest,**out)
    dest.with_suffix(".json").write_text(json.dumps(dict(source=source,sha256=digest(dest))))
    return seed


if __name__ == "__main__":
    (ROOT/"holdouts").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed in pool.map(holdouts, (*REPLAY,*FRESH)):
            print(seed,flush=True)
