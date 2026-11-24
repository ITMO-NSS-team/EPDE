"""Fit the two paired arms per bank and evaluate common held-out residuals."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from panel import ROOT, PRIOR, REPLAY, FRESH, SYSTEMS, records, record_path
sys.path.insert(0,str(records.LEGACY))
import experiment as E
import numpy as np
from implicit import fit, projective_error, score
from benchmark import stability

ARMS = ("local_time_off","local_time_on","broad_off","broad_on","legacy")


def corrected_fit(gram,norm):
    if np.any(norm<=0):
        return None,None,"zero training column"
    normed = gram/norm[:,None]/norm[None,:]
    eig,vec = np.linalg.eigh((normed+normed.T)/2)
    tol = 64*np.finfo(float).eps*len(eig)*max(abs(eig))
    if len(eig)>1 and eig[1]-eig[0]<=tol:
        return None,float(eig[0]),"nonunique direction"
    return vec[:,0]/norm,float(eig[0]),"ok"


def matrices(tr,g,tags):
    p = [tr[f"{g}_{tag}_point"] for tag in tags]
    return (np.vstack([a/np.sqrt(len(a)) for a in p]),
            sum(tr[f"{g}_{tag}_corrected"] for tag in tags))


def load_npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def run(job):
    system,seed = job
    path = record_path(system,seed)
    info = json.loads(path.with_suffix(".json").read_text())
    assert records.sha(path)==info["archive"]
    data = load_npz(path)
    tpath = ROOT/"training"/path.name
    tmeta = json.loads(tpath.with_suffix(".json").read_text())
    assert tmeta["raw"]==info["raw_hash"] and records.sha(tpath)==tmeta["sha256"]
    tr = load_npz(tpath)
    cs = info["candidates"]
    E.K.common_width(max(len(c["xi"]) for c in cs))
    layouts = {}
    for g in range(3):
        np.testing.assert_allclose(tr[f"{g}_local_point"],data[f"train{g}"],rtol=3e-12,atol=3e-13)
        for bank,tags in (("local_time",("local","time")),("broad",("broad",))):
            layouts[g,bank] = matrices(tr,g,tags)
    counters = dict(checks=0,max_angle=0.,max_score_delta=0.,max_replay_delta=0.)
    if seed in REPLAY:
        prior = json.loads((PRIOR/"augmented"/path.with_suffix(".json").name).read_text())
        assert len(prior["rows"])==len(cs)
    rows = []
    for j,c in enumerate(cs):
        ids = sorted([c["yi"],*c["xi"]])
        pivot = ids.index(c["yi"])
        assert len(set(ids))==len(ids)
        sy = [stability(data["dyad_matrix"],[k for k in ids if k!=t],t,info["shape"],E.K) for t in ids]
        dyad = float(np.median(sy))
        legacy_dyad = stability(data["dyad_matrix"],c["xi"],c["yi"],info["shape"],E.K)
        row = dict(id=c["id"],slot=c["slot"],label=c["label"],registered=c["registered"],ids=ids,
                   dyad=dyad,legacy_dyad=legacy_dyad,folds=[])
        if seed in REPLAY:
            archive = prior["rows"][j]
            assert archive["id"]==c["id"] and archive["ids"]==ids
            np.testing.assert_allclose([dyad,legacy_dyad],[archive["symmetric_dyad"],archive["dyad"]],rtol=3e-12)
        for g in range(3):
            left,right,scales = [data[f"{side}{g}"][...,ids] for side in ("left","right","scale")]
            old = data["old_coefficients"][g,ids,j]
            legacy = float((left@old)@(right@old)/scales[pivot]**2)
            fold = dict(legacy=dict(D=legacy,vector=old.tolist(),status="ok"))
            for bank in ("local_time","broad"):
                full,G = layouts[g,bank]
                mat,gram = full[:,ids],G[np.ix_(ids,ids)]
                norm = np.linalg.norm(mat,axis=0)
                for corr in ("off","on"):
                    if corr=="off":
                        v,status = fit(mat)
                        minimum = None
                    else:
                        v,minimum,status = corrected_fit(gram,norm)
                    res = dict(status=status,minimum=minimum,D=None,vector=None)
                    fold[f"{bank}_{corr}"] = res
                    if v is None:
                        continue
                    val = score(left,right,scales,v)
                    if val is None:
                        res["status"] = "zero validation scale"
                        continue
                    assert np.isfinite(val)
                    res.update(D=val,vector=v.tolist())
                    units = np.geomspace(.04,29.,len(ids))*(-1.)**np.arange(len(ids))
                    order = np.arange(len(ids))[::-1]
                    if corr=="off":
                        other,status2 = fit((mat*units)[:,order])
                    else:
                        G2 = (gram*units[:,None]*units[None,:])[np.ix_(order,order)]
                        other,_,status2 = corrected_fit(G2,(norm*abs(units))[order])
                    assert status2==status
                    angle = projective_error(v,other[np.argsort(order)]*units,norm)
                    val2 = score((left*units)[:,order],(right*units)[:,order],(scales*abs(units))[order],other)
                    delta = abs(val-val2)/max(1.,abs(val),abs(val2))
                    counters["checks"] += 1
                    counters["max_angle"] = max(counters["max_angle"],angle)
                    counters["max_score_delta"] = max(counters["max_score_delta"],delta)
                    if seed in REPLAY and bank=="local_time" and corr=="off":
                        delta = abs(val-archive["folds"][g]["augmented"])/max(1,abs(val))
                        counters["max_replay_delta"] = max(counters["max_replay_delta"],delta)
                        assert delta<1.e-9
            if c["label"]=="reference":
                truth = np.zeros(len(ids))
                truth[pivot] = 1.
                for term,coef in zip(c["xi"],c["exact"]):
                    truth[ids.index(term)] = -coef
                for arm in ARMS:
                    v = fold[arm]["vector"]
                    fold[arm]["angle"] = None if v is None else projective_error(
                        np.asarray(v),truth,data[f"clean_scale{g}"][ids])
            row["folds"].append(fold)
        rows.append(row)
    assert counters["max_angle"]<1.e-5 and counters["max_score_delta"]<1.e-6, counters
    out = dict(system=system,seed=seed,record=info["archive"],training=tmeta["sha256"],rows=rows,audit=counters)
    (ROOT/"results"/path.with_suffix(".json").name).write_text(json.dumps(out,allow_nan=False))
    return path.stem


if __name__=="__main__":
    (ROOT/"results").mkdir(exist_ok=True)
    jobs = [(s,n) for n in (*REPLAY,*FRESH) for s in SYSTEMS
            if (ROOT/"training"/f"{s}_{n}.json").exists() and record_path(s,n).with_suffix(".json").exists()
            and not (ROOT/"results"/f"{s}_{n}.json").exists()]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            print(f.result(),flush=True)
