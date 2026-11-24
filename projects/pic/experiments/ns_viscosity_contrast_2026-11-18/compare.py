"""Compare free fits on identical validation data; independently check replay."""
import csv
import json
import sys
from holdouts import E, FRESH, NAMES, PRIOR, REPLAY, ROOT, digest, np
sys.path.insert(0,str(PRIOR))
from implicit import fit, projective_error, score


BANKS = ("local","time","x1","x2","xy","broad0")


def eig_fit(gram,scales):
    normed = gram/scales[:,None]/scales[None,:]
    eig,basis = np.linalg.eigh((normed+normed.T)/2)
    tol = 64*np.finfo(float).eps*len(eig)*np.max(abs(eig))
    if eig[1]-eig[0]<=tol or np.any(scales<=0):
        raise ValueError("unresolved coefficient direction")
    return basis[:,0]/scales, float(eig[0]), tol


def load_record(directory,seed):
    path = ROOT/directory/f"{seed}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert digest(path)==meta["sha256"]
    with np.load(path) as z:
        return {k:z[k] for k in z.files},meta


def compare(seed):
    data,vmeta = load_record("holdouts",seed)
    tr,tmeta = load_record("training",seed)
    ids = data["ids"]
    assert np.array_equal(ids,tr["ids"])
    if seed in FRESH:
        assert tmeta["raw"]==vmeta["source"]
    else:
        historical = json.loads((PRIOR/"augmented"/f"ns_{seed}.json").read_text())
        archived = {r["id"]:r for r in historical["rows"]}
        raw_meta = json.loads((PRIOR/"records"/f"ns_{seed}.json").read_text())
        assert tmeta["raw"]==raw_meta["raw_hash"]
    target,missing = int(tr["target"]),int(tr["missing"])
    rows = []
    max_angle = 0.
    max_score_delta = 0.
    max_anchor_error = 0.
    max_replay_error = 0.
    for g in range(3):
        for bank in BANKS:
            tags = (bank,) if bank in ("local","broad0") else ("local",bank)
            mats = [tr[f"{g}_{tag}_point"] for tag in tags]
            pilot = np.vstack([m/np.sqrt(len(m)) for m in mats])
            grams = {kind:sum(tr[f"{g}_{tag}_{kind}"] for tag in tags) for kind in ("ols","clsm")}
            np.testing.assert_allclose(grams["ols"],(pilot.conj().T@pilot).real,rtol=3e-12,atol=3e-12)
            for k,name in enumerate(NAMES):
                ix = np.arange(len(ids)) if k==0 else np.delete(np.arange(len(ids)),missing)
                pivot = int(np.nonzero(ix==target)[0][0])
                left,right,scale = (data[f"{key}{g}"][...,ix] for key in ("left","right","scale"))
                matrix = pilot[:,ix]
                norms = np.linalg.norm(matrix,axis=0)
                for kind,gram in grams.items():
                    G = gram[np.ix_(ix,ix)]
                    vec,e0,tol = eig_fit(G,norms)
                    if kind=="ols":
                        sv,status = fit(matrix)
                        assert status=="ok"
                        assert projective_error(sv,vec,norms)<1.e-8
                    value = score(left,right,scale,vec)
                    order = np.roll(np.arange(len(ix))[::-1],1)
                    units = np.geomspace(0.04,29,len(ix))*(-1.)**np.arange(len(ix))
                    mapped_gram = (G*units[:,None]*units[None,:])[np.ix_(order,order)]
                    other,_,_ = eig_fit(mapped_gram,(norms*abs(units))[order])
                    angle = projective_error(vec,other[np.argsort(order)]*units,norms)
                    err = abs(score((left*units)[:,order],(right*units)[:,order],
                                    (scale*abs(units))[order],other)-value)
                    assert angle<1.e-8 and err<1.e-9
                    max_angle,max_score_delta = max(max_angle,angle),max(max_score_delta,err)
                    if seed in REPLAY and bank=="time" and kind=="ols":
                        expect = archived[name]["folds"][g]["augmented"]
                        max_replay_error = max(max_replay_error,abs(value-expect))
                        np.testing.assert_allclose(value,expect,rtol=1.e-8,atol=2.e-13)
                    nu = float(-vec[np.nonzero(ix==missing)[0][0]]/vec[pivot]) if k==0 else None
                    rows.append(dict(seed=seed,fold=g,bank=bank,fit=kind,candidate=name,D=value,
                                     viscosity=nu,negative_minimum=bool(e0 < -tol),
                                     minimum=e0,ids=ids[ix].tolist(),coefficients=vec.tolist()))
                    if bank=="broad0" and kind=="clsm":
                        rhs = [i for i in range(len(ix)) if i!=pivot]
                        coef,_ = E.R.signed_fit(G,matrix,rhs,pivot)
                        directed = np.ones(len(ix))
                        directed[rhs] = -coef
                        anchor = data["anchors"][g,ix,k]
                        np.testing.assert_allclose(directed,anchor,rtol=1.e-7,atol=2.e-8)
                        max_anchor_error = max(max_anchor_error,float(np.max(abs(directed-anchor))))
                        val = score(left,right,scale,directed)
                        rows.append(dict(seed=seed,fold=g,bank=bank,fit="directed_clsm",
                            candidate=name,D=val,viscosity=float(-directed[np.nonzero(ix==missing)[0][0]]) if k==0 else None,
                            negative_minimum=None,minimum=None,ids=ids[ix].tolist(),coefficients=directed.tolist()))
    return rows,dict(angle=max_angle,score=max_score_delta,anchor=max_anchor_error,replay=max_replay_error)


def main():
    rows,checks = [],[]
    for seed in (*REPLAY,*FRESH):
        per_seed,audit = compare(seed)
        rows.extend(per_seed)
        checks.append(audit)
    (ROOT/"fits.json").write_text(json.dumps(rows,allow_nan=False))
    paired = []
    for seed in (*REPLAY,*FRESH):
        for bank,kind in dict.fromkeys((r["bank"],r["fit"]) for r in rows):
            subs = [[r for r in rows if (r["seed"],r["bank"],r["fit"],r["candidate"])==
                     (seed,bank,kind,name)] for name in NAMES]
            full,drop = [float(np.mean([r["D"] for r in rr])) for rr in subs]
            assert all(len(rr)==3 for rr in subs)
            paired.append(dict(seed=seed,cohort="fresh" if seed in FRESH else "replay",
                bank=bank,fit=kind,full=full,drop=drop,drop_minus_full=drop-full,
                median_viscosity=float(np.median([r["viscosity"] for r in subs[0]]))))
    with (ROOT/"paired.csv").open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)
    table = []
    for cohort,bank,kind in dict.fromkeys((r["cohort"],r["bank"],r["fit"]) for r in paired):
        cases = [r for r in paired if (r["cohort"],r["bank"],r["fit"])==(cohort,bank,kind)]
        delta = np.array([r["drop_minus_full"] for r in cases])
        table.append(dict(cohort=cohort,bank=bank,fit=kind,wins=int(sum(delta>0)),ties=int(sum(delta==0)),
            total=len(delta),median_delta=float(np.median(delta)),range=[float(min(delta)),float(max(delta))],
            median_viscosity=float(np.median([r["median_viscosity"] for r in cases]))))
    maxima = {key:max(r[key] for r in checks) for key in checks[0]}
    (ROOT/"summary.json").write_text(json.dumps(dict(table=table,audit=maxima,
        source_sha256={name:digest(ROOT/name) for name in ("holdouts.py","training.py","compare.py","protocol.txt")},
        paired_sha256=digest(ROOT/"paired.csv"),fits_sha256=digest(ROOT/"fits.json")),indent=2,allow_nan=False))
    for r in table:
        print(r["cohort"],r["bank"],r["fit"],r["wins"],r["total"],
              f'{r["median_delta"]:.6g}',f'{r["median_viscosity"]:.6g}')
    print("checks",maxima)


if __name__=="__main__":
    main()
