"""Compare all supports with common held-outs and distinguish one-sided oracle controls."""
from concurrent.futures import ProcessPoolExecutor
import json
from common import ROOT, SEEDS, E, corrected_fit, matrices, np, records, score, stability
from setup_study import front

ARMS=("local","mean_staged","low_staged","low_only","pooled","odd_local","disjoint_mean","disjoint_low")
ORACLES=("oracle_reference","oracle_pair","oracle_all")


def bundle(directory,seed,hash_key="archive",base=ROOT):
    path=base/directory/f"ac_{seed}.npz"
    meta=json.loads(path.with_suffix(".json").read_text())
    assert records.sha(path)==meta[hash_key]
    with np.load(path) as archive:
        return {k:archive[k] for k in archive.files},meta


def restrict(block,ids):
    G,n=block
    return G[np.ix_(ids,ids)],n[ids]


def fit(block):
    v,_,status=corrected_fit(*block)
    return v,status


def constrained(block,kept,direction):
    """Preserve a homogeneous subdirection, freeing the other coordinates."""
    G,n=block
    columns=np.zeros(len(n))
    columns[kept]=direction
    if len(kept)==len(n):
        return columns,"ok"
    other=[i for i in range(len(n)) if i not in kept]
    basis=np.column_stack([columns,np.eye(len(n))[:,other]])
    z,status=fit((basis.T@G@basis,np.linalg.norm(n[:,None]*basis,axis=0)))
    return (None,status) if z is None else (basis@z,status)


def direction(arm,blocks,lap):
    local=blocks["odd" if arm.startswith("disjoint") or arm=="odd_local" else "local"]
    source="even" if arm.startswith("disjoint") else "full"
    if arm in ("local","odd_local"):
        return fit(local)
    if arm=="pooled":
        G,n=local
        S,m=blocks["full_low"]
        return fit((G+S,np.hypot(n,m)))
    bank="mean" if "mean" in arm else "high" if "high" in arm else "low"
    spectral=blocks[f"{source}_{bank}"]
    selected=[i for i in range(len(local[1])) if i!=lap]
    measured=selected if bank=="mean" else list(range(len(local[1])))
    v,status=fit(restrict(spectral,measured))
    if v is None or arm in ("low_only","high_only"):
        return v,status
    nonspatial=v if bank=="mean" else v[selected]
    return constrained(local,selected,nonspatial)


def fold_blocks(training,measurement,g):
    X,G=matrices(training,g,("local","time"))
    out={"local":(G,np.linalg.norm(X,axis=0))}
    for key in ("odd","full_mean","full_low","even_mean","even_low"):
        out[key]=(measurement[f"{g}_{key}_gram"],measurement[f"{g}_{key}_norm"])
    if f"{g}_full_high_gram" in measurement:
        out["full_high"]=(measurement[f"{g}_full_high_gram"],measurement[f"{g}_full_high_norm"])
    return out


def oracle(local,ids,target,coeff):
    subset=[ids.index(i) for i in [target,*coeff] if i in ids]
    vector=np.array([1.,*[-c for i,c in coeff.items() if i in ids]])
    v,status=constrained(local,subset,vector)
    if v is not None:
        for i,c in coeff.items():
            if i in ids:
                np.testing.assert_allclose(-v[ids.index(i)]/v[ids.index(target)],c,rtol=1e-12)
    return v,status


def evaluate(seed,directory=ROOT,arms=ARMS):
    measurement,mm=bundle("measurements",seed,base=directory)
    training,tm=bundle("training",seed,"sha256",base=directory)
    obs,meta=bundle("records",seed,base=directory)
    assert mm["raw"]==tm["raw"]["u"]==meta["raw_hash"]["u"]
    assert mm["protocol"]==records.sha(ROOT/"protocol.txt")
    candidates=meta["candidates"]
    E.K.common_width(max(len(c["xi"]) for c in candidates))
    all_blocks=[fold_blocks(training,measurement,g) for g in range(3)]
    rows=[]
    max_error=0.
    names=mm["names"]
    reference=next(c for c in candidates if c["label"]=="reference")
    exact={i:v for i,v in zip(reference["xi"],reference["exact"]) if names[i] in ("cube","u")}
    for c in candidates:
        ids=sorted([c["yi"],*c["xi"]])
        lap=ids.index(names.index("laplacian")) if names.index("laplacian") in ids else None
        dyad=float(np.median([stability(obs["dyad_matrix"],[j for j in ids if j!=k],k,meta["shape"],E.K) for k in ids]))
        row=dict(id=c["id"],label=c["label"],registered=c["registered"],dyad=dyad,ids=ids,folds=[])
        for g,banks in enumerate(all_blocks):
            blocks={name:restrict(b,ids) for name,b in banks.items()}
            left,right,norm=[obs[f"{key}{g}"][...,ids] for key in ("left","right","scale")]
            order=np.arange(len(ids))[::-1]
            units=np.geomspace(.2,5,len(ids))*(-1.)**np.arange(len(ids))
            changed={k:((G*np.outer(units,units))[np.ix_(order,order)],(n*abs(units))[order])
                     for k,(G,n) in blocks.items()}
            position=None if lap is None else list(order).index(lap)
            fold={}
            for arm in (*arms,*ORACLES):
                if arm in ORACLES:
                    apply=(c["label"]=="reference" or arm=="oracle_all" or
                           (arm=="oracle_pair" and c["label"]=="drop_0"))
                    if not apply:
                        fold[arm]=fold["local"]
                        continue
                    v,status=oracle(blocks["local"],ids,c["yi"],exact)
                else:
                    v,status=direction(arm,blocks,lap)
                result=dict(D=None,status=status)
                if v is not None:
                    d=score(left,right,norm,v)
                    assert d is not None and np.isfinite(d)
                    result["D"]=d
                    direct=v@(left.T@right)@v/np.sum((norm*v)**2)
                    np.testing.assert_allclose(d,direct,atol=3e-12,rtol=2e-8)
                    if arm in arms:
                        z,status2=direction(arm,changed,position)
                        assert status2==status
                        d2=score(left,right,norm,z[np.argsort(order)]*units)
                        diff=abs(d2-d)/max(1,abs(d))
                        max_error=max(max_error,diff)
                        assert diff<1e-6,(seed,c["id"],arm,diff)
                    if c["label"] in ("reference","drop_0"):
                        pivot=v[ids.index(c["yi"])]
                        result["coeff"]={names[i]:float(-v[ids.index(i)]/pivot) for i in c["xi"]}
                fold[arm]=result
            row["folds"].append(fold)
        rows.append(row)
    fronts={a:front(rows,a) for a in (*arms,*ORACLES)}
    result=dict(seed=seed,rows=rows,fronts=fronts,unit_error=max_error,raw_hash=mm["raw"],
                record=meta["archive"],training=tm["sha256"],measurement=mm["archive"])
    (directory/"results"/f"ac_{seed}.json").write_text(json.dumps(result,allow_nan=False))
    return seed


if __name__=="__main__":
    (ROOT/"results").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed in pool.map(evaluate,SEEDS):
            print(seed,flush=True)
