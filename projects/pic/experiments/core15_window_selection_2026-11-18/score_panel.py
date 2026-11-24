"""Evaluate frozen width choices on the independent, common outer holdouts."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from setup_study import ROOT, SYSTEMS, SEEDS, WIDTHS, np, E, records, corrected_fit, score, front
from benchmark import stability
from implicit import projective_error
from scipy.linalg import eigh

ARMS = (*WIDTHS, "shared", "individual", "pooled")


def load(directory, stem):
    path=ROOT/directory/(stem+".npz")
    meta=json.loads(path.with_suffix(".json").read_text())
    assert records.sha(path)==meta["archive"]
    with np.load(path) as z:
        return {k:z[k] for k in z.files}, meta


def evaluate(job):
    system,seed=job
    stem=f"{system}_{seed}"
    tr,tm=load("training",stem)
    choice_path=ROOT/"decisions"/(stem+".json")
    choices=json.loads(choice_path.read_text())
    assert choices["training"]==tm["archive"]
    data,obs=load("records",stem)
    cs=obs["candidates"]
    assert obs["raw_hash"]==tm["raw"] and tm["supports"]==[sorted([c["yi"],*c["xi"]]) for c in cs]
    E.K.common_width(max(len(c["xi"]) for c in cs))
    rows, dyads, num, errors = [], {}, Counter(), Counter()
    for n,c in enumerate(cs):
        ids=tm["supports"][n]
        idx=np.ix_(ids,ids)
        key=tuple(ids)
        if key not in dyads:
            dyads[key]=float(np.median([stability(data["dyad_matrix"],[k for k in ids if k!=p],p,obs["shape"],E.K)
                                        for p in ids]))
        row=dict(id=c["id"],slot=c["slot"],label=c["label"],registered=c["registered"],dyad=dyads[key],folds=[])
        for g,d in enumerate(choices["folds"]):
            l,r,sc=[data[f"{side}{g}"][...,ids] for side in ("left","right","scale")]
            gr,norm=tr[f"{g}_gram"],tr[f"{g}_norm"]
            fold={}
            for w,name in enumerate((*WIDTHS,"pooled")):
                G=(np.mean(gr,axis=0) if name=="pooled" else gr[w])[idx]
                s=(np.sqrt(np.mean(norm**2,axis=0)) if name=="pooled" else norm[w])[ids]
                v,_,status=corrected_fit(G,s)
                res=dict(D=None,status=status,angle=None)
                if status=="ok":
                    val=score(l,r,sc,v)
                    assert val is not None and np.isfinite(val)
                    res["D"]=val
                    # An alternate generalized eigensolve and an explicit held-out quadratic form.
                    alt=eigh((G+G.T)/2,np.diag(s*s),driver="gvx",subset_by_index=(0,0))[1][:,0]
                    independent=alt@(l.T@r)@alt/np.sum((sc*alt)**2)
                    diff=abs(val-independent)/max(1,abs(val))
                    errors["alternate_score"]=max(errors["alternate_score"],diff)
                    assert diff<2e-6, (system,c["id"],diff)
                    units=np.geomspace(.13,9,len(ids))*(-1.)**np.arange(len(ids))
                    perm=np.arange(len(ids))[::-1]
                    varied,_,sts=corrected_fit((G*np.outer(units,units))[np.ix_(perm,perm)],(s*abs(units))[perm])
                    assert sts=="ok"
                    dd=score(l,r,sc,varied[np.argsort(perm)]*units)
                    diff=abs(dd-val)/max(1,abs(val))
                    errors["unit_score"]=max(errors["unit_score"],diff)
                    assert diff<2e-6, (system,c["id"],diff)
                    if c["label"]=="reference":
                        truth=np.zeros(len(ids))
                        truth[ids.index(c["yi"])]=1.
                        for k,b in zip(c["xi"],c["exact"]):
                            truth[ids.index(k)]=-b
                        res["angle"]=float(np.degrees(projective_error(v,truth,data[f"clean_scale{g}"][ids])))
                num[status]+=1
                fold[name]=res
            fold["shared"]=fold[WIDTHS[d["shared"]]]
            fold["individual"]=fold[WIDTHS[d["personal"][n]]]
            row["folds"].append(fold)
        rows.append(row)
    slots=[]
    for slot in sorted({c["slot"] for c in cs}):
        sub=[r for r in rows if r["slot"]==slot]
        slots.append(dict(slot=slot,arms={arm:front(sub,arm) for arm in ARMS}))
    result=dict(rows=rows,slots=slots,checks=dict(num),max_error=dict(errors),choice_hash=records.sha(choice_path),
                training_hash=tm["archive"],record_hash=obs["archive"],pooling=records.sha(ROOT/"pooling_protocol.txt"))
    (ROOT/"results"/(stem+".json")).write_text(json.dumps(result,allow_nan=False))
    return system,seed


if __name__=="__main__":
    (ROOT/"results").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(evaluate,(s,n)) for n in SEEDS for s in SYSTEMS]):
            f.result()
