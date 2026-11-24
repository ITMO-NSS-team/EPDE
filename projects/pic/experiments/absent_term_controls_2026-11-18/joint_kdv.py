"""Shared intrinsic training Grams, scored only on the forced field."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import csv
from controls import ROOT,E,np,records,corrected_fit,score
from evaluate import matrices
import train

DEST=ROOT/"kdv_joint"
N=16


def canonical(label):
    return json.dumps(label,sort_keys=True,separators=(",",":"))


def pair(i):
    a,b=786001+i,787001+i
    records.ROOT=DEST
    train.ROOT=DEST
    fr=DEST/"records"/f"kdv_cossin_{a}.npz"
    ft=DEST/"training"/fr.name
    ut=DEST/"training"/f"kdv_{b}.npz"
    if not fr.with_suffix(".json").exists():
        records.build(("kdv_cossin",a))
    for system,seed,path in (("kdv_cossin",a,ft),("kdv",b,ut)):
        if not path.with_suffix(".json").exists():
            train.run((system,seed))
    fm=json.loads(fr.with_suffix(".json").read_text())
    tm=json.loads(ft.with_suffix(".json").read_text())
    um=json.loads(ut.with_suffix(".json").read_text())
    assert tm["raw"]==fm["raw_hash"] and records.sha(fr)==fm["archive"]
    assert records.sha(ft)==tm["sha256"] and records.sha(ut)==um["sha256"]
    p=E.R.datasets.load("kdv")
    cs,labels=E.R.panel(p)
    uref=next(c for c in cs if c["label"]=="reference")
    common={canonical(labels[k]):k for k in [uref["yi"],*uref["xi"]]}
    cs=[c for c in fm["candidates"] if c["label"]=="reference" or c["label"].startswith("add_")]
    rc=next(c for c in cs if c["label"]=="reference")
    assert set(common)<={canonical(fm["labels"][k]) for k in [rc["yi"],*rc["xi"]]}
    scores={a:{c["id"]:[] for c in cs} for a in ("single","joint")}
    co={a:[] for a in scores}
    fixed=[]
    with np.load(fr) as obs,np.load(ft) as ftrain,np.load(ut) as utrain:
        for g in range(3):
            F,fg=matrices(ftrain,g,("local","time"))
            U,ug=matrices(utrain,g,("local","time"))
            for c in cs:
                ids=sorted([c["yi"],*c["xi"]])
                l,r,scale=[obs[f"{k}{g}"][...,ids] for k in ("left","right","scale")]
                pos=[j for j,k in enumerate(ids) if canonical(fm["labels"][k]) in common]
                cols=[common[canonical(fm["labels"][ids[j]])] for j in pos]
                assert len(pos)==3
                base=fg[np.ix_(ids,ids)]
                norm=np.linalg.norm(F[:,ids],axis=0)
                joint=base.copy()
                joint[np.ix_(pos,pos)]+=ug[np.ix_(cols,cols)]
                combined=norm.copy()
                combined[pos]=np.hypot(norm[pos],np.linalg.norm(U[:,cols],axis=0))
                for kind,G,n in (("single",base,norm),("joint",joint,combined)):
                    v,_,status=corrected_fit(G,n)
                    assert status=="ok"
                    val=score(l,r,scale,v)
                    np.testing.assert_allclose(val,v@(l.T@r)@v/sum((v*scale)**2),atol=2e-12,rtol=2e-8)
                    units=(-1.)**np.arange(len(ids))*np.geomspace(.1,10.,len(ids))
                    order=np.arange(len(ids))[::-1]
                    mapped,_,s=corrected_fit((G*units[:,None]*units[None,:])[np.ix_(order,order)],(n*abs(units))[order])
                    assert s=="ok"
                    v2=mapped[np.argsort(order)]*units
                    np.testing.assert_allclose(score(l,r,scale,v2),val,atol=2e-10,rtol=1e-8)
                    scores[kind][c["id"]].append(val)
                    if c["label"]=="reference":
                        co[kind].append([-v[ids.index(k)]/v[ids.index(c["yi"])] for k in c["xi"]])
                if c["label"]=="reference":
                    v=np.zeros(len(ids))
                    v[ids.index(c["yi"])]=1.
                    for k,z in zip(c["xi"],c["exact"]):
                        v[ids.index(k)]=-z
                    fixed.append(score(l,r,scale,v))
    return dict(forced_seed=a,unforced_seed=b,forced_hash=tm["raw"],unforced_hash=um["raw"],
                scores=scores,co=co,fixed=fixed)


def main():
    for d in ("training","records"):
        (DEST/d).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows=[f.result() for f in as_completed([pool.submit(pair,i) for i in range(N)])]
    rows.sort(key=lambda r:r["forced_seed"])
    for k in ("forced_hash","unforced_hash"):
        assert len({tuple(sorted(r[k].items())) for r in rows})==N
    (DEST/"folds.json").write_text(json.dumps(rows,allow_nan=False))
    by={}
    for arm in ("single","joint"):
        result={}
        for key in rows[0]["scores"][arm]:
            if ":add_" not in key:
                continue
            vals=[np.mean(r["scores"][arm]["e0:reference"])-np.mean(r["scores"][arm][key]) for r in rows]
            result[key]=dict(wins=int(sum(x>0 for x in vals)),median=float(np.median(vals)))
        ref=np.array([np.mean(r["scores"][arm]["e0:reference"]) for r in rows])
        fixed=np.array([np.mean(r["fixed"]) for r in rows])
        by[arm]=dict(add_wins=sum(x["wins"] for x in result.values()),comparisons=N*len(result),
            add_results=result,reference_medians=np.median([np.median(r["co"][arm],axis=0) for r in rows],axis=0).tolist(),
            ref_median_score=float(np.median(ref)),physical_median_score=float(np.median(fixed)),
            ref_minus_physical_median=float(np.median(ref-fixed)))
    joint_improves=sum(np.mean(r["scores"]["joint"]["e0:reference"])<np.mean(r["scores"]["single"]["e0:reference"]) for r in rows)
    out=dict(arms=by,fields=N,joint_reference_improves=int(joint_improves),protocol=records.sha(ROOT/"joint_protocol.txt"))
    (DEST/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print(out)


if __name__=="__main__":
    main()
