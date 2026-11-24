"""Post hoc coefficient and bank checks on the 32 saved forced KdV fields."""
import json
import csv
from controls import ROOT,np,corrected_fit,score
from evaluate import matrices

dest=ROOT/"kdv_confirmation"
details=[]
coefficients=[]
for path in sorted((dest/"records").glob("*.npz")):
    meta=json.loads(path.with_suffix(".json").read_text())
    cs=meta["candidates"]
    ref=next(c for c in cs if c["label"]=="reference")
    adds=[c for c in cs if c["label"].startswith("add_")]
    ids=sorted([ref["yi"],*ref["xi"]])
    exact=np.zeros(len(ids))
    exact[ids.index(ref["yi"])]=1.
    for key,coeff in zip(ref["xi"],ref["exact"]):
        exact[ids.index(key)]=-coeff
    with np.load(path) as obs,np.load(dest/"training"/path.name) as tr:
        for kind,tags in (("local_time",("local","time")),("broad",("broad",))):
            folds={c["id"]:[] for c in adds}
            estimates=[]
            ref_scores=[]
            for g in range(3):
                point,gram=matrices(tr,g,tags)
                norm=np.linalg.norm(point,axis=0)
                def fitted(ix):
                    v,_,status=corrected_fit(gram[np.ix_(ix,ix)],norm[ix])
                    assert status=="ok"
                    return v
                def value(ix,v):
                    left,right,scale=[obs[f"{k}{g}"][...,ix] for k in ("left","right","scale")]
                    return score(left,right,scale,v)
                base=fitted(ids)
                assert abs(base[ids.index(ref["yi"])])>0.
                reference=value(ids,base)
                truth=value(ids,exact)
                estimates.append([-base[ids.index(k)]/base[ids.index(ref["yi"])] for k in ref["xi"]])
                ref_scores.append((reference,truth))
                for c in adds:
                    ix=sorted([c["yi"],*c["xi"]])
                    extra,=set(ix)-set(ids)
                    fitted_add=fitted(ix)
                    no_extra=fitted_add.copy()
                    no_extra[ix.index(extra)]=0.
                    extended=value(ix,fitted_add)
                    truncated=value(ix,no_extra)
                    folds[c["id"]].append((reference-extended,truncated-extended,truth-extended))
            rs=np.mean(ref_scores,axis=0)
            coefficients.append(dict(seed=meta["seed"],bank=kind,exp=ref["exact"],source=ref["source"],
                median_coefficients=np.median(estimates,axis=0).tolist(),reference=rs[0],physical=rs[1]))
            for key,values in folds.items():
                margin,clamp,fixed=np.mean(values,axis=0)
                details.append(dict(seed=meta["seed"],bank=kind,candidate=key,add_margin=margin,
                    zero_without_refit_margin=clamp,fixed_ref_minus_fitted_add=fixed))
with (dest/"bank_check.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,details[0].keys())
    w.writeheader()
    w.writerows(details)
out={}
for kind in ("local_time","broad"):
    cases=[row for row in details if row["bank"]==kind]
    co=[r for r in coefficients if r["bank"]==kind]
    out[kind]=dict(fields=len(co),candidate_cases=len(cases),with_wins=sum(r["add_margin"]>0 for r in cases),
        with_wins_when_others_fixed=sum(r["zero_without_refit_margin"]>0 for r in cases),
        fitted_addition_beats_fixed_truth=sum(r["fixed_ref_minus_fitted_add"]>0 for r in cases),
        reference_beats_fixed_truth=sum(r["reference"]<r["physical"] for r in co),
        exact=co[0]["exp"],reference_equation=co[0]["source"],
        reference_medians=np.median([r["median_coefficients"] for r in co],axis=0).tolist())
assert out["local_time"]["with_wins"]==sum(int(r["with_wins"]) for r in
    csv.DictReader((dest/"outcomes.csv").read_text().splitlines()) if ":add_" in r["candidate"])
(dest/"bank_summary.json").write_text(json.dumps(out,indent=2,default=int))
print(out)
