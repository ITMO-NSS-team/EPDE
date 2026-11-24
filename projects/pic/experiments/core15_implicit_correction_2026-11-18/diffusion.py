"""Retrospective check of the two contrasting diffusion deletions."""
import csv
import json
from panel import ROOT,FRESH,record_path
from evaluate import load_npz,score,np


def main():
    output=[]
    for system,competitor in (("ac","e0:drop_0"),("ns","e0:drop_3")):
        for seed in FRESH:
            path=record_path(system,seed)
            info=json.loads(path.with_suffix(".json").read_text())
            data=load_npz(path)
            results=json.loads((ROOT/"results"/path.with_suffix(".json").name).read_text())
            fits={r["id"]:r for r in results["rows"]}
            cases={c["id"]:c for c in info["candidates"]}
            ref=cases["e0:reference"]
            ids=fits[ref["id"]]["ids"]
            extra,=set(ids)-set(fits[competitor]["ids"])
            diff=ids.index(extra)
            target=ids.index(ref["yi"])
            truth=np.zeros(len(ids))
            truth[target]=1.
            for term,coef in zip(ref["xi"],ref["exact"]):
                truth[ids.index(term)]=-coef
            fixed_drop=truth.copy()
            fixed_drop[diff]=0.
            folds=[]
            for g in range(3):
                left,right,sc=[data[f"{side}{g}"][...,ids] for side in ("left","right","scale")]
                fixed_full=score(left,right,sc,truth)
                fixed_reduced=score(left,right,sc,fixed_drop)
                fold=dict(fixed_full=fixed_full,fixed_reduced=fixed_reduced,
                    fixed_margin=fixed_reduced-fixed_full,
                    true_diffusion_scale_share=abs(sc[diff]*truth[diff])/np.linalg.norm(sc*truth))
                for arm in ("local_time_off","local_time_on","broad_off","broad_on","legacy"):
                    full=fits[ref["id"]]["folds"][g][arm]
                    reduced=fits[competitor]["folds"][g][arm]
                    fold[arm+"_margin"]=reduced["D"]-full["D"]
                    if arm!="legacy":  # fixed_full uses the symmetric denominator
                        fold[arm+"_drop_minus_fixed_full"]=reduced["D"]-fixed_full
                    v=full["vector"]
                    fold[arm+"_viscosity"]=-v[diff]/v[target]
                folds.append(fold)
            row=dict(system=system,seed=seed,registered_viscosity=-truth[diff],
                     dyad_full=fits[ref["id"]]["dyad"],dyad_reduced=fits[competitor]["dyad"])
            row.update({k:float(np.median([f[k] for f in folds]) if k.endswith("_viscosity")
                                    else np.mean([f[k] for f in folds])) for k in folds[0]})
            output.append(row)
    with (ROOT/"diffusion_pairs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    for system in ("ac","ns"):
        rows=[r for r in output if r["system"]==system]
        print(system)
        for k in rows[0]:
            if k.endswith(("margin","drop_minus_fixed_full","viscosity","scale_share")):
                v=[r[k] for r in rows]
                print(k,"positive",sum(a>0 for a in v),"median",float(np.median(v)))
        print("reduced dyad wins",sum(r["dyad_reduced"]<r["dyad_full"] for r in rows))


if __name__=="__main__":
    main()
