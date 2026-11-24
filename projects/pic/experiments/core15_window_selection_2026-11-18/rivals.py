"""Identify who dominates each refitted law, then change only its coefficients to truth."""
from collections import Counter
import json
from setup_study import ROOT, np, score, records
from score_panel import ARMS
from followup import FOLLOW


def main():
    result={}
    for system in ("ac","kdv_cossin"):
        arms={a:dict(fitted=Counter(),fixed=Counter(),fixed_survival=0,matched_front=0,
                     fixed_better=0,delta=[]) for a in ARMS}
        for seed in FOLLOW:
            stem=f"{system}_{seed}"
            path=ROOT/"records"/(stem+".npz")
            meta=json.loads(path.with_suffix(".json").read_text())
            assert records.sha(path)==meta["archive"]
            c=next(c for c in meta["candidates"] if c["label"]=="reference")
            ids=sorted([c["yi"],*c["xi"]])
            v=np.zeros(len(ids))
            v[ids.index(c["yi"])]=1.
            for k,b in zip(c["xi"],c["exact"]):
                v[ids.index(k)]=-b
            with np.load(path) as z:
                fixed=np.mean([score(z[f"left{g}"][:,ids],z[f"right{g}"][:,ids],z[f"scale{g}"][ids],v)
                               for g in range(3)])
            doc=json.loads((ROOT/"results"/(stem+".json")).read_text())
            ref=next(r for r in doc["rows"] if r["id"]==c["id"])
            for a,counts in arms.items():
                dref=np.mean([f[a]["D"] for f in ref["folds"]])
                observed,fixed_doms=[],[]
                for r in doc["rows"]:
                    if r["id"]==c["id"] or r["dyad"]>ref["dyad"]:
                        continue
                    dr=np.mean([f[a]["D"] for f in r["folds"]])
                    if dr<=dref and (dr<dref or r["dyad"]<ref["dyad"]):
                        observed.append(r["id"])
                    if dr<=fixed and (dr<fixed or r["dyad"]<ref["dyad"]):
                        fixed_doms.append(r["id"])
                assert set(observed)==set(doc["slots"][0]["arms"][a]["reference_dominators"])
                counts["fitted"].update(observed)
                counts["fixed"].update(fixed_doms)
                counts["matched_front"]+=int(not observed)
                counts["fixed_survival"]+=int(not fixed_doms)
                counts["fixed_better"]+=int(fixed<dref)
                counts["delta"].append(float(dref-fixed))
        for a,counts in arms.items():
            counts["median_D_fitted_minus_physical"]=float(np.median(counts.pop("delta")))
        result[system]=arms
    (ROOT/"followup_rivals.json").write_text(json.dumps(result,indent=2))
    print({s:{a:v[a] for a in ("local","middle")} for s,v in result.items()})


if __name__=="__main__":
    main()
