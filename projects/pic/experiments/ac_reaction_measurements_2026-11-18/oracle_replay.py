"""Revisit the exact 24 prior observations under matched reaction coefficients."""
import json
from common import PRIOR, ROOT, np, score
from compare import oracle


def revisit():
    previous=json.loads((PRIOR/"coefficient_controls.json").read_text())["ac"]
    out={}
    for w,arm in enumerate(("local","middle")):
        totals={k:dict(front=0,diffusion=0) for k in ("original","reference_only","matched")}
        for n in range(793001,793025):
            p=f"ac_{n}"
            meta=json.loads((PRIOR/"records"/(p+".json")).read_text())
            data=json.loads((PRIOR/"results"/(p+".json")).read_text())
            ref=next(c for c in meta["candidates"] if c["label"]=="reference")
            coeff=dict(zip(ref["xi"][1:],ref["exact"][1:]))
            scores={r["label"]:np.mean([f[arm]["D"] for f in r["folds"]]) for r in data["rows"]}
            y={r["label"]:r["dyad"] for r in data["rows"]}
            calls={}
            with np.load(PRIOR/"training"/(p+".npz")) as tr, np.load(PRIOR/"records"/(p+".npz")) as obs:
                for c in (ref,next(c for c in meta["candidates"] if c["label"]=="drop_0")):
                    ids=sorted([c["yi"],*c["xi"]])
                    vals=[]
                    for g in range(3):
                        b=tr[f"{g}_gram"][w][np.ix_(ids,ids)],tr[f"{g}_norm"][w,ids]
                        v,status=oracle(b,ids,c["yi"],coeff)
                        assert status=="ok"
                        vals.append(score(obs[f"left{g}"][:,ids],obs[f"right{g}"][:,ids],obs[f"scale{g}"][ids],v))
                    calls[c["label"]]=np.mean(vals)
            for k,changes in (("original",{}),("reference_only",{"reference":calls["reference"]}),("matched",calls)):
                d=scores|changes
                dominates=any(d[key]<=d["reference"] and y[key]<=y["reference"] and
                              (d[key]<d["reference"] or y[key]<y["reference"]) for key in d if key!="reference")
                totals[k]["front"]+=int(not dominates)
                totals[k]["diffusion"]+=int(d["drop_0"]>d["reference"])
        assert totals["original"]["front"]==previous[arm]["none"]
        assert totals["reference_only"]["front"]==previous[arm]["cubic+linear"]
        out[arm]=totals
    (ROOT/"prior_oracle_replay.json").write_text(json.dumps(out,indent=2))
    print(out)


if __name__=="__main__":
    revisit()
