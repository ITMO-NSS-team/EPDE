"""Benchmark-only ratio locks. Refit all surviving free parameters at each lock."""
from itertools import combinations
import json
from setup_study import ROOT, np, corrected_fit, score
from followup import FOLLOW


def diagnose(system):
    names=(("diffusion","cubic","linear") if system=="ac" else ("nonlinear","dispersion","forcing"))
    out={"local":{},"middle":{}}
    for seed in FOLLOW:
        stem=f"{system}_{seed}"
        meta=json.loads((ROOT/"records"/(stem+".json")).read_text())
        case=next(c for c in meta["candidates"] if c["label"]=="reference")
        support=sorted([case["yi"],*case["xi"]])
        pivot=support.index(case["yi"])
        terms=[support.index(i) for i in case["xi"]]
        doc=json.loads((ROOT/"results"/(stem+".json")).read_text())
        ref=next(r for r in doc["rows"] if r["id"]==case["id"])
        with np.load(ROOT/"training"/(stem+".npz")) as tr, np.load(ROOT/"records"/(stem+".npz")) as z:
            for w,arm in enumerate(out):
                rivals=[(r["dyad"],np.mean([f[arm]["D"] for f in r["folds"]]))
                        for r in doc["rows"] if r["id"]!=case["id"] and r["dyad"]<=ref["dyad"]]
                for size in range(4):
                    for locked in combinations(range(3),size):
                        free=[j for j in range(len(support)) if j not in [terms[i] for i in locked]]
                        basis=np.eye(len(support))[:,free]
                        for i in locked:
                            basis[terms[i],free.index(pivot)]=-case["exact"][i]
                        values=[]
                        for g in range(3):
                            gram=tr[f"{g}_gram"][w][np.ix_(support,support)]
                            norm=tr[f"{g}_norm"][w,support]
                            submass=np.linalg.norm(norm[:,None]*basis,axis=0)
                            direction,_,status=corrected_fit(basis.T@gram@basis,submass)
                            assert status=="ok"
                            v=basis@direction
                            for i in locked:
                                assert np.isclose(-v[terms[i]]/v[pivot],case["exact"][i])
                            values.append(score(z[f"left{g}"][:,support],z[f"right{g}"][:,support],z[f"scale{g}"][support],v))
                        d=np.mean(values)
                        if not locked:
                            np.testing.assert_allclose(values,[f[arm]["D"] for f in ref["folds"]],rtol=1e-9,atol=1e-12)
                        survives=not any(d2<=d and (d2<d or y<ref["dyad"]) for y,d2 in rivals)
                        name="+".join(names[i] for i in locked) or "none"
                        out[arm][name]=out[arm].get(name,0)+int(survives)
    return out


if __name__=="__main__":
    result={s:diagnose(s) for s in ("ac","kdv_cossin")}
    control=json.loads((ROOT/"followup_rivals.json").read_text())
    for s in result:
        for arm,values in result[s].items():
            assert values["none"]==control[s][arm]["matched_front"]
            assert next(reversed(values.values()))==control[s][arm]["fixed_survival"]
    (ROOT/"coefficient_controls.json").write_text(json.dumps(result,indent=2))
    print(result)
