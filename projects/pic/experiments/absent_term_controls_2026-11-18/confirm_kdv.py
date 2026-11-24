"""Full-panel prospective replay for the system singled out by the archive."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
from controls import ROOT,np,records,corrected_fit,score
from evaluate import matrices
import train

DEST=ROOT/"kdv_confirmation"
SEEDS=tuple(range(785001,785033))


def case(seed):
    records.ROOT=DEST
    train.ROOT=DEST
    name=f"kdv_cossin_{seed}.npz"
    rp,tp=DEST/"records"/name,DEST/"training"/name
    if not rp.with_suffix(".json").exists():
        records.build(("kdv_cossin",seed))
    if not tp.with_suffix(".json").exists():
        train.run(("kdv_cossin",seed))
    meta=json.loads(rp.with_suffix(".json").read_text())
    tm=json.loads(tp.with_suffix(".json").read_text())
    assert tm["raw"]==meta["raw_hash"]
    assert records.sha(rp)==meta["archive"] and records.sha(tp)==tm["sha256"]
    cs=[c for c in meta["candidates"] if c["label"]=="reference" or c["label"].startswith(("add_","drop_"))]
    scores={c["id"]:[] for c in cs}
    fixed=[]
    with np.load(rp) as z,np.load(tp) as tr:
        for g in range(3):
            matrix,gram=matrices(tr,g,("local","time"))
            for c in cs:
                ids=sorted([c["yi"],*c["xi"]])
                v,_,status=corrected_fit(gram[np.ix_(ids,ids)],np.linalg.norm(matrix[:,ids],axis=0))
                assert status=="ok"
                l,r,scale=[z[f"{k}{g}"][...,ids] for k in ("left","right","scale")]
                value=score(l,r,scale,v)
                np.testing.assert_allclose(value,v@(l.T@r)@v/sum((v*scale)**2),atol=2e-12,rtol=2e-8)
                scores[c["id"]].append(value)
                if c["label"]=="reference":
                    exact=np.zeros(len(ids))
                    exact[ids.index(c["yi"])]=1.
                    for term,coef in zip(c["xi"],c["exact"]):
                        exact[ids.index(term)]=-coef
                    fixed.append(score(l,r,scale,exact))
    return dict(seed=seed,raw=meta["raw_hash"],scores=scores,fixed=fixed)


if __name__=="__main__":
    for d in ("records","training"):
        (DEST/d).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        all_rows=[f.result() for f in as_completed([pool.submit(case,s) for s in SEEDS])]
    all_rows.sort(key=lambda r:r["seed"])
    assert len({tuple(sorted(r["raw"].items())) for r in all_rows})==len(SEEDS)
    (DEST/"folds.json").write_text(json.dumps(all_rows,allow_nan=False))
    table=[]
    for key in all_rows[0]["scores"]:
        if key=="e0:reference":
            continue
        addition=":add_" in key
        differences=[]
        refit_exact=[]
        for row in all_rows:
            candidate=np.mean(row["scores"][key])
            reference=np.mean(row["scores"]["e0:reference"])
            differences.append((reference-candidate) if addition else (candidate-reference))
            refit_exact.append(np.mean(row["fixed"])-candidate)
        values=np.asarray(differences)
        table.append(dict(candidate=key,n=len(values),with_wins=int(sum(values>0)),without_wins=int(sum(values<0)),
                          median_margin=float(np.median(values)),candidate_beats_fixed_ref=int(sum(x>0 for x in refit_exact))))
    with (DEST/"outcomes.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,table[0].keys())
        w.writeheader()
        w.writerows(table)
    print(table)
