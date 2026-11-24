"""Compute paired fixed-width confirmation on the two affected single trajectories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from setup_study import ROOT, records
from holdouts import build
from training import estimate
from select_width import freeze
from score_panel import ARMS, evaluate

FOLLOW = range(793001,793025)


def run(job):
    s,n=job
    build(job)
    estimate(s,n)
    freeze(s,n)
    evaluate(job)
    return job


def collect():
    from collections import Counter
    from outcomes import dump_csv
    from setup_study import np
    rows=[]
    for system in ("ac","kdv_cossin"):
        for seed in FOLLOW:
            doc=json.loads((ROOT/"results"/f"{system}_{seed}.json").read_text())
            assert len(doc["slots"])==1
            for arm in ARMS:
                scores={r["label"]:np.mean([f[arm]["D"] for f in r["folds"]]) for r in doc["rows"]}
                base=scores["reference"]
                front=doc["slots"][0]["arms"][arm]
                assert not front["unknown"]
                rows.append(dict(system=system,seed=seed,arm=arm,registered=front["registered"][0],
                    extra_front=front["add_count"][0],add_wins=sum(int(v<base) for k,v in scores.items() if k.startswith("add_")),
                    focal_term_kept=int(scores["drop_0"]>base),genuine_kept=sum(int(v>base) for k,v in scores.items() if k.startswith("drop_"))))
    dump_csv("followup_fields.csv",rows)
    out={}
    for system in ("ac","kdv_cossin"):
        by={}
        for arm in ARMS:
            sample=[r for r in rows if r["system"]==system and r["arm"]==arm]
            old=[r for r in rows if r["system"]==system and r["arm"]=="local"]
            assert [r["seed"] for r in sample]==[r["seed"] for r in old]
            change=Counter(a["registered"]-b["registered"] for a,b in zip(sample,old))
            by[arm]={k:sum(r[k] for r in sample) for k in
                     ("registered","extra_front","add_wins","focal_term_kept","genuine_kept")}
            by[arm].update(paired_gains=change[1],paired_losses=change[-1],paired_ties=change[0])
        out[system]=by
    (ROOT/"followup_summary.json").write_text(json.dumps(dict(protocol=records.sha(ROOT/"followup_protocol.txt"),systems=out),indent=2))
    print(json.dumps({s:{k:v for k,v in value.items() if k in ("local","middle")} for s,value in out.items()},indent=2))


if __name__=="__main__":
    jobs=[(s,n) for n in FOLLOW for s in ("ac","kdv_cossin")]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            print(f.result(),flush=True)
    collect()
