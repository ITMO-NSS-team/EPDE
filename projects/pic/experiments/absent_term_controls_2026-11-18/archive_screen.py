"""Paired support comparisons for the previously observed default systems."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PANEL=ROOT.parent/"core15_implicit_correction_2026-11-18"
SEEDS=tuple(range(73001,73007))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,rows):
    with (ROOT/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive=json.loads((PANEL/"summary.json").read_text())["results_sha256"]
    pairs=[]
    checks=0
    worst=0.
    for seed in SEEDS:
        files=sorted((PANEL/"results").glob(f"*_{seed}.json"))
        assert len(files)==15
        for path in files:
            assert sha(path)==archive[path.name]
            doc=json.loads(path.read_text())
            inputs=PANEL/"records"/path.with_suffix(".npz").name
            observed=json.loads(inputs.with_suffix(".json").read_text())
            assert sha(inputs)==doc["record"]==observed["archive"]
            refs={r["slot"]:r for r in doc["rows"] if r["label"]=="reference"}
            selected=[r for r in doc["rows"] if r["label"]=="reference" or r["label"].startswith(("add_","drop_"))]
            values={}
            with np.load(inputs) as data:
                for r in selected:
                    ids=r["ids"]
                    scores=[]
                    for g,fold in enumerate(r["folds"]):
                        f=fold["local_time_on"]
                        assert f["status"]=="ok"
                        v=np.array(f["vector"])
                        left,right,sc=[data[f"{k}{g}"][...,ids] for k in ("left","right","scale")]
                        actual=(v@(left.T@right)@v)/sum((sc*v)**2)
                        err=abs(actual-f["D"])/max(1.,abs(f["D"]))
                        worst=max(worst,err)
                        checks+=1
                        scores.append(f["D"])
                    values[r["id"]]=float(np.mean(scores))
            for r in selected:
                if r["registered"] or r["label"]=="reference":
                    continue
                ref=refs[r["slot"]]
                absent=r["label"].startswith("add_")
                with_term,without=(r,ref) if absent else (ref,r)
                ix,=set(with_term["ids"])-set(without["ids"])
                assert set(without["ids"])<set(with_term["ids"])
                d0,d1=values[without["id"]],values[with_term["id"]]
                margin=d0-d1
                dyad=without["dyad"]-with_term["dyad"]
                either_strict=margin!=0 or dyad!=0
                result=("with" if margin>=0 and dyad>=0 and either_strict else
                        "without" if margin<=0 and dyad<=0 and either_strict else "tradeoff_or_tie")
                pairs.append(dict(system=doc["system"],slot=r["slot"],candidate=r["id"],seed=seed,
                    group="addition" if absent else "deletion",term=json.dumps(observed["labels"][ix]),
                    D_without=d0,D_with=d1,margin=margin,dyad_margin=dyad,pairwise_pareto=result))
    assert worst<1e-7
    groups=defaultdict(list)
    for row in pairs:
        groups[row["system"],row["slot"],row["candidate"]].append(row)
    summaries=[]
    for (system,slot,c),rows in groups.items():
        assert len(rows)==len(SEEDS)
        s=np.array([r["margin"] for r in rows])
        safe=np.array([abs(r["margin"])>1e-12*max(1.,abs(r["D_without"]),abs(r["D_with"])) for r in rows])
        summaries.append(dict(system=system,slot=slot,candidate=c,group=rows[0]["group"],term=rows[0]["term"],
            favor_with=int(sum(s>0)),favor_without=int(sum(s<0)),ties=int(sum(s==0)),median_margin=float(np.median(s)),
            all_with=bool(all(s>0)),all_without=bool(all(s<0)),all_numerically_separated=bool(all(safe)),
            pareto_with=sum(r["pairwise_pareto"]=="with" for r in rows),
            pareto_without=sum(r["pairwise_pareto"]=="without" for r in rows)))
    write("archive_margins.csv",pairs)
    write("archive_pairs.csv",summaries)
    result={}
    for kind in ("addition","deletion"):
        selected=[r for r in summaries if r["group"]==kind]
        result[kind]=dict(comparisons=len(selected),observations=len(selected)*len(SEEDS),
            all_with=sum(r["all_with"] for r in selected),all_without=sum(r["all_without"] for r in selected),
            mixed_or_tied=sum(not (r["all_with"] or r["all_without"]) for r in selected),
            all_with_separated=sum(r["all_with"] and r["all_numerically_separated"] for r in selected),
            all_without_separated=sum(r["all_without"] and r["all_numerically_separated"] for r in selected),
            wins_with=sum(r["favor_with"] for r in selected),wins_without=sum(r["favor_without"] for r in selected),
            ties=sum(r["ties"] for r in selected),pareto_with=sum(r["pareto_with"] for r in selected),
            pareto_without=sum(r["pareto_without"] for r in selected))
    assert result["addition"]["observations"]==1128
    (ROOT/"archive_summary.json").write_text(json.dumps(dict(outcomes=result,checks=checks,maximum_replay_error=worst,
        old_summary=sha(PANEL/"summary.json"),pairs=sha(ROOT/"archive_pairs.csv")),indent=2))
    print(result,checks,worst)


if __name__=="__main__":
    main()
