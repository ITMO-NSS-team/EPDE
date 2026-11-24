"""Report paired field outcomes and accuracy for reference coefficients."""
from collections import Counter
import csv
import json
from scipy.stats import binomtest
from common import ROOT, SEEDS, np, records
from compare import ARMS, ORACLES


def dump_csv(path,rows):
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def summarize(directory,seeds,arms,protocol=None,true_diffusion=.0001):
    fields=[]
    term_rows=[]
    all_hashes=set()
    max_error=0.
    for n in seeds:
        result=json.loads((directory/"results"/f"ac_{n}.json").read_text())
        assert result["raw_hash"] not in all_hashes
        all_hashes.add(result["raw_hash"])
        max_error=max(max_error,result["unit_error"])
        for a in arms:
            f=result["fronts"][a]
            assert not f["unknown"], (n,a,f["unknown"])
            held={r["label"]:np.mean([part[a]["D"] for part in r["folds"]]) for r in result["rows"]}
            base=held["reference"]
            added=[k for k in held if k.startswith("add_")]
            drops=[k for k in held if k.startswith("drop_")]
            assert len(added)==6 and len(drops)==3
            for k in added+drops:
                term_rows.append(dict(seed=n,arm=a,rival=k,rival_minus_reference=held[k]-base))
            ref=next(r for r in result["rows"] if r["label"]=="reference")
            coefficients={k:np.median([fold[a]["coeff"][k] for fold in ref["folds"]]) for k in ("u","cube","laplacian")}
            fields.append(dict(seed=n,arm=a,front=f["reference"][0],diffusion=int(held["drop_0"]>base),
                linear=int(held["drop_2"]>base),cubic=int(held["drop_1"]>base),
                add_wins=sum(int(held[k]<base) for k in added),add_front=f["add_count"][0],D=base,
                margin=held["drop_0"]-base,**coefficients))
    dump_csv(directory/"fields.csv",fields)
    dump_csv(directory/"terms.csv",term_rows)
    metrics={}
    for a in arms:
        rows=[r for r in fields if r["arm"]==a]
        old=[r for r in fields if r["arm"]=="local"]
        assert [r["seed"] for r in rows]==[r["seed"] for r in old]
        totals={k:sum(r[k] for r in rows) for k in ("front","diffusion","linear","cubic","add_wins","add_front")}
        totals["paired"]={}
        for k in ("front","diffusion"):
            change=Counter(r[k]-b[k] for r,b in zip(rows,old))
            wins,losses=change[1],change[-1]
            totals["paired"][k]=dict(gains=wins,losses=losses,ties=change[0],
                two_sided_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.)
        totals["median"]={k:float(np.median([r[k] for r in rows])) for k in ("D","margin","u","cube","laplacian")}
        totals["median_abs_error"]={k:float(np.median([abs(r[k]-v) for r in rows])) for k,v in
                                    (("u",5),("cube",-5),("laplacian",true_diffusion))}
        metrics[a]=totals
    out=dict(fields=len(all_hashes),metrics=metrics,max_unit_error=max_error,protocol=records.sha(protocol or directory/"protocol.txt"))
    (directory/"summary.json").write_text(json.dumps(out,indent=2))
    for a,v in metrics.items():
        print(a,[v[k] for k in ("front","diffusion","add_wins","add_front")],v["median_abs_error"])
    return out


if __name__=="__main__":
    summarize(ROOT,SEEDS,(*ARMS,*ORACLES))
