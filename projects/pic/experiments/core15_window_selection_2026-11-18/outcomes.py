"""Compare full Pareto fronts and the paired one-term benchmark contrasts."""
from collections import Counter, defaultdict
import csv
import json
from setup_study import ROOT, SYSTEMS, SEEDS, WIDTHS, np, records
from score_panel import ARMS


def dump_csv(name,rows):
    with (ROOT/name).open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def totals(rows):
    keys=("registered","registered_possible","reference","reference_possible","extra_front","extra_possible","add_wins",
          "add_ties","add_unknown","n_add","true_kept","drop_ties","drop_unknown","n_drop","unknown")
    return dict(slots=len(rows),**{k:sum(r[k] for r in rows) for k in keys})


def main():
    slot_rows,contrasts,width_choices=[],[],[]
    hash_index={}
    errors=Counter()
    checks=Counter()
    inner_issues=Counter()
    field_hashes=defaultdict(set)
    changes=defaultdict(Counter)
    for s in SYSTEMS:
        for seed in SEEDS:
            stem=f"{s}_{seed}.json"
            path=ROOT/"results"/stem
            doc=json.loads(path.read_text())
            choice_path=ROOT/"decisions"/stem
            assert doc["choice_hash"]==records.sha(choice_path)
            decisions=json.loads(choice_path.read_text())["folds"]
            obs=json.loads((ROOT/"records"/stem).read_text())
            hash_index[stem]=records.sha(path)
            field_hashes[s].add(tuple(sorted(obs["raw_hash"].items())))
            checks.update(doc["checks"])
            for k,v in doc["max_error"].items():
                errors[k]=max(errors[k],v)
            for g,d in enumerate(decisions):
                inner_issues.update({key:d[key] for key in ("unavailable","all_unavailable")})
                counts=Counter(d["personal"])
                width_choices.append(dict(system=s,seed=seed,fold=g,shared=WIDTHS[d["shared"]],
                                          **{w:counts[i] for i,w in enumerate(WIDTHS)}))
            for slot in doc["slots"]:
                rows=[r for r in doc["rows"] if r["slot"]==slot["slot"]]
                ref=next(r for r in rows if r["label"]=="reference")
                contrast_cases=[r for r in rows if r["label"].startswith(("add_","drop_")) and not r["registered"]]
                marks={r["id"]:dict(system=s,seed=seed,slot=slot["slot"],candidate=r["id"],kind=r["label"].split("_")[0])
                       for r in contrast_cases}
                for arm in ARMS:
                    def heldout(row):
                        vals=[f[arm]["D"] for f in row["folds"]]
                        return None if None in vals else float(np.mean(vals))
                    ds={r["id"]:heldout(r) for r in rows}
                    y={r["id"]:r["dyad"] for r in rows}
                    base=ds[ref["id"]]
                    front=slot["arms"][arm]
                    if not front["unknown"]:
                        independent={i for i in ds if not any(ds[j]<=ds[i] and y[j]<=y[i] and
                                     (ds[j]<ds[i] or y[j]<y[i]) for j in ds)}
                        assert independent==set(front["definite"])==set(front["possible"])
                    count=Counter()
                    for r in contrast_cases:
                        key=r["id"]
                        add=r["label"].startswith("add_")
                        prefix="add" if add else "drop"
                        count[f"n_{prefix}"]+=1
                        if ds[key] is None or base is None:
                            count[f"{prefix}_unknown"]+=1
                            marks[key][arm]=None
                        else:
                            margin=ds[key]-base
                            marks[key][arm]=margin
                            if margin==0:
                                count[f"{prefix}_ties"]+=1
                            count["add_wins" if add else "true_kept"]+=int(margin<0 if add else margin>0)
                    entry=dict(system=s,seed=seed,slot=slot["slot"],arm=arm,
                        registered=front["registered"][0],registered_possible=front["registered"][1],
                        reference=front["reference"][0],reference_possible=front["reference"][1],
                        extra_front=front["add_count"][0],extra_possible=front["add_count"][1],
                        unknown=len(front["unknown"]),reference_D=base,
                        angle=float(np.mean([f[arm]["angle"] for f in ref["folds"]])) if base is not None else None)
                    entry.update({k:count[k] for k in ("add_wins","add_ties","add_unknown","n_add","true_kept",
                                                      "drop_ties","drop_unknown","n_drop")})
                    slot_rows.append(entry)
                contrasts.extend(marks.values())
                base_front=slot["arms"]["local"]
                for arm in ARMS:
                    for name in ("registered","reference"):
                        change=slot["arms"][arm][name][0]-base_front[name][0]
                        changes[arm][name+("_gain" if change>0 else "_loss" if change<0 else "_same")]+=1
    assert all(len(v)==6 for v in field_hashes.values()) and len(field_hashes)==15
    metrics={a:totals([r for r in slot_rows if r["arm"]==a]) for a in ARMS}
    for m in metrics.values():
        assert (m["slots"],m["n_add"],m["n_drop"])==(120,1128,306)
    systems=[dict(system=s,arm=a,**totals([r for r in slot_rows if (r["arm"],r["system"])==(a,s)]))
             for s in SYSTEMS for a in ARMS]
    choice_count=Counter(r["shared"] for r in width_choices)
    dump_csv("slots.csv",slot_rows)
    dump_csv("contrasts.csv",contrasts)
    dump_csv("by_system.csv",systems)
    dump_csv("width_choices.csv",width_choices)
    summary=dict(metrics=metrics,local_transitions=changes,shared_choices=choice_count,
                 checks=checks,numerical_errors=errors,inner_issues=inner_issues,results_sha256=hash_index,
                 protocol=records.sha(ROOT/"protocol.txt"))
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("results_sha256","protocol")},indent=2))


if __name__=="__main__":
    main()
