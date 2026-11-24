"""Locate which systems and candidate comparisons drive the pooled endpoints."""
from collections import Counter, defaultdict
import csv
import json
from setup_study import ROOT, SEEDS, SYSTEMS, WIDTHS, np
from score_panel import ARMS


if __name__=="__main__":
    with (ROOT/"slots.csv").open() as f:
        fields=list(csv.DictReader(f))
    with (ROOT/"contrasts.csv").open() as f:
        pairs=list(csv.DictReader(f))
    with (ROOT/"by_system.csv").open() as f:
        per_system=list(csv.DictReader(f))
    lookup={(r["system"],r["seed"],r["slot"],r["arm"]):r for r in fields}
    ref={}
    for arm in ARMS:
        changes=Counter()
        trans=defaultdict(lambda:Counter())
        for r in fields:
            if r["arm"]!=arm:
                continue
            old=lookup[r["system"],r["seed"],r["slot"],"local"]
            for key in ("angle","reference_D"):
                if r[key] and old[key]:
                    d=float(r[key])-float(old[key])
                    changes[key+("_lower" if d<0 else "_higher" if d>0 else "_tie")]+=1
            reg=int(r["registered"])-int(old["registered"])
            if reg:
                trans[r["system"]]["gain" if reg>0 else "loss"]+=1
        ref[arm]=dict(changes=changes,front_transitions=trans)
    issues={}
    for arm in ARMS:
        cc=defaultdict(list)
        for p in pairs:
            if p["kind"]=="drop":
                cc[p["system"]+"/"+p["candidate"]].append(float(p[arm]) if p[arm] else np.nan)
        issues[arm]={k:int(sum(x>0 for x in v)) for k,v in cc.items() if not all(x>0 for x in v)}
    best=Counter()
    for r in fields:
        if r["arm"]=="local":
            fixed=[lookup[r["system"],r["seed"],r["slot"],w] for w in WIDTHS]
            best["registered_on_any_fixed_width"]+=int(any(int(x["registered"]) for x in fixed))
            best["registered_on_all_fixed_widths"]+=int(all(int(x["registered"]) for x in fixed))
    matches=defaultdict(Counter)
    for s in SYSTEMS:
        for n in SEEDS:
            stem=f"{s}_{n}.json"
            d=json.loads((ROOT/"decisions"/stem).read_text())["folds"]
            doc=json.loads((ROOT/"results"/stem).read_text())
            for row in doc["rows"]:
                if row["label"]!="reference":
                    continue
                for fold in row["folds"]:
                    vals=[fold[w]["D"] for w in WIDTHS]
                    if None in vals:
                        continue
                    tol=1e-12*max(1.,max(abs(x) for x in vals))
                    for arm in (*WIDTHS,"shared","individual"):
                        matches[arm]["folds"]+=1
                        matches[arm]["tied_for_lowest_heldout"]+=int(fold[arm]["D"]<=min(vals)+tol)
    arms_by_system={}
    for s in SYSTEMS:
        arms_by_system[s]={r["arm"]:[int(r[k]) for k in ("registered","add_wins","true_kept","extra_front")]
                           for r in per_system if r["system"]==s}
    doc=dict(reference=ref,genuine_retention_below_six=issues,fixed_width_bounds=best,
             reference_width_matches=matches,systems=arms_by_system)
    (ROOT/"diagnostics.json").write_text(json.dumps(doc,indent=2))
    print(json.dumps({k:doc[k] for k in ("genuine_retention_below_six","fixed_width_bounds","reference_width_matches")},indent=2))
