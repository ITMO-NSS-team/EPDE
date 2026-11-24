"""Slot-level Pareto outcomes and paired changes on the complete candidate panels."""
from collections import Counter
import csv
import json
from panel import REPLAY, FRESH, ROOT, SYSTEMS, records
from evaluate import ARMS, np


def category(row,kind):
    if kind=="registered":
        return row["registered"]
    if kind=="reference":
        return row["label"]=="reference"
    return row["label"].startswith("add_") and not row["registered"]


def front(rows,arm):
    d = {}
    y = {}
    for r in rows:
        i = r["id"]
        v = [f[arm]["D"] for f in r["folds"]]
        d[i] = None if None in v else float(np.mean(v))
        y[i] = r["legacy_dyad" if arm=="legacy" else "dyad"]
    known = [i for i in d if d[i] is not None]
    unknown = set(d)-set(known)
    dominators = {i:[k for k in known if d[k]<=d[i] and y[k]<=y[i] and (d[k]<d[i] or y[k]<y[i])]
                  for i in known}
    known_front = {i for i in known if not dominators[i]}
    possible = known_front | unknown
    sure = {i for i in known_front if all(y[u]>y[i] for u in unknown)}
    sure.update(i for i in unknown if all(y[k]>y[i] for k in d if k!=i))
    result = {"definite":sorted(sure),"possible":sorted(possible),"unknown":sorted(unknown),"front":[len(sure),len(possible)]}
    for kind in ("registered","reference","add"):
        ids = {r["id"] for r in rows if category(r,kind)}
        lower = bool(sure & ids) or any(all(u in ids or y[u]>y[i] for u in unknown) for i in known_front & ids)
        result[kind] = [int(lower),int(bool(possible & ids))]
        if kind=="add":
            result["add_count"] = [len(sure & ids),len(possible & ids)]
            result["add_trials"] = len(ids)
        if kind=="reference":
            result["reference_dominators"] = dominators.get(next(iter(ids)),[])
    return result


def metrics(slots,arm):
    res = dict(slots=len(slots))
    for kind in ("registered","reference","add","add_count","front"):
        res[kind] = np.sum([s["arms"][arm][kind] for s in slots],axis=0).tolist()
    res["add_trials"] = sum(s["arms"][arm]["add_trials"] for s in slots)
    res["unresolved"] = sum(len(s["arms"][arm]["unknown"]) for s in slots)
    return res


def main():
    slots,angles,manifests = [],[],{}
    audits = Counter()
    invalid = Counter()
    for seed in (*REPLAY,*FRESH):
        cohort = "replay" if seed in REPLAY else "fresh"
        for system in SYSTEMS:
            path = ROOT/"results"/f"{system}_{seed}.json"
            out = json.loads(path.read_text())
            manifests[path.name] = records.sha(path)
            assert (out["system"],out["seed"])==(system,seed)
            for key,val in out["audit"].items():
                if key=="checks":
                    audits[key]+=val
                else:
                    audits[key]=max(audits[key],val)
            for s in sorted({r["slot"] for r in out["rows"]}):
                subset = [r for r in out["rows"] if r["slot"]==s]
                slots.append(dict(system=system,seed=seed,cohort=cohort,slot=s,
                                  arms={a:front(subset,a) for a in ARMS}))
                for r in subset:
                    for g,f in enumerate(r["folds"]):
                        for a in ARMS:
                            if f[a]["status"]!="ok":
                                invalid[f'{cohort}/{a}/{f[a]["status"]}']+=1
                        if r["label"]=="reference":
                            angles.append(dict(system=system,cohort=cohort,seed=seed,slot=s,fold=g,
                                **{a:(None if f[a]["angle"] is None else np.degrees(f[a]["angle"])) for a in ARMS}))
    def write_csv(name,rows):
        with (ROOT/name).open("w",newline="") as f:
            writer = csv.DictWriter(f,rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_csv("reference_angles.csv",angles)
    cohorts,by_system,transitions = {},[],[]
    for cohort in ("replay","fresh"):
        sample = [s for s in slots if s["cohort"]==cohort]
        assert len(sample)==(60 if cohort=="replay" else 120)
        cohorts[cohort] = {a:metrics(sample,a) for a in ARMS}
        for a in ARMS:
            deg = [row[a] for row in angles if row["cohort"]==cohort and row[a] is not None]
            cohorts[cohort][a].update(angle_folds=len(deg),median_reference_angle=float(np.median(deg)))
        for system in SYSTEMS:
            for a in ARMS:
                m = metrics([s for s in sample if s["system"]==system],a)
                row = dict(cohort=cohort,system=system,arm=a,slots=m["slots"],add_trials=m["add_trials"])
                for kind in ("registered","reference","add","add_count","front"):
                    row[kind+"_lower"],row[kind+"_upper"] = m[kind]
                by_system.append(row)
        for bank in ("local_time","broad"):
            for s in sample:
                a,b = [s["arms"][f"{bank}_{corr}"] for corr in ("off","on")]
                for outcome in ("registered","reference","add","add_count"):
                    if a[outcome]!=b[outcome]:
                        transitions.append(dict(cohort=cohort,bank=bank,system=s["system"],seed=s["seed"],
                            slot=s["slot"],outcome=outcome,off=a[outcome],on=b[outcome],
                            removed=sorted(set(a["definite"])-set(b["possible"])),
                            gained=sorted(set(b["definite"])-set(a["possible"]))))
    write_csv("by_system.csv",by_system)
    for name,value in (("slots",slots),("transitions",transitions),("summary",dict(cohorts=cohorts,
        audits=dict(audits),invalid_folds=dict(invalid),results_sha256=manifests))):
        (ROOT/(name+".json")).write_text(json.dumps(value,indent=2,allow_nan=False))
    for cohort,arms in cohorts.items():
        print(cohort)
        for a,m in arms.items():
            print(a,{k:v for k,v in m.items() if k!="angle_folds"})
    print(audits,invalid)


if __name__=="__main__":
    main()
