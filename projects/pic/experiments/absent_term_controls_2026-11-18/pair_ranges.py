"""Freeze one-field decisions and measure their prospective coverage."""
import argparse
import csv
import json
from pathlib import Path
from controls import ANCHORS,B,CHUNK,NEW,ROOT,np,records
from scipy.linalg import eigh
from scipy.stats import binomtest

NAMES={"drop_0":"u_xx","drop_1":"u³","drop_2":"u","add_0":"1","add_1":"t","add_2":"t²",
       "add_3":"sin(2t)","add_4":"cos(2t)","add_6":"u²"}


def load_group(tag):
    parts=[]
    manifest={}
    for i in range(0,B,CHUNK):
        path=ROOT/"draws"/f"{tag}_{i:03}.npz"
        manifest[path.name]=records.sha(path)
        with np.load(path) as z:
            if tag=="fresh":
                assert str(z["freeze_hash"])==records.sha(ROOT/"ranges.json")
            else:
                assert z["replay_error"]<5.e-12
            parts.append({k:z[k] for k in ("seeds","scores","vec","cross","scale","gram","norm","raw_hash")})
    group={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(set(group["raw_hash"]))==B
    first=NEW if tag=="fresh" else 800000+(int(tag)-73001)*1000
    np.testing.assert_array_equal(group["seeds"],np.arange(first,first+B))
    return group,manifest


def audit(data,supports):
    v=data["vec"]
    denom=np.sum((v*data["scale"][:,:,None,:])**2,axis=-1)
    independent=np.einsum("...ki,...ij,...kj->...k",v,data["cross"],v)/denom
    error=float(np.max(abs(independent-data["scores"])))
    np.testing.assert_allclose(independent,data["scores"],rtol=3e-8,atol=4e-12)
    angle=0.
    for j,ids in enumerate(supports):
        outside=np.setdiff1d(np.arange(v.shape[-1]),ids)
        assert np.all(v[:,:,j,outside]==0.)
        for i in range(len(v)):
            for g in range(3):
                scale=data["norm"][i,g,ids]
                gram=data["gram"][i,g][np.ix_(ids,ids)]/scale[:,None]/scale[None,:]
                _,basis=eigh((gram+gram.T)/2,driver="evx")
                est=v[i,g,j,ids]*scale
                est/=np.linalg.norm(est)
                dist=min(np.linalg.norm(est-basis[:,0]),np.linalg.norm(est+basis[:,0]))
                angle=max(angle,float(2*np.arcsin(dist/2)))
    assert angle<1.e-6
    return dict(verified_scores=int(independent.size),maximum_absolute_score_error=error,max_angle=angle)


def margins(scores,names):
    # Both inputs (single observation or repetition ensemble) carry the fold axis second to last.
    means=scores.mean(axis=-2)
    reference=means[...,0]
    return {k:((reference-means[...,j]) if k.startswith("add_") else (means[...,j]-reference))
            for j,k in enumerate(names) if j}


def decision(lo,hi):
    return "retain" if lo>0 else "drop" if hi<0 else "unresolved"


def freeze():
    ranges=[]
    checks={}
    sources={}
    seen=set()
    for a in ANCHORS:
        p=ROOT/"anchors"/f"{a}.npz"
        meta=json.loads(p.with_suffix(".json").read_text())
        assert meta["archive"]==records.sha(p)
        names=[c.split(":")[1] for c in meta["cases"]]
        with np.load(p) as z:
            observed=margins(z["scores"],names)
        boot,manifest=load_group(str(a))
        assert not seen.intersection(boot["raw_hash"])
        seen.update(boot["raw_hash"])
        sources.update(manifest)
        checks[str(a)]=audit(boot,meta["supports"])
        for k,x in margins(boot["scores"],names).items():
            lo,hi=np.sort(x)[[2,125]]
            median=float(np.median(x))
            shift=observed[k]-median
            ranges.append(dict(anchor=a,key=k,term=NAMES[k],observed=observed[k],median=median,
                original_lower=lo,original_upper=hi,shifted_lower=lo+shift,shifted_upper=hi+shift,
                positive=int(sum(x>0)),original_decision=decision(lo,hi),shifted_decision=decision(lo+shift,hi+shift)))
    assert len(seen)==6*B
    dest=ROOT/"ranges.json"
    options=dict(ranges=ranges,checks=checks,bootstrap_sha256=sources,protocol=records.sha(ROOT/"protocol.txt"))
    encoded=json.dumps(options,indent=2,allow_nan=False)
    if dest.exists():
        assert dest.read_text()==encoded
    else:
        dest.write_text(encoded)
    print("Frozen ranges:",len(ranges))


def write(name,rows):
    with (ROOT/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def analyze():
    data,manifest=load_group("fresh")
    frozen=json.loads((ROOT/"ranges.json").read_text())
    assert frozen["protocol"]==records.sha(ROOT/"protocol.txt")
    for p in (ROOT/"draws").glob("73*.npz"):
        assert records.sha(p)==frozen["bootstrap_sha256"][p.name]
        with np.load(p) as z:
            assert set(z["raw_hash"]).isdisjoint(data["raw_hash"])
    meta=json.loads((ROOT/"anchors"/f"{ANCHORS[0]}.json").read_text())
    names=[k.split(":")[1] for k in meta["cases"]]
    checks=audit(data,meta["supports"])
    gaps=margins(data["scores"],names)
    rows=[]
    for row in frozen["ranges"]:
        x=gaps[row["key"]]
        f=dict(row)
        for arm in ("original","shifted"):
            f[f"{arm}_covered"]=int(sum((f[f"{arm}_lower"]<=x)&(x<=f[f"{arm}_upper"])))
        rows.append(f)
    table=[]
    for key,x in gaps.items():
        cohort=[r for r in rows if r["key"]==key]
        n_pos=int(sum(x>0))
        ci=binomtest(n_pos,B).proportion_ci()
        entry=dict(key=key,term=NAMES[key],kind="absent" if key.startswith("add_") else "present",
            wins_retain=n_pos,wins_drop=int(sum(x<0)),ties=int(sum(x==0)),total=B,median=float(np.median(x)),
            lower=float(np.sort(x)[2]),upper=float(np.sort(x)[125]),p_retain_ci=[ci.low,ci.high])
        for arm in ("original","shifted"):
            for d in ("retain","drop","unresolved"):
                entry[arm+"_"+d]=sum(r[arm+"_decision"]==d for r in cohort)
            cov=[r[arm+"_covered"] for r in cohort]
            entry[arm+"_coverage_min"]=min(cov)
            entry[arm+"_coverage_max"]=max(cov)
        table.append(entry)
    write("term_outcomes.csv",table)
    write("per_anchor.csv",rows)
    write("fresh_margins.csv",[dict(seed=int(s),**{k:x[i] for k,x in gaps.items()}) for i,s in enumerate(data["seeds"])])
    summary=dict(outcomes=table,checks=checks,anchor_count=len(ANCHORS),freeze=records.sha(ROOT/"ranges.json"),
        fresh_sha256=manifest,analysis=records.sha(Path(__file__)),runner=records.sha(ROOT/"controls.py"))
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    for e in table:
        print(e["term"],e["wins_retain"],e["wins_drop"],[e[k] for k in ("shifted_retain","shifted_drop","shifted_unresolved")],
              [e[k] for k in ("shifted_coverage_min","shifted_coverage_max")])
    print("checks",checks)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["freeze","report"])
    if p.parse_args().stage=="freeze":
        freeze()
    else:
        analyze()
