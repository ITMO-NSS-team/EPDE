"""Prospective evaluation of a location-only forecast correction."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
from ns_pair import ROOT,ANCHORS,B,CHUNK,geometry,field_digest,hashes,records,np
from summarize_pair import verify,stats,write_csv

DEST=ROOT/"confirmation"
START=770001


def freeze():
    with (ROOT/"forecast_check.csv").open() as f:
        forecasts=list(csv.DictReader(f))
    rows=[]
    for r in forecasts:
        observed,median,lo,hi=[float(r[k]) for k in ("observed","bootstrap_median","lower","upper")]
        shift=observed-median
        rows.append(dict(anchor=int(r["anchor"]),original_lower=lo,original_upper=hi,
                         shifted_lower=lo+shift,shifted_upper=hi+shift,shift=shift))
    dest=DEST/"frozen.json"
    encoded=json.dumps(dict(ranges=rows,input_sha256=records.sha(ROOT/"forecast_check.csv"),
        protocol_sha256=records.sha(ROOT/"recenter_protocol.txt")),indent=2)
    if dest.exists():
        assert dest.read_text()==encoded
    else:
        dest.write_text(encoded)


def chunk(start):
    geo=geometry()
    freeze_hash=records.sha(DEST/"frozen.json")
    results=[]
    digests=[]
    seeds=np.arange(START+start,START+start+CHUNK)
    for n in seeds:
        raw=geo.p.noisy(5,int(n))
        results.append(geo.estimate(raw,int(n)))
        digests.append(field_digest(hashes(raw)))
    path=DEST/f"fresh_{start:03}.npz"
    with path.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack([r[k] for r in results]) for k in results[0]},seeds=seeds,
                            field_hashes=digests,freeze_hash=freeze_hash)
    path.with_suffix(".pending").replace(path)
    return start


def analyze():
    frozen_path=DEST/"frozen.json"
    frozen=json.loads(frozen_path.read_text())
    assert frozen["input_sha256"]==records.sha(ROOT/"forecast_check.csv")
    assert frozen["protocol_sha256"]==records.sha(ROOT/"recenter_protocol.txt")
    freeze_hash=records.sha(frozen_path)
    archives={}
    chunks=[]
    for i in range(0,B,CHUNK):
        path=DEST/f"fresh_{i:03}.npz"
        archives[path.name]=records.sha(path)
        with np.load(path) as z:
            assert str(z["freeze_hash"])==freeze_hash
            chunks.append({k:z[k] for k in z.files if k!="freeze_hash"})
    data={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    np.testing.assert_array_equal(data["seeds"],np.arange(START,START+B))
    earlier=set()
    for path in (ROOT/"draws").glob("*.npz"):
        with np.load(path) as z:
            earlier.update(z["field_hashes"])
    assert len(set(data["field_hashes"]))==128 and set(data["field_hashes"]).isdisjoint(earlier)
    meta=json.loads((ROOT/"replay"/f"{ANCHORS[0]}.json").read_text())
    score_error,angle_error=verify(data,meta)
    scores=data["scores"].mean(axis=1)
    delta=scores[:,1]-scores[:,0]
    rows=[]
    for r in frozen["ranges"]:
        row=dict(anchor=r["anchor"])
        for tag in ("original","shifted"):
            lo,hi=r[f"{tag}_lower"],r[f"{tag}_upper"]
            row.update({f"{tag}_lower":lo,f"{tag}_upper":hi,f"{tag}_covered":int(sum((lo<=delta)&(delta<=hi))),
                        f"{tag}_decision":"retain" if lo>0 else "delete" if hi<0 else "unresolved"})
        rows.append(row)
    write_csv(DEST/"per_anchor.csv",rows)
    write_csv(DEST/"margins.csv",[dict(seed=int(s),refitted=float(x)) for s,x in zip(data["seeds"],delta)])
    out=dict(fresh=stats.describe(delta),ranges=rows,freeze_hash=freeze_hash,archives=archives,
             independent_fold_scores=B*3*4,score_error=score_error,angle_error=angle_error,
             code=records.sha(ROOT/"recenter.py"))
    (DEST/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print(out["fresh"])
    for r in rows:
        print(r)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["run","report"])
    args=ap.parse_args()
    if args.stage=="report":
        analyze()
    else:
        DEST.mkdir(exist_ok=True)
        freeze()
        jobs=[i for i in range(0,B,CHUNK) if not (DEST/f"fresh_{i:03}.npz").exists()]
        with ProcessPoolExecutor(max_workers=8) as pool:
            for n,f in enumerate(as_completed([pool.submit(chunk,i) for i in jobs]),1):
                f.result()
                if n%8==0 or n==len(jobs):
                    print(n,len(jobs),flush=True)
