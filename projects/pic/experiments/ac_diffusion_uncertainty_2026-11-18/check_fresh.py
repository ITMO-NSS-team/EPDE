"""Replay two new fields with the original full-panel code path."""
import argparse
import json
from paired import ROOT,NEW,B,CHUNK,np,records
import train
from evaluate import corrected_fit,score,matrices


def prepare(seed,dest):
    # These modules' original functions use ROOT solely as the result destination.
    records.ROOT=dest
    train.ROOT=dest
    records.build(("ac",seed))
    train.run(("ac",seed))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["prepare","check"])
    stage=ap.parse_args().stage
    dest=ROOT/"crosscheck"
    for d in ("records","training"):
        (dest/d).mkdir(parents=True,exist_ok=True)
    errors=[]
    for i in (0,B-1):
        seed=NEW+i
        if stage=="prepare":
            prepare(seed,dest)
            continue
        path=dest/"records"/f"ac_{seed}.npz"
        meta=json.loads(path.with_suffix(".json").read_text())
        trainpath=dest/"training"/path.name
        tm=json.loads(trainpath.with_suffix(".json").read_text())
        assert records.sha(path)==meta["archive"] and records.sha(trainpath)==tm["sha256"]
        with np.load(ROOT/"draws"/f"fresh_{i-i%CHUNK:03}.npz") as z:
            archived=z["scores"][i%CHUNK]
            assert z["field_hashes"][i%CHUNK]==meta["raw_hash"]["u"]
        observed=[]
        with np.load(path) as data,np.load(trainpath) as tr:
            for g in range(3):
                point,gram=matrices(tr,g,("local","time"))
                values=[]
                for key in ("e0:reference","e0:drop_0"):
                    c=next(c for c in meta["candidates"] if c["id"]==key)
                    ids=sorted([c["yi"],*c["xi"]])
                    v,_,status=corrected_fit(gram[np.ix_(ids,ids)],np.linalg.norm(point[:,ids],axis=0))
                    assert status=="ok"
                    l,r,s=[data[f"{side}{g}"][...,ids] for side in ("left","right","scale")]
                    values.append(score(l,r,s,v))
                observed.append(values)
        np.testing.assert_allclose(observed,archived[:,:2],rtol=3e-8,atol=5e-13)
        errors.append(float(np.max(abs(observed-archived[:,:2]))))
    if stage=="check":
        (dest/"checks.json").write_text(json.dumps(dict(seeds=[NEW,NEW+B-1],max_per_seed=errors,
            sha256=records.sha(ROOT/"check_fresh.py")),indent=2))
        print(errors)


if __name__=="__main__":
    main()
