"""Replay two prospective Allen–Cahn fields using the archived full-panel builders."""
import json
from controls import ROOT,NEW,CHUNK,B,records,np,corrected_fit,score
from evaluate import matrices
import train


def main():
    dest=ROOT/"crosscheck"
    for d in ("records","training"):
        (dest/d).mkdir(parents=True,exist_ok=True)
    records.ROOT=dest
    train.ROOT=dest
    names=json.loads((ROOT/"anchors/73001.json").read_text())["cases"]
    errors=[]
    for i in (0,B-1):
        seed=NEW+i
        records.build(("ac",seed))
        train.run(("ac",seed))
        name=f"ac_{seed}.npz"
        rp,tp=dest/"records"/name,dest/"training"/name
        meta=json.loads(rp.with_suffix(".json").read_text())
        tm=json.loads(tp.with_suffix(".json").read_text())
        assert meta["raw_hash"]==tm["raw"] and records.sha(rp)==meta["archive"] and records.sha(tp)==tm["sha256"]
        with np.load(ROOT/"draws"/f"fresh_{i-i%CHUNK:03}.npz") as z:
            assert meta["raw_hash"]["u"]==z["raw_hash"][i%CHUNK]
            expected=z["scores"][i%CHUNK]
        original=[]
        cs={c["id"]:c for c in meta["candidates"]}
        with np.load(rp) as data,np.load(tp) as tr:
            for g in range(3):
                matrix,gram=matrices(tr,g,("local","time"))
                vals=[]
                for key in names:
                    c=cs[key]
                    ids=sorted([c["yi"],*c["xi"]])
                    v,_,status=corrected_fit(gram[np.ix_(ids,ids)],np.linalg.norm(matrix[:,ids],axis=0))
                    assert status=="ok"
                    left,right,sc=[data[f"{k}{g}"][...,ids] for k in ("left","right","scale")]
                    vals.append(score(left,right,sc,v))
                original.append(vals)
        np.testing.assert_allclose(original,expected,rtol=3e-8,atol=5e-12)
        errors.append(float(np.max(abs(expected-original))))
    (dest/"replay.json").write_text(json.dumps(dict(seeds=[NEW,NEW+B-1],errors=errors,fold_scores=60)))
    print(errors)


if __name__=="__main__":
    main()
