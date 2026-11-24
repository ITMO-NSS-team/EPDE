"""Independent full-panel replay on the first and last prospective fields."""
import argparse
import json
from ns_pair import ROOT,NEW,B,CHUNK,CASES,field_digest,np,records
import train
from evaluate import corrected_fit,score,matrices


def run(stage):
    dest=ROOT/"crosscheck"
    for directory in ("training","records"):
        (dest/directory).mkdir(exist_ok=True,parents=True)
    errors=[]
    for i in (0,B-1):
        seed=NEW+i
        name=f"ns_{seed}.npz"
        if stage=="prepare":
            records.ROOT=dest
            train.ROOT=dest
            records.build(("ns",seed))
            train.run(("ns",seed))
            continue
        rp,tp=dest/"records"/name,dest/"training"/name
        meta=json.loads(rp.with_suffix(".json").read_text())
        tm=json.loads(tp.with_suffix(".json").read_text())
        assert records.sha(rp)==meta["archive"] and records.sha(tp)==tm["sha256"]
        assert meta["raw_hash"]==tm["raw"]
        with np.load(ROOT/"draws"/f"fresh_{i-i%CHUNK:03}.npz") as z:
            assert z["field_hashes"][i%CHUNK]==field_digest(meta["raw_hash"])
            expected=z["scores"][i%CHUNK,:,:2]
        observed=[]
        with np.load(rp) as data,np.load(tp) as training:
            for g in range(3):
                matrix,gram=matrices(training,g,("local","time"))
                scores=[]
                for case in CASES:
                    c=next(c for c in meta["candidates"] if c["id"]==case)
                    ids=sorted([c["yi"],*c["xi"]])
                    v,_,status=corrected_fit(gram[np.ix_(ids,ids)],np.linalg.norm(matrix[:,ids],axis=0))
                    assert status=="ok"
                    left,right,scales=[data[f"{k}{g}"][...,ids] for k in ("left","right","scale")]
                    scores.append(score(left,right,scales,v))
                observed.append(scores)
        np.testing.assert_allclose(observed,expected,rtol=3.e-8,atol=5.e-13)
        errors.append(float(np.max(abs(expected-np.array(observed)))))
    if stage=="check":
        (dest/"checks.json").write_text(json.dumps(dict(seeds=[NEW,NEW+B-1],max_errors=errors,
            code_sha256=records.sha(ROOT/"full_panel_check.py")),indent=2))
        print(errors)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","check"])
    run(parser.parse_args().stage)
