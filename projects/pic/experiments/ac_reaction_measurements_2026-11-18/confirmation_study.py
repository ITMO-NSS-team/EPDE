"""A matched reaction-only control with an exact closed-form observation fixture."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from unittest.mock import patch
import json
from common import E, ROOT, np, records
from compare import ARMS, ORACLES, evaluate
from measure import geometry, measure
from prepare import prepare

DEST=ROOT/"confirmation"
NEW=range(795001,795065)
METHODS=(*ARMS,"high_only","high_staged")


def fixture(kind):
    problem=E.R.datasets.load("ac")
    if kind=="positive":
        return problem
    t=problem.grids[0]
    initial=problem.data["u"][:1]
    denom=1+initial**2*np.expm1(10*t)
    field=initial*np.exp(5*t)/np.sqrt(denom)
    # Differentiate the manufactured expression independently, point by point.
    dt=5*initial*np.exp(5*t)*(1-initial**2)/denom**1.5
    np.testing.assert_array_equal(field[:1],initial)
    np.testing.assert_allclose(dt,5*field*(1-field**2),rtol=1e-11,atol=16*np.finfo(float).eps*5)
    return replace(problem,data={"u":field})


def batch(kind,start,stop):
    path=DEST/kind
    problem=fixture(kind)
    geometry.cache_clear()
    with patch.object(E.R.datasets,"load",side_effect=lambda name: {"ac":problem}[name]):
        for seed in range(start,stop):
            prepare(seed,path)
            measure(seed,path,extended=True)
            evaluate(seed,path,METHODS)
    geometry.cache_clear()
    return kind,start,stop


def collect():
    import csv
    from outcomes import summarize
    rows={}
    for kind in ("positive","zero"):
        summarize(DEST/kind,NEW,(*METHODS,*ORACLES),DEST/"protocol.txt",.0001 if kind=="positive" else 0.)
        with (DEST/kind/"fields.csv").open() as f:
            rows[kind]={(int(r["seed"]),r["arm"]):r for r in csv.DictReader(f)}
    out={}
    for a in (*METHODS,*ORACLES):
        pos,neg=[np.array([float(rows[k][n,a]["margin"]) for n in NEW]) for k in ("positive","zero")]
        better=pos[:,None]>neg
        ties=pos[:,None]==neg
        out[a]=dict(positive_wins=int(sum(pos>0)),false_wins=int(sum(neg>0)),
                    correct_null_front=0,paired_positive_margin_larger=int(sum(pos>neg)),
                    auc=float(np.mean(better+.5*ties)),
                    pos_median_margin=float(np.median(pos)),zero_median_margin=float(np.median(neg)))
    for n in NEW:
        doc=json.loads((DEST/"zero"/"results"/f"ac_{n}.json").read_text())
        for a in out:
            assert not doc["fronts"][a]["unknown"]
            out[a]["correct_null_front"]+=int("e0:drop_0" in doc["fronts"][a]["definite"])
    (DEST/"specificity.json").write_text(json.dumps(out,indent=2))
    print(out)


if __name__=="__main__":
    for kind in ("positive","zero"):
        for key in ("records","training","measurements","results"):
            (DEST/kind/key).mkdir(exist_ok=True,parents=True)
    for name in ("positive","zero"):
        p=fixture(name)
        info=dict(field=E.R.digest(p.data["u"]),initial=E.R.digest(p.data["u"][0]),shape=p.data["u"].shape,
                  noise_sigma=.05*float(np.std(p.data["u"])),protocol=records.sha(DEST/"protocol.txt"))
        (DEST/name/"fixture.json").write_text(json.dumps(info,indent=2))
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(batch,kind,n,n+4) for n in range(NEW.start,NEW.stop,4) for kind in ("positive","zero")]
        for f in as_completed(jobs):
            print(f.result(),flush=True)
    collect()
