"""Navier–Stokes positive control using the existing whole-field refit engine."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
AC=ROOT.parent/"ac_diffusion_uncertainty_2026-11-18"
sys.path.insert(0,str(AC))
import paired as previous
from paired import ANCHORS,B,E,np,records,PRIOR

NEW=760001
CHUNK=2
CASES=("e0:reference","e0:drop_3")


class Geometry(previous.Geometry):
    def __init__(self):
        self.p=E.R.datasets.load("ns")
        self.axes=E.R.axes_of(self.p)
        cs,labels=E.R.panel(self.p)
        full,drop=[next(c for c in cs if c["id"]==key) for key in CASES]
        self.ids=sorted([full["yi"],*full["xi"]])
        self.target=self.ids.index(full["yi"])
        miss,=set(full["xi"])-set(drop["xi"])
        assert full["exact"][full["xi"].index(miss)]==0.01
        self.missing=self.ids.index(miss)
        self.small=np.delete(np.arange(len(self.ids)),self.missing)
        self.truth=np.zeros(len(self.ids))
        self.truth[self.target]=1.
        for i,k in zip(full["xi"],full["exact"]):
            self.truth[self.ids.index(i)]=-k
        terms,opaque=E.R.symbols(self.p,labels)
        use=[terms[i] for i in self.ids]
        base,old=E.F.baseline("ns",0)
        paths=[E.F.ROOT/"fit_results/ns_seed0.json",E.M.ROOT/"results/ns_seed0.json"]
        local,prior=[json.loads(p.read_text()) for p in paths]
        self.inputs={str(p):records.sha(p) for p in (base,*paths)}
        self.vseg=slice(*prior["segment"])
        self.ax=[self.axes[0][self.vseg],*self.axes[1:]]
        opq={k:y[self.vseg] for k,y in opaque.items()}
        keys={f["selection"]["q0"]["key"] for f in prior["folds"]}
        assert len(keys)==1
        meta=prior["banks"][keys.pop()]
        self.validation=E.S.N.V.make_broad(self.ax,terms,opq,meta["plan"],meta["k"])
        self.basis=E.M.RealBasis(meta["k"],len(self.ax))
        modes=np.r_[np.arange(5),-np.arange(1,5)]
        freqs=[(modes if i==0 else np.array([0.]))*np.pi/(m*(a[1]-a[0]))
               for i,(a,m) in enumerate(zip(self.ax,meta["plan"]["ms"]))]
        self.time=E.Q.Bank(self.ax,use,opq,plan=meta["plan"],frequencies=freqs)
        self.folds=[]
        for g,group in enumerate(old["groups"]):
            idx=np.asarray(group["indices"])
            seg,tax,topq,plan=E.Q.mapped_plan(old,g,local["plans"][g]["plan"],self.axes,opaque)
            self.folds.append((idx,seg,E.Q.Bank(tax,use,topq,plan=plan),E.O.interpolation(self.axes[0],idx)[self.vseg]))


@lru_cache(maxsize=1)
def geometry():
    return Geometry()


@lru_cache(maxsize=6)
def anchor(seed):
    geo=geometry()
    raw=geo.p.noisy(5,seed)
    pilot,_=E.R.reconstruct(raw,geo.axes,np.arange(len(geo.axes[0])))
    return pilot,{v:E.R.WF.noise_model(y) for v,y in raw.items()}


def hashes(raw):
    return {v:E.R.digest(y) for v,y in raw.items()}


def field_digest(h):
    return hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest()


def replay(seed):
    geo=geometry()
    raw=geo.p.noisy(5,seed)
    fit=geo.estimate(raw,seed)
    src=PRIOR/"results"/f"ns_{seed}.json"
    rows={r["id"]:r for r in json.loads(src.read_text())["rows"]}
    expected=np.array([[f["local_time_on"]["D"] for f in rows[key]["folds"]] for key in CASES]).T
    actual=np.asarray(fit["scores"])[:,:2]
    np.testing.assert_allclose(actual,expected,rtol=3.e-8,atol=5.e-13)
    path=ROOT/"replay"/f"{seed}.npz"
    np.savez_compressed(path,**fit)
    pilot,model=anchor(seed)
    E.R.dump(path.with_suffix(".json"),dict(input=records.sha(src),geometry=geo.inputs,ids=geo.ids,
        target=geo.target,missing=geo.missing,truth=geo.truth,raw=hashes(raw),pilot=hashes(pilot),noise=model,
        max_difference=float(np.max(abs(expected-actual))),archive=records.sha(path)))
    return seed


def batch(job):
    tag,start=job
    geo=geometry()
    results=[]
    seeds=[]
    digests=[]
    for i in range(start,start+CHUNK):
        if tag=="fresh":
            seed=NEW+i
            raw=geo.p.noisy(5,seed)
        else:
            a=int(tag)
            pilot,model=anchor(a)
            rng=np.random.default_rng(np.random.SeedSequence([58283,a,i]))
            raw={v:y+E.R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in pilot.items()}
            seed=810000+(a-73001)*1000+i
        results.append(geo.estimate(raw,seed))
        seeds.append(seed)
        digests.append(field_digest(hashes(raw)))
    path=ROOT/"draws"/f"{tag}_{start:03}.npz"
    with path.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack([x[k] for x in results]) for k in results[0]},
                            seeds=seeds,field_hashes=digests)
    path.with_suffix(".pending").replace(path)
    return path.stem


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["replay","run"])
    opt=ap.parse_args()
    if opt.stage=="replay":
        (ROOT/"replay").mkdir(exist_ok=True)
        jobs=list(ANCHORS)
        fn=replay
    else:
        assert all((ROOT/"replay"/f"{a}.json").exists() for a in ANCHORS)
        (ROOT/"draws").mkdir(exist_ok=True)
        jobs=[(tag,i) for i in range(0,B,CHUNK) for tag in (*map(str,ANCHORS),"fresh")
              if not (ROOT/"draws"/f"{tag}_{i:03}.npz").exists()]
        fn=batch
    with ProcessPoolExecutor(max_workers=min(8,len(jobs) or 1)) as pool:
        futures=[pool.submit(fn,j) for j in jobs]
        for n,f in enumerate(as_completed(futures),1):
            result=f.result()
            if n%10==0 or n==len(jobs) or opt.stage=="replay":
                print(n,len(jobs),result,flush=True)
