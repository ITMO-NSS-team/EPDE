"""Whole-field paired refits with independently rebuilt training and validation weights."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"core15_implicit_correction_2026-11-18"
sys.path.insert(0,str(PRIOR))
from train import TrainingMoments, E, np
from evaluate import corrected_fit, score
from panel import records

ANCHORS=tuple(range(73001,73007))
B=128
NEW=750001
CHUNK=8


class Geometry:
    def __init__(self):
        self.p=E.R.datasets.load("ac")
        self.axes=E.R.axes_of(self.p)
        candidates,labels=E.R.panel(self.p)
        full,small=[next(c for c in candidates if c["id"]==key) for key in ("e0:reference","e0:drop_0")]
        ids=self.ids=sorted([full["yi"],*full["xi"]])
        self.target=ids.index(full["yi"])
        missing,=set(full["xi"])-set(small["xi"])
        self.missing=ids.index(missing)
        self.small=np.delete(np.arange(len(ids)),self.missing)
        self.truth=np.zeros(len(ids))
        self.truth[self.target]=1.
        for term,b in zip(full["xi"],full["exact"]):
            self.truth[ids.index(term)]=-b
        terms,opaque=E.R.symbols(self.p,labels)
        fitted_terms=[terms[i] for i in ids]
        base,old=E.F.baseline("ac",0)
        localpath=E.F.ROOT/"fit_results/ac_seed0.json"
        valpath=E.M.ROOT/"results/ac_seed0.json"
        local,prior=[json.loads(p.read_text()) for p in (localpath,valpath)]
        self.inputs={str(p):records.sha(p) for p in (base,localpath,valpath)}
        self.vseg=slice(*prior["segment"])
        self.ax=[self.axes[0][self.vseg],*self.axes[1:]]
        opq={k:y[self.vseg] for k,y in opaque.items()}
        keys=[f["selection"]["q0"]["key"] for f in prior["folds"]]
        assert len(set(keys))==1
        meta=prior["banks"][keys[0]]
        self.validation=E.S.N.V.make_broad(self.ax,terms,opq,meta["plan"],meta["k"])
        self.basis=E.M.RealBasis(meta["k"],len(self.ax))
        modes=np.r_[np.arange(5),-np.arange(1,5)]
        freqs=[(modes if i==0 else np.array([0.]))*np.pi/(m*(a[1]-a[0]))
               for i,(a,m) in enumerate(zip(self.ax,meta["plan"]["ms"]))]
        self.time=E.Q.Bank(self.ax,fitted_terms,opq,plan=meta["plan"],frequencies=freqs)
        self.folds=[]
        for g,grp in enumerate(old["groups"]):
            idx=np.asarray(grp["indices"])
            seg,tax,topq,plan=E.Q.mapped_plan(old,g,local["plans"][g]["plan"],self.axes,opaque)
            localbank=E.Q.Bank(tax,fitted_terms,topq,plan=plan)
            mapping=E.O.interpolation(self.axes[0],idx)[self.vseg]
            self.folds.append((idx,seg,localbank,mapping))

    def estimate(self,raw,seed):
        points,whites,weights,vectors,grams,norms=[],[],[],[],[],[]
        for g,(idx,seg,localbank,mapping) in enumerate(self.folds):
            obs={v:y[idx] for v,y in raw.items()}
            model={v:E.R.WF.noise_model(y) for v,y in obs.items()}
            pilot,_=E.R.reconstruct(obs,self.axes,idx)
            part={v:y[self.vseg] for v,y in pilot.items()}
            local=TrainingMoments(localbank.matrix({v:y[seg] for v,y in pilot.items()}))
            time=TrainingMoments(self.time.matrix(part))
            rng=np.random.default_rng(np.random.SeedSequence([1171,seed,g]))
            for _ in range(B):
                draw={v:pilot[v][idx]+E.R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
                field,_=E.R.reconstruct(draw,self.axes,idx)
                local.update(localbank.matrix({v:y[seg] for v,y in field.items()}))
                time.update(self.time.matrix({v:y[self.vseg] for v,y in field.items()}))
            gram=sum(s.corrected()/len(s.point) for s in (local,time))
            mat=np.vstack([s.point/np.sqrt(len(s.point)) for s in (local,time)])
            norm=np.linalg.norm(mat,axis=0)
            grams.append(gram)
            norms.append(norm)
            vfull,_,s1=corrected_fit(gram,norm)
            short,_,s2=corrected_fit(gram[np.ix_(self.small,self.small)],norm[self.small])
            if s1!="ok" or s2!="ok":
                raise ValueError(f"Unresolved fit for {seed}, fold {g}: {s1}, {s2}")
            vdrop=np.zeros_like(vfull)
            vdrop[self.small]=short
            fixed_drop=self.truth.copy()
            fixed_drop[self.missing]=0.
            vectors.append(np.stack([vfull,vdrop,self.truth,fixed_drop]))
            cov=E.O.Covariance(self.ax,mapping,model)
            def raw_matrix(values):
                return E.O.matrix(self.validation,E.O.map_values(mapping,values),cov)
            valpilot=self.basis.convert(self.validation.matrix(part))
            points.append(self.basis.convert(raw_matrix(obs))[:,self.ids])
            rng=np.random.default_rng(np.random.SeedSequence([834721,seed,g]))
            draws=[]
            for _ in range(B):
                draw={v:pilot[v][idx]+E.R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
                draws.append(raw_matrix(draw))
            white,info=E.M.whitening(self.basis.convert(np.stack(draws)),valpilot)
            if white is None:
                raise ValueError(f"Infeasible whitening for {seed}, fold {g}: {info}")
            whites.append(white)
            weights.append(np.linalg.norm(white@valpilot,axis=0)[self.ids])
        cross,values=[],[]
        for g,w in enumerate(whites):
            left,right=[w@p for h,p in enumerate(points) if h!=g]
            cross.append(left.T@right)
            values.append([score(left,right,weights[g],v) for v in vectors[g]])
        return dict(scores=values,vec=vectors,cross=cross,scale=weights,gram=grams,norm=norms)


@lru_cache(maxsize=1)
def geometry():
    return Geometry()


@lru_cache(maxsize=6)
def anchor(seed):
    geo=geometry()
    raw=geo.p.noisy(5,seed)
    pilot,_=E.R.reconstruct(raw,geo.axes,np.arange(len(geo.axes[0])))
    model={v:E.R.WF.noise_model(y) for v,y in raw.items()}
    return pilot,model


def replay(seed):
    geo=geometry()
    raw=geo.p.noisy(5,seed)
    fit=geo.estimate(raw,seed)
    path=PRIOR/"results"/f"ac_{seed}.json"
    old=json.loads(path.read_text())
    rows={r["id"]:r for r in old["rows"]}
    truth=np.array([[f["local_time_on"]["D"] for f in rows[key]["folds"]]
                    for key in ("e0:reference","e0:drop_0")]).T
    err=float(np.max(abs(np.asarray(fit["scores"])[:,:2]-truth)))
    np.testing.assert_allclose(np.asarray(fit["scores"])[:,:2],truth,rtol=3e-8,atol=5e-13)
    pilot,model=anchor(seed)
    meta=dict(seed=seed,previous=records.sha(path),inputs=geo.inputs,ids=geo.ids,target=geo.target,
              diffusion=geo.missing,raw={v:E.R.digest(y) for v,y in raw.items()},
              pilot={v:E.R.digest(y) for v,y in pilot.items()},model=model,max_replay_error=err)
    dest=ROOT/"replay"/f"{seed}.npz"
    np.savez_compressed(dest,**fit)
    meta["archive"]=records.sha(dest)
    E.R.dump(dest.with_suffix(".json"),meta)
    return str(seed),err


def batch(job):
    tag,start=job
    geo=geometry()
    fields=[]
    raw_hashes=[]
    rep_seeds=[]
    for i in range(start,start+CHUNK):
        if tag=="fresh":
            seed=NEW+i
            raw=geo.p.noisy(5,seed)
        else:
            a=int(tag)
            sig,model=anchor(a)
            rng=np.random.default_rng(np.random.SeedSequence([48283,a,i]))
            raw={v:y+E.R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in sig.items()}
            seed=800000+(a-73001)*1000+i
        fields.append(geo.estimate(raw,seed))
        rep_seeds.append(seed)
        raw_hashes.append(E.R.digest(raw["u"]))
    dest=ROOT/"draws"/f"{tag}_{start:03}.npz"
    np.savez_compressed(dest,**{k:np.stack([f[k] for f in fields]) for k in fields[0]},
                        seeds=rep_seeds,field_hashes=raw_hashes)
    return dest.stem


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["replay","run"])
    args=ap.parse_args()
    if args.stage=="replay":
        (ROOT/"replay").mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=6) as pool:
            for result in pool.map(replay,ANCHORS):
                print(result,flush=True)
    else:
        assert all((ROOT/"replay"/f"{a}.json").exists() for a in ANCHORS)
        (ROOT/"draws").mkdir(exist_ok=True)
        jobs=[(tag,i) for i in range(0,B,CHUNK) for tag in (*map(str,ANCHORS),"fresh")
              if not (ROOT/"draws"/f"{tag}_{i:03}.npz").exists()]
        with ProcessPoolExecutor(max_workers=7) as pool:
            for f in as_completed([pool.submit(batch,j) for j in jobs]):
                print(f.result(),flush=True)
