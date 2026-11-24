"""Whole-field Allen–Cahn comparisons of all registered single additions/deletions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
AC=ROOT.parent/"ac_diffusion_uncertainty_2026-11-18"
sys.path.insert(0,str(AC))
import paired as ac
from paired import ANCHORS, B, CHUNK, E, np, records, TrainingMoments, corrected_fit, score

NEW=780001


class Panel:
    def __init__(self):
        # The held-out covariance pools all terms of the historical panel.
        self.geom=ac.geometry()
        g=self.geom
        cases,labels=E.R.panel(g.p)
        self.cases=[c for c in cases if c["id"]=="e0:reference" or c["label"].startswith(("drop_","add_"))]
        self.ids=[sorted([c["yi"],*c["xi"]]) for c in self.cases]
        assert len(self.cases)==10 and self.cases[0]["id"]=="e0:reference"
        terms,opaque=E.R.symbols(g.p,labels)
        self.count=len(terms)
        _,old=E.F.baseline("ac",0)
        local=json.loads((E.F.ROOT/"fit_results/ac_seed0.json").read_text())
        prior=json.loads((E.M.ROOT/"results/ac_seed0.json").read_text())
        key=prior["folds"][0]["selection"]["q0"]["key"]
        plan=prior["banks"][key]["plan"]
        modes=np.r_[np.arange(5),-np.arange(1,5)]
        freqs=[(modes if j==0 else np.array([0.]))*np.pi/(m*(a[1]-a[0]))
               for j,(a,m) in enumerate(zip(g.ax,plan["ms"]))]
        self.timebank=E.Q.Bank(g.ax,terms,{k:y[g.vseg] for k,y in opaque.items()},plan=plan,frequencies=freqs)
        self.localbanks=[]
        for i in range(3):
            _,ax,op,plan=E.Q.mapped_plan(old,i,local["plans"][i]["plan"],g.axes,opaque)
            self.localbanks.append(E.Q.Bank(ax,terms,op,plan=plan))

    def estimate(self,raw,seed):
        g=self.geom
        points,whites,scales,vecs,grams,norms=[],[],[],[],[],[]
        for h,(idx,seg,_,mapping) in enumerate(g.folds):
            obs={k:y[idx] for k,y in raw.items()}
            noise={k:E.R.WF.noise_model(y) for k,y in obs.items()}
            pilot,_=E.R.reconstruct(obs,g.axes,idx)
            def blocks(field):
                return (self.localbanks[h].matrix({k:y[seg] for k,y in field.items()}),
                        self.timebank.matrix({k:y[g.vseg] for k,y in field.items()}))
            local,temporal=[TrainingMoments(m) for m in blocks(pilot)]
            rng=np.random.default_rng(np.random.SeedSequence([1171,seed,h]))
            def draw(rng):
                return {k:pilot[k][idx]+E.R.WF.ar_noise(rng,y.shape,*noise[k][:2]) for k,y in obs.items()}
            for _ in range(B):
                reconstructed,_=E.R.reconstruct(draw(rng),g.axes,idx)
                l,t=blocks(reconstructed)
                local.update(l)
                temporal.update(t)
            gram=sum(st.corrected()/len(st.point) for st in (local,temporal))
            norm=np.linalg.norm(np.vstack([st.point/np.sqrt(len(st.point)) for st in (local,temporal)]),axis=0)
            vs=np.zeros((len(self.ids),self.count))
            for j,ids in enumerate(self.ids):
                v,_,status=corrected_fit(gram[np.ix_(ids,ids)],norm[ids])
                if status!="ok":
                    raise ArithmeticError((seed,h,self.cases[j]["id"],status))
                vs[j,ids]=v
            vecs.append(vs)
            grams.append(gram)
            norms.append(norm)
            cov=E.O.Covariance(g.ax,mapping,noise)
            def val(x):
                return E.O.matrix(g.validation,E.O.map_values(mapping,x),cov)
            points.append(g.basis.convert(val(obs)))
            rng=np.random.default_rng(np.random.SeedSequence([834721,seed,h]))
            vp=g.basis.convert(g.validation.matrix({k:y[g.vseg] for k,y in pilot.items()}))
            draws=g.basis.convert(np.stack([val(draw(rng)) for _ in range(B)]))
            white,info=E.M.whitening(draws,vp)
            if white is None:
                raise ArithmeticError((seed,h,info))
            whites.append(white)
            scales.append(np.linalg.norm(white@vp,axis=0))
        scores,cross=[],[]
        for h,white in enumerate(whites):
            left,right=[white@p for i,p in enumerate(points) if i!=h]
            cross.append(left.T@right)
            scores.append([score(left,right,scales[h],v) for v in vecs[h]])
        return dict(scores=scores,cross=cross,scale=scales,vec=vecs,gram=grams,norm=norms)


@lru_cache(maxsize=1)
def panel():
    return Panel()


def observed(seed):
    p=panel()
    result=p.estimate(p.geom.p.noisy(5,seed),seed)
    previous=ac.PRIOR/"results"/f"ac_{seed}.json"
    rows={r["id"]:r for r in json.loads(previous.read_text())["rows"]}
    expected=np.array([[f["local_time_on"]["D"] for f in rows[c["id"]]["folds"]] for c in p.cases]).T
    np.testing.assert_allclose(result["scores"],expected,atol=5e-12,rtol=2e-8)
    path=ROOT/"anchors"/f"{seed}.npz"
    np.savez_compressed(path,**result)
    path.with_suffix(".json").write_text(json.dumps(dict(cases=[c["id"] for c in p.cases],supports=p.ids,
        source=records.sha(previous),archive=records.sha(path),error=float(np.max(abs(result["scores"]-expected))))))
    return str(seed)


def run_chunk(job):
    tag,start=job
    p=panel()
    fits,digests,seeds=[],[],[]
    for i in range(start,start+CHUNK):
        if tag=="fresh":
            seed=NEW+i
            raw=p.geom.p.noisy(5,seed)
        else:
            a=int(tag)
            pilot,model=ac.anchor(a)
            rng=np.random.default_rng(np.random.SeedSequence([48283,a,i]))
            raw={k:y+E.R.WF.ar_noise(rng,y.shape,*model[k][:2]) for k,y in pilot.items()}
            seed=800000+(a-73001)*1000+i
        digests.append(E.R.digest(raw["u"]))
        seeds.append(seed)
        fits.append(p.estimate(raw,seed))
    arrays={k:np.stack([f[k] for f in fits]) for k in fits[0]}
    metadata=dict(seeds=seeds,raw_hash=digests)
    name=f"{tag}_{start:03}.npz"
    if tag!="fresh":
        with np.load(AC/"draws"/name) as prev:
            assert np.array_equal(prev["field_hashes"],digests)
            j=next(j for j,c in enumerate(p.cases) if c["id"]=="e0:drop_0")
            actual=arrays["scores"][:,:,[0,j]]
            np.testing.assert_allclose(actual,prev["scores"][:,:,:2],atol=5e-12,rtol=2e-8)
            metadata["replay_error"]=np.max(abs(actual-prev["scores"][:,:,:2]))
    else:
        metadata["freeze_hash"]=records.sha(ROOT/"ranges.json")
    path=ROOT/"draws"/name
    with path.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**arrays,**metadata)
    path.with_suffix(".pending").replace(path)
    return name


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["observed","bootstrap","fresh"])
    stage=parser.parse_args().stage
    if stage=="observed":
        (ROOT/"anchors").mkdir(exist_ok=True)
        jobs=list(ANCHORS)
        func=observed
    else:
        assert all((ROOT/"anchors"/f"{a}.json").exists() for a in ANCHORS)
        if stage=="fresh":
            assert (ROOT/"ranges.json").is_file()
        (ROOT/"draws").mkdir(exist_ok=True)
        tags=list(map(str,ANCHORS)) if stage=="bootstrap" else ["fresh"]
        jobs=[(t,i) for i in range(0,B,CHUNK) for t in tags if not (ROOT/"draws"/f"{t}_{i:03}.npz").exists()]
        func=run_chunk
    with ProcessPoolExecutor(max_workers=min(8,len(jobs) or 1)) as pool:
        for i,f in enumerate(as_completed([pool.submit(func,j) for j in jobs]),1):
            name=f.result()
            if i%8==0 or i==len(jobs):
                print(i,len(jobs),name,flush=True)
