"""Independent spatial-channel weak measurements and odd-grid local corrections."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from common import ROOT, SEEDS, E, TrainingMoments, np, records
from training import geometry, stack


@lru_cache(maxsize=8)
def temporal_weights(times):
    t=np.asarray(times)
    out=[]
    for radius in (3,5,7):
        centers=np.arange(radius,len(t)-radius)
        assert len(centers)
        weights,_=E.Q.axis_weights(t,0,centers,radius,3,np.array([0.]),1,3)
        w,dt=weights[0],-weights[1]
        for k in range(1,4):
            np.testing.assert_allclose(dt @ t**k,w @ (k*t**(k-1)),atol=1e-11)
        out.append((w,dt))
    return out


def label_names(labels):
    names=[]
    for term in labels:
        if not term:
            names.append("constant")
            continue
        (name,args),=term
        params=dict(args)
        if name=="u":
            names.append({1:"u",2:"square",3:"cube"}[params["power"]])
        elif name=="x":
            assert params["dim"]==0
            names.append({1:"time",2:"time2"}[params["power"]])
        elif name in ("cos","sin"):
            assert params["dim"]==0 and params["freq"]==2 and params["power"]==1
            names.append(name)
        else:
            names.append({"d^2u/dx1^2":"laplacian","du/dx0":"dt"}[name])
    assert len(set(names))==len(names)
    return names


def fourier_side(values,t,x,period,names,noise=True,highest=3):
    """All input timestamps and positions belong to one spatial channel of one outer fold."""
    assert len(x)>=2 and np.allclose(np.diff(x),period/len(x))
    modes=np.arange(1,highest+1)
    phase=2*np.pi*x[:,None]*modes/period
    phi=np.column_stack([np.ones(len(x)),np.sqrt(2)*np.cos(phase),np.sqrt(2)*np.sin(phase)])/len(x)
    freq2=(2*np.pi/period*np.r_[0,modes,modes])**2
    second=np.roll(values,1,axis=1)-2*values+np.roll(values,-1,axis=1)
    sigma=np.median(abs(second))/(0.6744897501960817*np.sqrt(6)) if noise else 0.
    moments={"u":values@phi, "square":(values**2-sigma**2)@phi,
             "cube":(values**3-3*sigma**2*values)@phi}
    deterministic=dict(constant=np.ones_like(t),time=t,time2=t**2,sin=np.sin(2*t),cos=np.cos(2*t))
    for name,signal in deterministic.items():
        m=np.zeros_like(moments["u"])
        m[:,0]=signal
        moments[name]=m
    moments["laplacian"]=-moments["u"]*freq2
    blocks={k:[] for k in ("mean","low")}
    for w,dt in temporal_weights(tuple(t)):
        columns=[dt@moments["u"] if name=="dt" else w@moments[name] for name in names]
        mat=np.stack(columns,axis=-1)
        for key,count in (("mean",1),("low",len(freq2))):
            sub=mat[:,:count,:].reshape(-1,len(names))
            blocks[key].append(sub/np.sqrt(3*len(sub)))
    return {k:np.vstack(v) for k,v in blocks.items()},float(sigma)


def paired_measurement(raw,t,x,names,channels,noise=True,highest=3):
    period=len(x)*(x[1]-x[0])
    sides=[fourier_side(raw[:,offset::stride],t,x[offset::stride],period,names,noise,highest)
           for offset,stride in channels]
    out={}
    for key in ("mean","low"):
        left,right=[s[0][key] for s in sides]
        cross=left.T@right
        out[key+"_gram"]=(cross+cross.T)/2
        out[key+"_norm"]=np.sqrt((np.sum(left**2,axis=0)+np.sum(right**2,axis=0))/2)
        out[key+"_left"]=left
        out[key+"_right"]=right
    out["noise"]=[s[1] for s in sides]
    return out


def interpolate_odd(native):
    full=np.empty((len(native),2*native.shape[1]))
    full[:,1::2]=native
    full[:,::2]=(native+np.roll(native,1,axis=1))/2
    return full


def odd_local(odd,axes,idx,seg,bank,common,time,seed,g):
    native_axes=[axes[0],axes[1][1::2]]
    pilot,_=E.R.reconstruct({"u":odd},native_axes,idx)
    model=E.R.WF.noise_model(odd)
    def matrices(values):
        full=interpolate_odd(values)
        return bank.matrix({"u":full[seg]}),time.matrix({"u":full[common]})
    stats=[TrainingMoments(v) for v in matrices(pilot["u"])]
    rng=np.random.default_rng(np.random.SeedSequence([3971,seed,g]))
    for _ in range(128):
        pert=pilot["u"][idx]+E.R.WF.ar_noise(rng,odd.shape,*model[:2])
        refit,_=E.R.reconstruct({"u":pert},native_axes,idx)
        for st,m in zip(stats,matrices(refit["u"])):
            st.update(m)
    gram=sum(st.corrected()/len(st.point) for st in stats)
    norm=np.linalg.norm(stack(stats[0].point,stats[1].point),axis=0)
    return gram,norm


def measure(seed,directory=ROOT,extended=False):
    path=directory/"measurements"/f"ac_{seed}.npz"
    if path.with_suffix(".json").exists():
        return seed
    p,axes,folds,_,_=geometry("ac")
    _,labels=E.R.panel(p)
    names=label_names(labels)
    raw=p.noisy(5,seed)["u"]
    bundle={}
    for g,(idx,seg,banks,common,time,*_) in enumerate(folds):
        sub=raw[idx]
        for tag,channels in (("full",((0,2),(1,2))),("even",((0,4),(2,4)))):
            z=paired_measurement(sub,axes[0][idx],axes[1],names,channels)
            for k,v in z.items():
                bundle[f"{g}_{tag}_{k}"]=v
        if extended:
            high=paired_measurement(sub,axes[0][idx],axes[1],names,((0,2),(1,2)),highest=12)
            for k in ("gram","norm","left","right"):
                bundle[f"{g}_full_high_{k}"]=high["low_"+k]
        gram,norm=odd_local(sub[:,1::2],axes,idx,seg,banks[0],common,time,seed,g)
        bundle[f"{g}_odd_gram"]=gram
        bundle[f"{g}_odd_norm"]=norm
        bundle[f"{g}_indices"]=idx
    np.savez_compressed(path,**bundle)
    path.with_suffix(".json").write_text(json.dumps(dict(archive=records.sha(path),raw=E.R.digest(raw),
        names=names,protocol=records.sha(ROOT/"protocol.txt"))))
    return seed


if __name__=="__main__":
    (ROOT/"measurements").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for seed in pool.map(measure,SEEDS):
            print(seed,flush=True)
