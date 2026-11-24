"""Plug-in joint objective uncertainty checked against fresh measurement noise."""
import criteria as K
R,F,M,np,ROOT=K.R,K.F,K.M,K.np,K.ROOT
S,N,O=K.S,K.S.N,K.S.O
from pathlib import Path
import json,time,argparse
B=128


def sources():
    paths=[Path(__file__),Path(K.__file__),Path(R.__file__),Path(R.SV.__file__),Path(O.__file__),Path(M.__file__),
           Path(N.V.__file__),ROOT/'uncertainty_protocol.txt']
    return {str(p):S.sha(p) for p in paths}


def run(name,seed):
    start=time.monotonic();basepath,old=F.baseline(name,seed)
    vpath=M.ROOT/'results'/f'{name}_seed{seed}.json';prior=json.loads(vpath.read_text())
    apath=ROOT/'results'/vpath.name;ablation=json.loads(apath.read_text())
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    K.common_width(max(len(c['xi']) for c in cs));raw=p.noisy(5,seed)
    indices=[np.asarray(g['indices']) for g in old['groups']];seg=slice(*prior['segment'])
    ax=[axes[0][seg],*axes[1:]];opq={k:y[seg] for k,y in opaque.items()}
    keys=[f['selection']['q0']['key'] for f in prior['folds']];banks={};basis={}
    for key in set(keys):
        meta=prior['banks'][key];banks[key]=N.V.make_broad(ax,terms,opq,meta['plan'],meta['k']);basis[key]=M.RealBasis(meta['k'],len(ax))
    states=[]
    for h,idx in enumerate(indices):
        obs={v:y[idx] for v,y in raw.items()};pilot,_=R.reconstruct(obs,axes,idx);tm=O.interpolation(axes[0],idx)[seg]
        cov=O.Covariance(ax,tm,old['groups'][h]['noise']);rng=np.random.default_rng(np.random.SeedSequence([834721,seed,h]))
        draws={key:[] for key in banks}
        for _ in range(128):
            draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,*old['groups'][h]['noise'][v][:2]) for v,y in obs.items()}
            fields=O.map_values(tm,draw)
            for key,bank in banks.items():draws[key].append(basis[key].convert(O.matrix(bank,fields,cov)))
        own=keys[h];pilot_matrix=basis[own].convert(banks[own].matrix({v:y[seg] for v,y in pilot.items()}))
        white,wmeta=M.whitening(np.stack(draws[own]),pilot_matrix);assert white is not None
        entries=[r for r in prior['folds'][h]['arms']['q0_white128'] if r['kind']=='joint']
        assert [r['candidate'] for r in entries]==[c['id'] for c in cs]
        vectors=np.zeros((len(labels),len(cs)))
        for j,c in enumerate(cs):vectors[c['yi'],j]=1.;vectors[c['xi'],j]-=entries[j]['coefficients']
        states.append(dict(map=tm,cov=cov,white=white,scale=np.sum((white@pilot_matrix)**2,axis=0)[[c['yi'] for c in cs]],
                           vectors=vectors,weight_meta=wmeta))
    plan={k:tuple(v) for k,v in old['base_plan'].items()};dbank=R.LocalBank(axes,terms,opaque,plan)
    def evaluate(fields,correct=True):
        matrix=dbank.matrix(fields);dyad=[];prediction=[];weight=np.ones(len(matrix))
        for c in cs:
            x,y=matrix[:,c['xi']],matrix[:,c['yi']]
            with np.errstate(over='ignore',invalid='ignore'):
                dyad.append(float(np.sum(R.SV.dyad_scores(x,y,weight,plan['shape'],fit_intercept=False))) if len(c['xi']) else 0.)
            prediction.append(K.prediction_dyad(x,y,plan['shape']))
        points=[]
        for h,idx in enumerate(indices):
            mapped=O.map_values(states[h]['map'],{v:y[idx] for v,y in fields.items()})
            points.append({key:basis[key].convert(O.matrix(bank,mapped,states[h]['cov']) if correct else bank.matrix(mapped)) for key,bank in banks.items()})
        scores=[]
        for g,st in enumerate(states):
            left,right=[h for h in range(3) if h!=g];key=keys[g]
            a=st['white']@points[left][key]@st['vectors'];b=st['white']@points[right][key]@st['vectors']
            scores.append(np.sum(a*b,axis=0)/st['scale'])
        result=np.stack([np.mean(scores,axis=0),dyad,prediction]);assert np.isfinite(result[0]).all()
        return result
    observed=evaluate(raw)
    np.testing.assert_allclose(observed[0],ablation['discrepancy']['q0_white128/joint'],rtol=2e-9,atol=2e-11)
    np.testing.assert_allclose(observed[1],ablation['instability']['dyad_sum'],rtol=2e-10,atol=2e-11)
    np.testing.assert_allclose(observed[2],ablation['instability']['dyad_prediction'],rtol=2e-10,atol=2e-11)
    pilot,fitinfo=R.reconstruct(raw,axes,np.arange(len(axes[0])))
    model={v:R.WF.noise_model(y) for v,y in raw.items()}
    pilot_center=evaluate(pilot,correct=False)
    # No metric/pilot/model is selected using clean fields below.
    clean_center=evaluate(p.data,correct=False)
    bootstrap=[];repeats=[];rng=np.random.default_rng(np.random.SeedSequence([663103,seed]))
    for b in range(B):
        pseudo={v:pilot[v]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in raw.items()}
        bootstrap.append(evaluate(pseudo))
        repeats.append(evaluate(p.noisy(5,390000+1000*seed+b)))
        if (b+1)%32==0:print(name,seed,'measurement pairs',b+1,round(time.monotonic()-start,1),'s',flush=True)
    bootstrap=np.stack(bootstrap);repeats=np.stack(repeats)
    arrpath=ROOT/'uncertainty'/f'{name}_seed{seed}.npz';arrpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(arrpath,observed=observed,pilot_center=pilot_center,clean_center=clean_center,bootstrap=bootstrap,benchmark=repeats)
    out=dict(system=name,seed=seed,draws=B,candidates=cs,axes=['discrepancy','dyad_sum','dyad_prediction'],
        conditional_scope='Main fitted coefficients, validation weights/corrections and geometry fixed; dyad slab refits recomputed.',
        estimated_noise=model,pilot_reconstruction=fitinfo,actual_iid_sigma={v:.05*float(np.std(y)) for v,y in p.data.items()},
        score_reproduction_max_error=float(np.max(np.abs(observed[0]-ablation['discrepancy']['q0_white128/joint']))),
        bootstrap_nonfinite=int(np.sum(~np.isfinite(bootstrap))),benchmark_nonfinite=int(np.sum(~np.isfinite(repeats))),
        actual_noise_seeds=[390000+1000*seed+b for b in range(B)],
        raw_data_sha256={v:R.digest(y) for v,y in raw.items()},pilot_sha256={v:R.digest(y) for v,y in pilot.items()},
        input_sha256={str(p):S.sha(p) for p in (basepath,vpath,apath)},source_sha256=sources(),array_sha256=S.sha(arrpath),
        elapsed_seconds=time.monotonic()-start)
    R.dump(arrpath.with_suffix('.json'),out)
    print(name,seed,'UNCERTAINTY DONE',round(out['elapsed_seconds'],1),'s',flush=True)


if __name__=='__main__':
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--system',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    if not (ROOT/'uncertainty'/f'{a.system}_seed{a.seed}.json').exists():run(a.system,a.seed)
