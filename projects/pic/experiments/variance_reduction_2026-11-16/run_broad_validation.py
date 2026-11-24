"""Frozen coefficients on common-interior broad raw moments."""
import moments as M
O,J,R,Q,F,A,np,B=M.O,M.J,M.R,M.Q,M.F,M.A,M.np,M.B
V=M.V;P=J.L.P
from pathlib import Path
import json,time,hashlib,argparse
ROOT=M.ROOT


def evaluate(bank,entries,vectors,stats,g,w,uniform,signs,frozen):
    left,right=[h for h in range(3) if h!=g]
    point=[M.transform(v,w) for v in stats['point']];pilot=M.transform(stats['pilot'][g],w)
    unc=[M.transform(v,w) for v in stats['uncorrected']]
    clean=M.transform(stats['clean'],w);subclean=[M.transform(v,w) for v in stats['subclean']]
    scale=np.sum(pilot**2,axis=0)[[c['yi'] for c in entries]]
    assert np.all(scale>0)
    rl,rr=point[left]@vectors,point[right]@vectors;rc=clean@vectors
    ql,qr=subclean[left]@vectors,subclean[right]@vectors
    ul,ur=unc[left]@vectors,unc[right]@vectors
    clean_scale=np.sum(clean**2,axis=0)[[c['yi'] for c in entries]]
    values=dict(cross=np.sum(rl*rr,axis=0)/scale,ordinary=np.sum(rl**2+rr**2,axis=0)/(2*scale),
        uncorrected=np.sum(ul*ur,axis=0)/scale,clean_full=np.sum(rc**2,axis=0)/scale,
        clean_subsample=np.sum(ql*qr,axis=0)/scale,bank_clean_energy=np.sum(rc**2,axis=0)/clean_scale,
        frozen_clean_energy=np.einsum('ij,ik,kj->j',vectors,frozen[f'{g}_loo_oracle'],vectors)/frozen[f'{g}_loo_clean_scale'][[c['yi'] for c in entries]])
    samples={}
    for n in (64,128):
        el=M.transform(stats['draws'][left][:n],w)@vectors;er=M.transform(stats['draws'][right][:n],w)@vectors
        el=(el-el.mean(axis=0))*np.sqrt(n/(n-1));er=(er-er.mean(axis=0))*np.sqrt(n/(n-1))
        _,samples[n],_=P.sample_scores(rl,rr,el,er,scale,(uniform*n).astype(int),signs,left,right)
    rows=[dict(c,**{k:float(v[i]) for k,v in values.items()}) for i,c in enumerate(entries)]
    assert all(np.isfinite(v).all() for v in values.values())
    return rows,samples


def run(name,seed):
    started=time.monotonic();basepath,old=F.baseline(name,seed)
    jointpath=J.ROOT/'results'/f'{name}_seed{seed}.json';joint=json.loads(jointpath.read_text())
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels);ci=labels.index(())
    indices=[np.asarray(g['indices']) for g in old['groups']]
    segment=slice(max(i[0] for i in indices),min(i[-1] for i in indices)+1)
    ax=[axes[0][segment],*axes[1:]];opq={k:v[segment] for k,v in opaque.items()}
    raw=p.noisy(5,seed);pilots=[];maps=[]
    for h,idx in enumerate(indices):
        observed={v:y[idx] for v,y in raw.items()}
        assert all(R.digest(y)==old['groups'][h]['observed_sha256'][v] for v,y in observed.items())
        pp,_=R.reconstruct(observed,axes,idx);pilots.append(pp);maps.append(O.interpolation(axes[0],idx)[segment])
        assert ax[0][0]>=axes[0][idx[0]] and ax[0][-1]<=axes[0][idx[-1]]
    selected=[];banks={};basis={};metadata={};stats={}
    print(name,seed,'broad raw validation, fixed fits',flush=True)
    # All geometry and weighting inputs are training-only. The bank cache just
    # avoids recomputing identical fixed operators across fitting rotations.
    for g in range(3):
        one={};kmax=min(s['loo']['max_mode'] for s in old['groups'][g]['selection'])
        for extra in (0,1):
            bank,meta=B.identify_bank(ax,terms,opq,{v:y[segment] for v,y in pilots[g].items()},cs,kmax,extra)
            if bank is None:one[f'q{extra}']=dict(meta=meta,key=None);continue
            key=f"q{extra}_k{meta['k']}";one[f'q{extra}']=dict(meta=meta,key=key)
            if key not in banks:
                banks[key]=bank;basis[key]=M.RealBasis(meta['k'],len(ax));metadata[key]=meta
                stats[key]={k:[] for k in ('point','pilot','uncorrected','draws','subclean')}
        selected.append(one)
    for h,idx in enumerate(indices):
        cov=O.Covariance(ax,maps[h],old['groups'][h]['noise'])
        observed=O.map_values(maps[h],{v:y[idx] for v,y in raw.items()})
        for key,bank in banks.items():
            convert=basis[key].convert;st=stats[key]
            st['point'].append(convert(O.matrix(bank,observed,cov)))
            st['uncorrected'].append(convert(bank.matrix(observed)))
            st['pilot'].append(convert(bank.matrix({v:y[segment] for v,y in pilots[h].items()})))
            if h:np.testing.assert_allclose(st['point'][-1][:,ci],st['point'][0][:,ci],atol=3e-13,rtol=3e-13)
        nr=np.random.default_rng(np.random.SeedSequence([834721,seed,h]));draws={key:[] for key in banks}
        for _ in range(128):
            fields={v:pilots[h][v][idx]+R.WF.ar_noise(nr,y[idx].shape,*old['groups'][h]['noise'][v][:2]) for v,y in raw.items()}
            mapped=O.map_values(maps[h],fields)
            for key,bank in banks.items():draws[key].append(O.matrix(bank,mapped,cov))
        for key in banks:stats[key]['draws'].append(basis[key].convert(np.stack(draws[key])))
        print(' group',h,'moments complete',round(time.monotonic()-started,1),'s',flush=True)
    weights={};weight_meta={}
    for key,st in stats.items():
        weights[key]={};weight_meta[key]={}
        for g in range(3):
            weights[key][g]={'plain':None};weight_meta[key][g]={}
            for n in (64,128):
                w,meta=M.whitening(st['draws'][g][:n],st['pilot'][g]);weight_meta[key][g][f'white{n}']=meta
                if w is not None:weights[key][g][f'white{n}']=w
    # Clean controls enter only after banks, coefficients and weights freeze.
    for key,bank in banks.items():
        st=stats[key];convert=basis[key].convert
        st['clean']=convert(bank.matrix({v:y[segment] for v,y in p.data.items()}))
        for h,idx in enumerate(indices):st['subclean'].append(convert(bank.matrix(O.map_values(maps[h],{v:y[idx] for v,y in p.data.items()}))))
    frozen=F.evaluation_grams(p,old,terms,opaque,labels)
    rng=np.random.default_rng(np.random.SeedSequence([934171,seed]));uniform=rng.random((512,3));signs=rng.choice([-1.,1.],(512,3))
    folds=[];noise={};available={};failures=[]
    for g in range(3):
        entries=V.catalog(cs,joint['folds'][g]);vectors=np.zeros((len(labels),len(entries)))
        for i,c in enumerate(entries):vectors[c['yi'],i]=1.;vectors[c['xi'],i]-=c['coefficients']
        arms={}
        for q,selection in selected[g].items():
            key=selection['key']
            if key is None:failures.append(dict(training=g,arm=q,reason=selection['meta']['reason']));continue
            for method,w in weights[key][g].items():
                arm=q+'_'+method;rows,ss=evaluate(banks[key],entries,vectors,stats[key],g,w,uniform,signs,frozen)
                arms[arm]=rows;available.setdefault(arm,[]).append(g)
                for n,v in ss.items():noise.setdefault(f'{arm}_{n}',[]).append(v)
            for method,meta in weight_meta[key][g].items():
                if not meta['feasible']:failures.append(dict(training=g,arm=q+'_'+method,reason=meta['reason']))
        folds.append(dict(training=g,selection=selected[g],arms=arms))
    aggregates={};sample_out={};aggregate_rows={}
    for arm,done in available.items():
        if done!=[0,1,2]:continue
        rows=folds[0]['arms'][arm];values={metric:np.mean([[r[metric] for r in f['arms'][arm]] for f in folds],axis=0)
            for metric in ('cross','ordinary','uncorrected','clean_full','clean_subsample','bank_clean_energy','frozen_clean_energy')}
        for n in (64,128):sample_out[f'{arm}_{n}']=np.mean(noise[f'{arm}_{n}'],axis=0)
        aggregate_rows[arm]=[]
        for i,c in enumerate(rows):
            row={k:v for k,v in c.items() if k not in values and k not in ('coefficients','estimated_training_risk')}
            row.update({k:float(v[i]) for k,v in values.items()})
            for n in (64,128):row[f'sd{n}']=float(np.std(sample_out[f'{arm}_{n}'][:,i],ddof=1))
            aggregate_rows[arm].append(row)
        aggregates[arm]={}
        for kind in ('baseline','joint'):
            ids=[i for i,c in enumerate(rows) if c['kind']==kind]
            aggregates[arm][kind]={metric:dict(scores=v[ids],dominance=R.dominance(v[ids],old['instability'],cs))
                for metric,v in values.items() if metric in ('cross','ordinary','clean_full','clean_subsample')}
    out=dict(system=name,seed=seed,segment=[segment.start,segment.stop],candidates=cs,labels=labels,instability=old['instability'],
        folds=folds,banks=metadata,weighting=weight_meta,infeasible=failures,aggregates=aggregates,rows=aggregate_rows,
        elapsed_seconds=time.monotonic()-started,
        input_sha256={str(basepath):hashlib.sha256(basepath.read_bytes()).hexdigest(),str(jointpath):hashlib.sha256(jointpath.read_bytes()).hexdigest()},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(M.__file__),Path(O.__file__),Path(Q.__file__),Path(B.__file__),Path(V.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    np.savez_compressed(ROOT/'results'/f'{name}_seed{seed}_samples.npz',**sample_out)
    print('DONE',name,seed,round(out['elapsed_seconds'],1),'s',flush=True)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['wave','ks','ac','kdv_cossin']);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            if not (ROOT/'results'/f'{name}_seed{seed}.json').exists():run(name,seed)


if __name__=='__main__':main()
