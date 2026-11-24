"""Frozen-coefficient held-out raw weak validation with explicit offset controls."""
import raw_operator as O
J,R,Q,F,A,np=O.J,O.R,O.Q,O.F,O.A,O.np
L,E,P=J.L,J.E,J.L.P
from pathlib import Path
import json,hashlib,time,argparse
ROOT=O.ROOT


def catalog(cs,fold):
    entries=[]
    for kind,arm in [('baseline','local_fixed_local_clsm'),('joint','local_joint_all_128')]:
        for c,row in zip(cs,fold['arms'][arm]):
            entries.append(dict(id=kind+'/'+c['id'],kind=kind,candidate=c['id'],slot=c['slot'],
                coefficients=row['coefficients'],xi=c['xi'],yi=c['yi'],registered=c['registered'],label=c['label']))
    for i,c in enumerate(cs):
        if c['label']!='reference':continue
        for option,entry in fold['options'][i]['options'].items():
            entries.append(dict(id='option/'+c['id']+'/'+option,kind='option',candidate=c['id'],slot=c['slot'],
                option=option,coefficients=entry['coefficients'],xi=c['xi'],yi=c['yi'],
                estimated_training_risk=entry['diagnostics']['128']['risk']))
        entries.append(dict(id='exact/'+c['id'],kind='exact',candidate=c['id'],slot=c['slot'],
            coefficients=c['exact'],xi=c['xi'],yi=c['yi']))
    return entries


def run(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    paths=dict(joint=J.ROOT/'results'/f'{name}_seed{seed}.json',local=L.ROOT/'results'/f'{name}_seed{seed}.json',
        plan=F.ROOT/'fit_results'/f'{name}_seed{seed}.json')
    saved={k:json.loads(p.read_text()) for k,p in paths.items()}
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels);ci=labels.index(())
    raw=p.noisy(5,seed);groups=E.reconstruct_groups(p,old)
    indices=[np.asarray(m['indices']) for m in old['groups']]
    interp=[O.interpolation(axes[0],idx) for idx in indices]
    for h,idx in enumerate(indices):assert all(R.digest(y[idx])==old['groups'][h]['observed_sha256'][v] for v,y in raw.items())
    frozen=F.evaluation_grams(p,old,terms,opaque,labels)
    rng=np.random.default_rng(np.random.SeedSequence([934171,seed]));uniform=rng.random((512,3));signs=rng.choice([-1.,1.],(512,3))
    folds=[];samples={64:[],128:[]};reproduction=0.
    print(name,seed,'raw validation, fixed fits',flush=True)
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,saved['plan']['plans'][g]['plan'],axes,opaque)
        bank=Q.Bank(ax,terms,opq,plan=plan);entries=catalog(cs,saved['joint']['folds'][g]);vectors=np.zeros((len(labels),len(entries)))
        for j,c in enumerate(entries):vectors[c['yi'],j]=1.;vectors[c['xi'],j]-=c['coefficients']
        reach=max(R.WF.leftover_orders(m,len(ax))[0] for m in bank.monos)*R.WF.FD_REACH
        valid_time=[]
        for row in np.any(np.stack(bank.kernels[0])!=0,axis=0):
            temporal=np.flatnonzero(row)+segment.start
            valid_time.append(temporal.min()-reach>=max(i[0] for i in indices) and temporal.max()+reach<=min(i[-1] for i in indices))
        mask=np.repeat(valid_time,int(np.prod(bank.shape[1:])))
        assert np.any(mask)
        point=[];uncorrected=[];smooth=[];draws=[];subclean=[];pilot=None;quad=[]
        for h,idx in enumerate(indices):
            tm=interp[h][segment];cov=O.Covariance(ax,tm,old['groups'][h]['noise'])
            observed=O.map_values(tm,{v:y[idx] for v,y in raw.items()})
            mm=O.matrix(bank,observed,cov);uu=bank.matrix(observed)
            np.testing.assert_allclose(O.matrix(bank,observed),uu,rtol=3e-13,atol=3e-11)
            np.testing.assert_allclose(mm[:,ci],1.,atol=3e-13)
            point.append(mm);uncorrected.append(uu)
            sm=np.stack([bank.matrix({v:y[segment] for v,y in f.items()}) for f in groups[h]])
            smooth.append(2*sm[0]-sm[1:].mean(axis=0))
            if h==g:pilot=sm[0]
            subclean.append(bank.matrix(O.map_values(tm,{v:y[idx] for v,y in p.data.items()})))
            nr=np.random.default_rng(np.random.SeedSequence([834721,seed,h]));mc=[]
            for _ in range(128):
                perturbed={v:groups[h][0][v][idx]+R.WF.ar_noise(nr,y[idx].shape,*old['groups'][h]['noise'][v][:2]) for v,y in raw.items()}
                mc.append(O.matrix(bank,O.map_values(tm,perturbed),cov))
            draws.append(np.stack(mc))
        scale=np.sum(pilot**2,axis=0)[[c['yi'] for c in entries]]
        assert np.all(scale>0);left,right=[h for h in range(3) if h!=g]
        original_smooth=np.sum((smooth[left]@vectors)*(smooth[right]@vectors),axis=0)/scale
        expected=np.array([r['cross'] for r in saved['local']['folds'][g]['arms']['local_continuous']])
        np.testing.assert_allclose(original_smooth[:len(cs)],expected,rtol=3e-8,atol=2e-9)
        reproduction=max(reproduction,float(np.max(np.abs(original_smooth[:len(cs)]-expected)/np.maximum(1,np.abs(expected)))))
        point=[m[mask] for m in point];uncorrected=[m[mask] for m in uncorrected]
        smooth=[m[mask] for m in smooth];subclean=[m[mask] for m in subclean];draws=[m[:,mask] for m in draws]
        rl,rr=point[left]@vectors,point[right]@vectors
        sl,sr=smooth[left]@vectors,smooth[right]@vectors
        ul,ur=uncorrected[left]@vectors,uncorrected[right]@vectors
        ql,qr=subclean[left]@vectors,subclean[right]@vectors
        clean=bank.matrix({v:y[segment] for v,y in p.data.items()})[mask];rc=clean@vectors
        ns=len(clean);offset=-vectors[ci];clean_scale=np.sum(clean**2,axis=0)[[c['yi'] for c in entries]]
        moments=dict(raw_cross=np.sum(rl*rr,axis=0)/scale,smooth_cross=np.sum(sl*sr,axis=0)/scale,original_smooth_cross=original_smooth,
            uncorrected_cross=np.sum(ul*ur,axis=0)/scale,raw_ordinary=np.sum(rl**2+rr**2,axis=0)/(2*scale),
            clean_subsample_cross=np.sum(ql*qr,axis=0)/scale,clean_full=np.sum(rc**2,axis=0)/scale,
            local_clean_energy=np.sum(rc**2,axis=0)/clean_scale,
            frozen_clean_energy=np.einsum('ij,ik,kj->j',vectors,frozen[f'{g}_loo_oracle'],vectors)/frozen[f'{g}_loo_clean_scale'][[c['yi'] for c in entries]],
            raw_without_offset=np.sum((rl+offset)*(rr+offset),axis=0)/scale,
            smooth_without_offset=np.sum((sl+offset)*(sr+offset),axis=0)/scale,
            clean_without_offset=np.sum((rc+offset)**2,axis=0)/scale,
            raw_mean_part=ns*rl.mean(axis=0)*rr.mean(axis=0)/scale,
            smooth_mean_part=ns*sl.mean(axis=0)*sr.mean(axis=0)/scale,
            clean_mean_part=ns*rc.mean(axis=0)**2/scale,
            raw_left_mean=rl.mean(axis=0),raw_right_mean=rr.mean(axis=0),clean_mean=rc.mean(axis=0),offset=offset)
        for n in (64,128):
            el=draws[left][:n]@vectors;er=draws[right][:n]@vectors
            el=(el-el.mean(axis=0))*np.sqrt(n/(n-1));er=(er-er.mean(axis=0))*np.sqrt(n/(n-1))
            _,noise,_=P.sample_scores(rl,rr,el,er,scale,(uniform*n).astype(int),signs,left,right)
            samples[n].append(noise)
        for h in range(3):
            quad.append(dict(group=h,column_relative_error=np.linalg.norm(subclean[h]-clean,axis=0)/np.maximum(np.linalg.norm(clean,axis=0),np.finfo(float).tiny)))
        rows=[dict(c,**{k:float(v[j]) for k,v in moments.items()}) for j,c in enumerate(entries)]
        assert all(np.isfinite(v).all() for v in moments.values())
        folds.append(dict(training=g,quadrature=bank.metadata,retained_rows=int(mask.sum()),original_rows=int(len(mask)),subsample_controls=quad,rows=rows))
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    scores={k:np.mean([[r[k] for r in f['rows']] for f in folds],axis=0) for k in moments}
    noise={n:np.mean(v,axis=0) for n,v in samples.items()}
    aggregates={}
    for kind in ('baseline','joint'):
        ids=[i for i,c in enumerate(folds[0]['rows']) if c['kind']==kind]
        aggregates[kind]={k:dict(scores=ss[ids],dominance=R.dominance(ss[ids],old['instability'],cs)) for k,ss in scores.items() if k in ('raw_cross','smooth_cross','original_smooth_cross','clean_full','clean_subsample_cross','uncorrected_cross')}
    rows=[]
    for i,c in enumerate(folds[0]['rows']):
        row={k:v for k,v in c.items() if k not in moments and k not in ('coefficients','estimated_training_risk')}
        row.update({k:float(v[i]) for k,v in scores.items()});row['sd64']=float(np.std(noise[64][:,i],ddof=1));row['sd128']=float(np.std(noise[128][:,i],ddof=1))
        ref=next(j for j,x in enumerate(folds[0]['rows']) if x['kind']=='baseline' and x['slot']==c['slot'] and x['label']=='reference')
        row['gap_to_reference']=float(scores['raw_cross'][i]-scores['raw_cross'][ref])
        row['paired_gap_sd']=float(np.std(noise[128][:,i]-noise[128][:,ref],ddof=1));rows.append(row)
    out=dict(system=name,seed=seed,candidates=cs,labels=labels,instability=old['instability'],folds=folds,rows=rows,aggregates=aggregates,
        covariance_draws=128,paired_samples=512,baseline_scaled_error=reproduction,elapsed_seconds=time.monotonic()-started,
        input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(O.__file__),Path(Q.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    np.savez_compressed(ROOT/'results'/f'{name}_seed{seed}_samples.npz',**{f'noise{n}':v for n,v in noise.items()})
    print('DONE',name,seed,round(out['elapsed_seconds'],1),'s',flush=True)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['wave','ks','ac','kdv_cossin']);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    for n in args.systems:
        for s in args.seeds:
            if not (ROOT/'results'/f'{n}_seed{s}.json').exists():run(n,s)


if __name__=='__main__':main()
