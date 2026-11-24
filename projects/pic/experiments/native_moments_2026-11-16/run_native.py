"""Paired native/dense weak fitting and common physical-window validation."""
import native_quadrature as N
V,E,M,R,Q,O,F,np=N.V,N.E,N.M,N.R,N.Q,N.O,N.F,N.np
ROOT=N.ROOT
from pathlib import Path
import argparse,json,hashlib,time


def sources():
    paths=[Path(__file__),Path(N.__file__),Path(V.__file__),Path(E.__file__),Path(M.__file__),
           Path(O.__file__),Path(Q.__file__),Path(R.__file__),Path(R.WF.__file__)]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fine_plan(plan):
    return {k:([3*v[0],*v[1:]] if k in ('ms','ss','cut') else list(v)) for k,v in plan.items()}


def join(left,right):
    return np.concatenate([left,right],axis=-2)


def fit_all(stats,candidates):
    rows=[]
    for name,st in stats.items():
        for c in candidates:
            for method in ('ols','irls'):
                coef,info=E.ordinary(st['point'],c) if method=='ols' else E.irls(st['point'],st['draws'],c)
                rows.append(dict(bank=name,method=method,candidate=c['id'],coefficients=coef,fit=info))
    return rows


def run(name,seed):
    start=time.monotonic();base_path,old=F.baseline(name,0)
    paths=dict(base=base_path,old=E.ROOT/'results'/f'{name}_seed{seed}.json',
               local=F.ROOT/'fit_results'/f'{name}_seed0.json',
               broad=V.J.ROOT/'results'/f'{name}_seed0.json',
               validation=M.ROOT/'results'/f'{name}_seed0.json')
    saved={k:json.loads(p.read_text()) for k,p in paths.items() if k!='base'}
    p=R.datasets.load(name);axes=R.axes_of(p);panel,labels=R.panel(p)
    candidates=[c for c in panel if c['registered']];terms,opaque=R.symbols(p,labels);ci=labels.index(())
    raw=p.noisy(5,seed);indices=[np.asarray(g['indices']) for g in old['groups']]
    native_axes=[[axes[0][idx],*axes[1:]] for idx in indices]
    # Physical supports are chosen using only the saved geometry and grid indices.
    specs={};selection=[];fit_geometry=[]
    for g,ax in enumerate(native_axes):
        local=saved['local']['plans'][g]['plan']
        meta=saved['broad']['folds'][g]['banks']['broad0'];broad=dict(meta['plan'])
        broad['ms']=[(len(a)-2*c-1)//2 for a,c in zip(ax,broad['cut'])]
        broad['ss']=broad['ms'];broad['shape']=[1]*len(ax)
        local_spec=N.spec_from_plan(ax,local)
        vlocal=N.common_spec(native_axes,terms,local['q'],radii=local_spec['radii'])
        vbroad=N.common_spec(native_axes,terms,broad['q'],k=meta['k'])
        keys={}
        for tag,spec in [('local',vlocal),('broad',vbroad)]:
            key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
            specs[key]=dict(spec=spec,tag=tag,k=meta['k'] if tag=='broad' else 0);keys[tag]=key
        selection.append(keys);fit_geometry.append(dict(local=local,broad=broad,k=meta['k']))
    vseg=slice(*saved['validation']['segment']);vaxes=[axes[0][vseg],*axes[1:]]
    vopq={k:y[vseg] for k,y in opaque.items()}
    vmeta=next(meta for key,meta in saved['validation']['banks'].items() if key.startswith('q0_'))
    oldval=V.make_broad(vaxes,terms,vopq,vmeta['plan'],vmeta['k']);oldbasis=M.RealBasis(vmeta['k'],len(axes))
    validation={key:[] for key in specs};oldstats=[];folds=[];controls=[]
    print(name,seed,'native and mixed weak moments',flush=True)
    for g,idx in enumerate(indices):
        ax=native_axes[g];opq={k:y[idx] for k,y in opaque.items()};observed={v:y[idx] for v,y in raw.items()}
        model={v:R.WF.noise_model(y) for v,y in observed.items()};pilot,_=R.reconstruct(observed,axes,idx)
        assert all(R.digest(y)==saved['old']['groups'][g]['observed_sha256'][v] for v,y in observed.items())
        np.testing.assert_allclose([model[v][0] for v in observed],[saved['old']['groups'][g]['noise'][v][0] for v in observed],rtol=1e-14)
        ncov=O.Covariance(ax,np.eye(len(idx)),model)
        segment=slice(idx[0],idx[-1]+1);fax=[axes[0][segment],*axes[1:]];fopq={k:y[segment] for k,y in opaque.items()}
        tm=O.interpolation(axes[0],idx)[segment];fcov=O.Covariance(fax,tm,model)
        vm=O.interpolation(axes[0],idx)[vseg];vcov=O.Covariance(vaxes,vm,model)
        geom=fit_geometry[g];basis=M.RealBasis(geom['k'],len(ax));subset=V.natural_rows(geom['local'])
        fitting={}
        for tag,plan in [('local',geom['local']),('broad',geom['broad'])]:
            spec=N.spec_from_plan(ax,plan,geom['k'] if tag=='broad' else 0)
            native=N.Bank(ax,terms,opq,spec)
            fine=Q.Bank(fax,terms,fopq,plan=fine_plan(plan)) if tag=='local' else V.make_broad(fax,terms,fopq,fine_plan(plan),geom['k'])
            convert=(lambda matrix:matrix[...,subset,:]) if tag=='local' else basis.convert
            fitting['native_'+tag]=(native,convert)
            fitting['interpolated_'+tag]=(fine,convert)
        banks={};vrows={}
        for key,desc in specs.items():
            bank=N.Bank(ax,terms,opq,desc['spec'])
            convert=M.RealBasis(desc['k'],len(ax)).convert if desc['tag']=='broad' else (lambda x:x)
            banks[key]=(bank,convert)
            vrows[key]=dict(point=convert(O.matrix(bank,observed,ncov)),
                            pilot=convert(bank.matrix({v:y[idx] for v,y in pilot.items()})),draws=[])
        fvalues=O.map_values(tm,observed)
        stats={key:dict(point=convert(O.matrix(bank,observed if key.startswith('native') else fvalues,
                                              ncov if key.startswith('native') else fcov)),draws=[])
               for key,(bank,convert) in fitting.items()}
        oldpoint=oldbasis.convert(O.matrix(oldval,O.map_values(vm,observed),vcov))
        oldpilot=oldbasis.convert(oldval.matrix({v:y[vseg] for v,y in pilot.items()}));olddraws=[]
        rng=np.random.default_rng(np.random.SeedSequence([175103,seed,g]))
        for b in range(512):
            draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in observed.items()}
            fine_values=O.map_values(tm,draw)
            for key,(bank,convert) in fitting.items():
                value=O.matrix(bank,draw if key.startswith('native') else fine_values,ncov if key.startswith('native') else fcov)
                stats[key]['draws'].append(convert(value))
            for key,(bank,convert) in banks.items():vrows[key]['draws'].append(convert(O.matrix(bank,draw,ncov)))
            if b<128:olddraws.append(oldbasis.convert(O.matrix(oldval,O.map_values(vm,draw),vcov)))
        for st in stats.values():st['draws']=np.stack(st['draws'])
        for prefix in ('native','interpolated'):
            stats[prefix+'_mixed']={key:join(stats[prefix+'_local'][key],stats[prefix+'_broad'][key]) for key in ('point','draws')}
        fits=fit_all(stats,candidates)
        for key,st in vrows.items():
            st['draws']=np.stack(st['draws']);validation[key].append(st)
        for st in stats.values():
            assert np.isfinite(st['point']).all() and np.isfinite(st['draws']).all()
            np.testing.assert_allclose(st['draws'][:,:,ci],np.broadcast_to(st['point'][None,:,ci],st['draws'][:,:,ci].shape),atol=1e-12,rtol=1e-12)
        ow,owmeta=M.whitening(np.stack(olddraws),oldpilot);assert ow is not None
        oldstats.append(dict(point=oldpoint,pilot=oldpilot,weight=ow))
        folds.append(dict(training=g,rows=fits,geometry=geom,
                          fitting_rows={k:len(v['point']) for k,v in stats.items()},validation_selection=selection[g],
                          noise=model,raw_group_sha256={v:R.digest(y) for v,y in observed.items()}))
        controls.append((fitting,segment,tm,subset,basis))
        print(' group',g,'IRLS',[(r['bank'],r['candidate'],r['fit']['status']) for r in fits if r['method']=='irls'],
              round(time.monotonic()-start,1),'s',flush=True)
    # Freeze all common validation weights using training data only.
    validation_metrics=[]
    for g in range(3):
        keys=selection[g];data={tag:validation[key] for tag,key in keys.items()}
        data['mixed']=[{key:join(data['local'][h][key],data['broad'][h][key]) for key in ('point','pilot','draws')} for h in range(3)]
        metrics={}
        for tag,states in data.items():
            w,meta=M.whitening(states[g]['draws'],states[g]['pilot'])
            metrics[tag]=dict(states=states,weight=w,meta=meta)
        validation_metrics.append(metrics)
    # Fits and all weights now frozen. Clean values used below only for controls/evaluation.
    frozen=F.evaluation_grams(p,old,terms,opaque,labels);oldclean=oldbasis.convert(oldval.matrix({v:y[vseg] for v,y in p.data.items()}))
    by_id={c['id']:c for c in candidates};reproduction=0.
    for g,fold in enumerate(folds):
        fitting,segment,tm,subset,basis=controls[g]
        for kind in ('native_clean','interpolated_clean','full_clean'):
            mm={}
            values={v:y[indices[g]] for v,y in p.data.items()} if kind=='native_clean' else (
                O.map_values(tm,{v:y[indices[g]] for v,y in p.data.items()}) if kind=='interpolated_clean' else {v:y[segment] for v,y in p.data.items()})
            prefix='native' if kind=='native_clean' else 'interpolated'
            for tag in ('local','broad'):
                bank,convert=fitting[prefix+'_'+tag];mm[tag]=convert(bank.matrix(values))
            mm['mixed']=join(mm['local'],mm['broad'])
            for tag,mat in mm.items():
                for c in candidates:
                    coef,info=E.ordinary(mat,c)
                    fold['rows'].append(dict(bank=tag,method=kind,candidate=c['id'],coefficients=coef,fit=info))
        # Evaluate prior frozen fits too, so validation metrics have identical model comparisons.
        for row in saved['old']['folds'][g]['rows']:
            if row['method'] in ('raw_irls','recon_clsm'):
                fold['rows'].append(dict(bank='prior_'+row['bank'],method=row['method'],candidate=row['candidate'],
                                         coefficients=row['coefficients'],fit=row['fit']))
        left,right=[h for h in range(3) if h!=g];oldw=oldstats[g]['weight']
        oldscale=np.sum((oldw@oldstats[g]['pilot'])**2,axis=0)
        for row in fold['rows']:
            if row['coefficients'] is None:continue
            c=by_id[row['candidate']];coef=np.asarray(row['coefficients']);v=np.zeros(len(labels));v[c['yi']]=1.;v[c['xi']]-=coef
            row['frozen_clean_energy']=float(v@frozen[f'{g}_loo_oracle']@v/frozen[f'{g}_loo_clean_scale'][c['yi']])
            row['coefficient_error']=coef-c['exact']
            row['old_cross']=float((oldw@oldstats[left]['point']@v)@(oldw@oldstats[right]['point']@v)/oldscale[c['yi']])
            row['old_clean']=float(np.sum((oldw@oldclean@v)**2)/oldscale[c['yi']])
            if row['bank']=='prior_local_dense' and row['method']=='recon_clsm':
                expected=next(r['cross'] for r in saved['old']['folds'][g]['rows'] if r['bank']=='local_dense' and r['method']==row['method'] and r['candidate']==c['id'])
                np.testing.assert_allclose(row['old_cross'],expected,rtol=2e-10,atol=2e-12)
                reproduction=max(reproduction,abs(row['old_cross']-expected))
            for tag,metric in validation_metrics[g].items():
                states=metric['states']
                for method,w in [('plain',None),('white',metric['weight'])]:
                    if method=='white' and w is None:continue
                    a=M.transform(states[left]['point'],w)@v;b=M.transform(states[right]['point'],w)@v
                    pilot=M.transform(states[g]['pilot'],w)[:,c['yi']];den=float(pilot@pilot)
                    row[f'native_{tag}_{method}_cross']=float(a@b/den)
        fold['validation_weighting']={tag:v['meta'] for tag,v in validation_metrics[g].items()}
    aggregates=[]
    keys=sorted({(r['bank'],r['method'],r['candidate']) for f in folds for r in f['rows']})
    for bank,method,candidate in keys:
        rows=[r for f in folds for r in f['rows'] if (r['bank'],r['method'],r['candidate'])==(bank,method,candidate)]
        good=[r for r in rows if r['fit']['status']=='ok' and r['coefficients'] is not None]
        entry=dict(bank=bank,method=method,candidate=candidate,successful_folds=len(good),statuses=[r['fit']['status'] for r in rows])
        if len(good)==3:
            metrics=set.intersection(*(set(r) for r in good))
            for key in metrics:
                if key in ('frozen_clean_energy','old_cross','old_clean') or key.endswith('_cross'):
                    entry[key]=float(np.mean([r[key] for r in good]))
        aggregates.append(entry)
    out=dict(system=name,seed=seed,noise_percent=5,candidates=candidates,labels=labels,folds=folds,aggregates=aggregates,
             validation_specs=specs,baseline_score_reproduction_max_error=reproduction,
             raw_data_sha256={v:R.digest(y) for v,y in raw.items()},source_sha256=sources(),
             input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
             elapsed_seconds=time.monotonic()-start)
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    print('DONE',name,seed,round(out['elapsed_seconds'],1),'s',flush=True)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','kdv_cossin','wave','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[3,4,5]);args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            if not (ROOT/'results'/f'{name}_seed{seed}.json').exists():run(name,seed)


if __name__=='__main__':main()
