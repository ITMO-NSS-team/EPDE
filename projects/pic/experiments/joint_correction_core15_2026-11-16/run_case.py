"""Frozen core15 factorial experiment; native moments, independent outer noise."""
import methods as A
S,N,E,M,R,O,F,np,ROOT=A.S,A.N,A.E,A.M,A.R,A.O,A.F,A.np,A.ROOT
from pathlib import Path
import json,time,argparse
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']
METHODS=('ols','irls','gls','corrected_ols','corrected_gls')
RULES=('broad_white','broad_transfer','mixed_transfer')


def sources():
    paths={Path(p) for p in S.sources()};paths.update([Path(__file__),Path(A.__file__),ROOT/'protocol.txt'])
    return {str(p):S.sha(p) for p in sorted(paths)}


def join(a,b):return S.V.join(a,b)


def fit_all(stats,candidates):
    rows=[];metadata={}
    for bank,st in stats.items():
        w,meta=M.whitening(st['draws'],st['pilot']);metadata[bank]=meta
        plain=A.prepare(st['draws'],st['pilot']);weighted=None if w is None else A.prepare(st['draws'],st['pilot'],w)
        for c in candidates:
            for method in METHODS:
                if method=='ols':coef,info=E.ordinary(st['point'],c)
                elif method=='irls':coef,info=E.irls(st['point'],st['draws'],c)
                elif method=='corrected_ols':coef,info=A.normal(st['point'],c,plain)
                elif weighted is None:coef,info=None,dict(status='infeasible_covariance')
                else:coef,info=A.normal(st['point'],c,weighted,corrected=method=='corrected_gls')
                rows.append(dict(bank=bank,method=method,option=bank+'/'+method,candidate=c['id'],coefficients=coef,fit=info))
    return rows,metadata


def run(name,seed):
    start=time.monotonic();base_path,old=F.baseline(name,0)
    paths=dict(base=base_path,local=F.ROOT/'fit_results'/f'{name}_seed0.json',
               broad=N.V.J.ROOT/'results'/f'{name}_seed0.json')
    saved={k:json.loads(p.read_text()) for k,p in paths.items() if k!='base'}
    p=R.datasets.load(name);axes=R.axes_of(p);panel,labels=R.panel(p)
    candidates=[c for c in panel if c['registered']];terms,opaque=R.symbols(p,labels);ci=labels.index(())
    idxs=[np.asarray(g['indices']) for g in old['groups']];axlist=[[axes[0][idx],*axes[1:]] for idx in idxs]
    assert all(not np.intersect1d(a,b).size for i,a in enumerate(idxs) for b in idxs[:i])
    specs={};selection=[];geometry=[]
    for g,ax in enumerate(axlist):
        local=saved['local']['plans'][g]['plan'];meta=saved['broad']['folds'][g]['banks']['broad0'];broad=dict(meta['plan'])
        broad['ms']=[(len(a)-2*c-1)//2 for a,c in zip(ax,broad['cut'])];broad['ss']=broad['ms'];broad['shape']=[1]*len(ax)
        radii=N.spec_from_plan(ax,local)['radii'];keys={}
        for tag,spec in [('local',N.common_spec(axlist,terms,local['q'],radii=radii)),('broad',N.common_spec(axlist,terms,broad['q'],k=meta['k']))]:
            key=S.hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
            specs[key]=dict(spec=spec,k=meta['k'] if tag=='broad' else 0,tag=tag);keys[tag]=key
        selection.append(keys);geometry.append(dict(local=local,broad=broad,k=meta['k']))
    raw=p.noisy(5,seed);states=[];folds=[]
    print(name,seed,'START',flush=True)
    for g,idx in enumerate(idxs):
        ax=axlist[g];obs={v:y[idx] for v,y in raw.items()};opq={k:y[idx] for k,y in opaque.items()}
        model={v:R.WF.noise_model(y) for v,y in obs.items()};pilot,_=R.reconstruct(obs,axes,idx);pp={v:y[idx] for v,y in pilot.items()}
        cov=O.Covariance(ax,np.eye(len(idx)),model);geom=geometry[g];subset=N.V.natural_rows(geom['local']);basis=M.RealBasis(geom['k'],len(ax))
        banks={}
        for tag in ('local','broad'):
            spec=N.spec_from_plan(ax,geom[tag],geom['k'] if tag=='broad' else 0)
            banks[tag]=(N.Bank(ax,terms,opq,spec),(lambda a:a[...,subset,:]) if tag=='local' else basis.convert)
        vbanks={key:(N.Bank(ax,terms,opq,d['spec']),M.RealBasis(d['k'],len(ax)).convert if d['tag']=='broad' else (lambda a:a)) for key,d in specs.items()}
        stats={tag:dict(point=convert(O.matrix(bank,obs,cov)),pilot=convert(bank.matrix(pp)),draws=[]) for tag,(bank,convert) in banks.items()}
        vals={key:dict(point=convert(O.matrix(bank,obs,cov)),pilot=convert(bank.matrix(pp)),draws=[]) for key,(bank,convert) in vbanks.items()}
        rng=np.random.default_rng(np.random.SeedSequence([175103,seed,g]))
        for _ in range(512):
            draw={v:pp[v]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
            for tag,(bank,convert) in banks.items():stats[tag]['draws'].append(convert(O.matrix(bank,draw,cov)))
            for key,(bank,convert) in vbanks.items():vals[key]['draws'].append(convert(O.matrix(bank,draw,cov)))
        for st in [*stats.values(),*vals.values()]:st['draws']=np.stack(st['draws'])
        stats['mixed']={key:join(stats['local'][key],stats['broad'][key]) for key in ('point','pilot','draws')}
        for st in stats.values():
            assert np.isfinite(st['point']).all() and np.isfinite(st['draws']).all()
            np.testing.assert_allclose(st['draws'][:,:,ci],np.broadcast_to(st['point'][None,:,ci],st['draws'][:,:,ci].shape),rtol=1e-12,atol=1e-12)
        rows,fitmeta=fit_all(stats,candidates)
        frequencies=[]
        for sel in old['groups'][g]['selection']:
            k=sel['loo']['max_mode'];frequencies.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*sel['spacing'])
        rich=R.SpectralBank(axes,terms,opaque,frequencies).matrix(pilot)
        own={tag:vals[key] for tag,key in selection[g].items()}
        own['mixed']={k:join(own['local'][k],own['broad'][k]) for k in ('point','pilot','draws')}
        weights={tag:M.whitening(own[tag]['draws'],own[tag]['pilot']) for tag in ('broad','mixed')}
        metrics={}
        for c in candidates:
            metrics[c['id']]={}
            for rule in RULES:
                tag='mixed' if rule=='mixed_transfer' else 'broad';w,meta=weights[tag]
                if w is None:t,info=None,dict(status='infeasible_covariance')
                elif rule=='broad_white':t,info=w,dict(status='ok',scale=float(np.sum((w@own[tag]['pilot'][:,c['yi']])**2)))
                else:t,info=A.transfer(own[tag]['pilot'],w,rich,c)
                metrics[c['id']][rule]=dict(tag=tag,transform=t,**info)
        states.append(dict(points={k:v['point'] for k,v in vals.items()},banks=vbanks,metrics=metrics))
        folds.append(dict(training=g,rows=rows,geometry=geom,validation_keys=selection[g],metrics=metrics,
                          fit_weighting=fitmeta,validation_weighting={tag:x[1] for tag,x in weights.items()},noise=model,
                          raw_group_sha256={v:R.digest(y) for v,y in obs.items()}))
        print(name,seed,'fold',g,'failures',sum(r['fit']['status']!='ok' for r in rows),round(time.monotonic()-start,1),'s',flush=True)
    by_id={c['id']:c for c in candidates}
    def matrices(points,g,tag):
        keys=selection[g];return [points[h][keys['broad']] if tag=='broad' else join(points[h][keys['local']],points[h][keys['broad']]) for h in range(3) if h!=g]
    for g,fold in enumerate(folds):
        for row in fold['rows']:
            if row['fit']['status']!='ok':continue
            c=by_id[row['candidate']]
            for rule,metric in fold['metrics'][c['id']].items():
                if metric['status']=='ok':row['inner_'+rule]=A.score(matrices([s['points'] for s in states],g,metric['tag']),metric['transform'],metric['scale'],c,row['coefficients'])
        decisions=[]
        for c in candidates:
            for rule in RULES:
                options=[dict(option=r['option'],status=r['fit']['status'],inner=r.get('inner_'+rule)) for r in fold['rows']
                         if r['candidate']==c['id'] and r['method'] in ('irls','corrected_gls')]
                decisions.append(dict(candidate=c['id'],rule=rule,options=options,**A.choose(options)))
        fold['decisions']=decisions
    frozen_path=ROOT/'frozen'/f'{name}_seed{seed}.json'
    out=dict(system=name,seed=seed,outer_seed=seed+3,noise_percent=5,candidates=candidates,labels=labels,folds=folds,
             validation_specs=specs,source_sha256=sources(),input_sha256={str(p):S.sha(p) for p in paths.values()},
             raw_data_sha256={v:R.digest(y) for v,y in raw.items()})
    R.dump(frozen_path,out);freeze_hash=S.sha(frozen_path)
    print(name,seed,'FROZEN; opening outer',seed+3,flush=True)
    outer=p.noisy(5,seed+3);outer_points=[];clean_points=[];outer_noise=[]
    for g,idx in enumerate(idxs):
        obs={v:y[idx] for v,y in outer.items()};model={v:R.WF.noise_model(y) for v,y in obs.items()}
        cov=O.Covariance(axlist[g],np.eye(len(idx)),model)
        outer_points.append({k:convert(O.matrix(bank,obs,cov)) for k,(bank,convert) in states[g]['banks'].items()})
        clean_points.append({k:convert(bank.matrix({v:y[idx] for v,y in p.data.items()})) for k,(bank,convert) in states[g]['banks'].items()});outer_noise.append(model)
    cache=F.ROOT/'evaluation_cache'/f"{old['system']}_s{old['seed']}_b{old['draws']}.npz";assert cache.exists()
    grams=F.evaluation_grams(p,old,terms,opaque,labels)
    for g,fold in enumerate(folds):
        for row in fold['rows']:
            if row['fit']['status']!='ok':continue
            c=by_id[row['candidate']];coef=np.asarray(row['coefficients']);v=S.vector(c,coef,len(labels))
            row['frozen_clean_energy']=float(v@grams[f'{g}_loo_oracle']@v/grams[f'{g}_loo_clean_scale'][c['yi']])
            row['coefficient_error']=coef-c['exact']
            for rule,metric in fold['metrics'][c['id']].items():
                if metric['status']!='ok':continue
                for tag,points in [('outer',outer_points),('clean',clean_points)]:row[tag+'_'+rule]=A.score(matrices(points,g,metric['tag']),metric['transform'],metric['scale'],c,coef)
        for decision in fold['decisions']:
            if decision['status']!='ok':continue
            row=next(r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['option']==decision['option'])
            decision['evaluation']={k:v for k,v in row.items() if k.startswith(('inner_','outer_','clean_')) or k in ('frozen_clean_energy','coefficients','coefficient_error')}
    assert S.sha(frozen_path)==freeze_hash
    out.update(outer_data_sha256={v:R.digest(y) for v,y in outer.items()},outer_noise=outer_noise,selection_freeze_sha256=freeze_hash,
               elapsed_seconds=time.monotonic()-start)
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    print(name,seed,'DONE',round(out['elapsed_seconds'],1),'s',flush=True)


if __name__=='__main__':
    if S.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--system',choices=SYSTEMS,required=True);ap.add_argument('--seed',type=int,choices=(12,13,14),required=True);a=ap.parse_args()
    if not (ROOT/'results'/f'{a.system}_seed{a.seed}.json').exists():run(a.system,a.seed)
