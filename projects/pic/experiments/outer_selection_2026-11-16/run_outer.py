"""Training/inner selection followed by an independent outer measurement record."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'native_moments_2026-11-16'))
import run_native as V
N,E,M,R,Q,O,F,np=V.N,V.E,V.M,V.R,V.Q,V.O,V.F,V.np
import argparse,json,hashlib,time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths={Path(__file__),ROOT/'protocol.txt',Path(R.WF._central4.__code__.co_filename),
           Path(R.datasets.__file__),Path(F.__file__)}
    paths.update(Path(p) for p in V.sources())
    return {str(p):sha(p) for p in sorted(paths)}


def choose(options):
    """Only training status and inner score enter this function."""
    feasible=[r for r in options if r['status']=='ok' and r['inner'] is not None and np.isfinite(r['inner'])]
    if not feasible:return dict(status='no_feasible_choice',bank=None,forced=False)
    best=min(feasible,key=lambda r:(r['inner'],r['bank']!='local'))
    return dict(status='ok',bank=best['bank'],forced=len(feasible)==1,inner=best['inner'])


def vector(c,coef,width):
    out=np.zeros(width);out[c['yi']]=1.;out[c['xi']]-=coef;return out


def cross(states,weight,scale,c,coef):
    v=vector(c,coef,states[0].shape[1])
    return float((weight@states[0]@v)@(weight@states[1]@v)/scale[c['yi']])


def run(name,seed):
    started=time.monotonic();outer_seed=seed+3
    base_path,old=F.baseline(name,0)
    geometry_path=N.ROOT/'results'/f'{name}_seed3.json'
    validation_path=M.ROOT/'results'/f'{name}_seed0.json'
    geometry=json.loads(geometry_path.read_text());validation=json.loads(validation_path.read_text())
    p=R.datasets.load(name);axes=R.axes_of(p);panel,labels=R.panel(p)
    candidates=[c for c in panel if c['registered']];terms,opaque=R.symbols(p,labels);ci=labels.index(())
    assert [c['id'] for c in candidates]==[c['id'] for c in geometry['candidates']]
    for c,previous in zip(candidates,geometry['candidates']):
        assert c['xi']==previous['xi'] and c['yi']==previous['yi']
        np.testing.assert_array_equal(c['exact'],previous['exact'])
    indices=[np.asarray(g['indices']) for g in old['groups']]
    assert all(not np.intersect1d(a,b).size for i,a in enumerate(indices) for b in indices[:i])
    axlist=[[axes[0][idx],*axes[1:]] for idx in indices]
    specs={f['validation_selection']['broad']:geometry['validation_specs'][f['validation_selection']['broad']]
           for f in geometry['folds']}
    vseg=slice(*validation['segment']);vaxes=[axes[0][vseg],*axes[1:]]
    vmeta=next(m for k,m in validation['banks'].items() if k.startswith('q0_'))
    oldbank=N.V.make_broad(vaxes,terms,{k:y[vseg] for k,y in opaque.items()},vmeta['plan'],vmeta['k'])
    oldbasis=M.RealBasis(vmeta['k'],len(axes))
    raw=p.noisy(5,seed);folds=[];states=[];boot_payload={}
    print(name,seed,'train/inner; outer record remains unopened',flush=True)
    for g,idx in enumerate(indices):
        ax=axlist[g];opq={k:y[idx] for k,y in opaque.items()};obs={v:y[idx] for v,y in raw.items()}
        model={v:R.WF.noise_model(y) for v,y in obs.items()};pilot,_=R.reconstruct(obs,axes,idx)
        cov=O.Covariance(ax,np.eye(len(idx)),model);geom=geometry['folds'][g]['geometry']
        subset=N.V.natural_rows(geom['local']);basis=M.RealBasis(geom['k'],len(axes))
        banks={}
        for tag in ('local','broad'):
            spec=N.spec_from_plan(ax,geom[tag],geom['k'] if tag=='broad' else 0)
            convert=(lambda a:a[...,subset,:]) if tag=='local' else basis.convert
            banks[tag]=(N.Bank(ax,terms,opq,spec),convert)
        vbanks={k:(N.Bank(ax,terms,opq,d['spec']),M.RealBasis(d['k'],len(axes)).convert) for k,d in specs.items()}
        tm=O.interpolation(axes[0],idx)[vseg];ocov=O.Covariance(vaxes,tm,model)
        pp={v:y[idx] for v,y in pilot.items()}
        stats={k:dict(point=convert(O.matrix(bank,obs,cov)),pilot=convert(bank.matrix(pp)),draws=[])
               for k,(bank,convert) in banks.items()}
        native={k:dict(point=convert(O.matrix(bank,obs,cov)),pilot=convert(bank.matrix(pp)),draws=[])
                for k,(bank,convert) in vbanks.items()}
        previous=dict(point=oldbasis.convert(O.matrix(oldbank,O.map_values(tm,obs),ocov)),
                      pilot=oldbasis.convert(oldbank.matrix({v:y[vseg] for v,y in pilot.items()})),draws=[])
        rng=np.random.default_rng(np.random.SeedSequence([175103,seed,g]))
        for b in range(512):
            draw={v:pp[v]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
            for tag,(bank,convert) in banks.items():stats[tag]['draws'].append(convert(O.matrix(bank,draw,cov)))
            for k,(bank,convert) in vbanks.items():native[k]['draws'].append(convert(O.matrix(bank,draw,cov)))
            if b<128:previous['draws'].append(oldbasis.convert(O.matrix(oldbank,O.map_values(tm,draw),ocov)))
        for st in [*stats.values(),*native.values(),previous]:st['draws']=np.stack(st['draws'])
        stats['mixed']={k:V.join(stats['local'][k],stats['broad'][k]) for k in ('point','pilot','draws')}
        rows=V.fit_all(stats,candidates)
        for st in stats.values():
            assert np.isfinite(st['draws']).all()
            np.testing.assert_allclose(st['draws'][:,:,ci],np.broadcast_to(st['point'][None,:,ci],st['draws'][:,:,ci].shape),rtol=1e-12,atol=1e-12)
        vkey=geometry['folds'][g]['validation_selection']['broad'];metrics={}
        for tag,st in [('native_broad',native[vkey]),('old_broad',previous)]:
            w,meta=M.whitening(st['draws'],st['pilot'])
            metrics[tag]=dict(weight=w,meta=meta,scale=None if w is None else np.sum((w@st['pilot'])**2,axis=0))
        states.append(dict(native={k:st['point'] for k,st in native.items()},old=previous['point'],metrics=metrics,vkey=vkey,
                           native_banks=vbanks,old_map=tm))
        folds.append(dict(training=g,rows=rows,geometry=geom,noise=model,
                          raw_group_sha256={v:R.digest(y) for v,y in obs.items()},
                          validation_weighting={k:st['meta'] for k,st in metrics.items()}))
        if name=='kdv_cossin':
            independent={tag:[] for tag in banks};brng=np.random.default_rng(np.random.SeedSequence([250611,seed,g]))
            for _ in range(64):
                draw={v:pp[v]+R.WF.ar_noise(brng,y.shape,*model[v][:2]) for v,y in obs.items()}
                for tag,(bank,convert) in banks.items():independent[tag].append(convert(O.matrix(bank,draw,cov)))
            independent={tag:np.stack(x) for tag,x in independent.items()}
            independent['mixed']=V.join(independent['local'],independent['broad'])
            for tag,st in stats.items():
                for key,value in st.items():boot_payload[f'g{g}_{tag}_{key}']=value
                boot_payload[f'g{g}_{tag}_independent']=independent[tag]
        print(' group',g,'fits',[(r['bank'],r['candidate'],r['fit']['status']) for r in rows if r['method']=='irls'],
              round(time.monotonic()-started,1),'s',flush=True)
    # Inner decisions are made without outer data or clean score fields.
    by_id={c['id']:c for c in candidates}
    for g,fold in enumerate(folds):
        other=[h for h in range(3) if h!=g];st=states[g]
        for row in fold['rows']:
            if row['fit']['status']!='ok' or row['coefficients'] is None:continue
            c=by_id[row['candidate']]
            for metric,meta in st['metrics'].items():
                if meta['weight'] is None:continue
                pp=[states[h]['native'][st['vkey']] if metric=='native_broad' else states[h]['old'] for h in other]
                row['inner_'+metric]=cross(pp,meta['weight'],meta['scale'],c,row['coefficients'])
        decisions=[]
        for c in candidates:
            for metric in ('native_broad','old_broad'):
                options=[]
                for bank in ('local','broad'):
                    r=next(r for r in fold['rows'] if r['candidate']==c['id'] and r['bank']==bank and r['method']=='irls')
                    options.append(dict(bank=bank,status=r['fit']['status'],inner=r.get('inner_'+metric)))
                decisions.append(dict(candidate=c['id'],rule=metric,options=options,**choose(options)))
        fold['decisions']=decisions
    if boot_payload:
        (ROOT/'training_arrays').mkdir(parents=True,exist_ok=True)
        np.savez_compressed(ROOT/'training_arrays'/f'{name}_seed{seed}.npz',**boot_payload)
    freeze_path=ROOT/'frozen'/f'{name}_seed{seed}.json'
    frozen=dict(system=name,seed=seed,outer_seed=outer_seed,candidates=candidates,labels=labels,folds=folds,
                raw_data_sha256={v:R.digest(y) for v,y in raw.items()},source_sha256=sources(),
                input_sha256={str(p):sha(p) for p in (base_path,geometry_path,validation_path)})
    R.dump(freeze_path,frozen);freeze_hash=sha(freeze_path)
    print(' frozen',name,seed,'now evaluating independent outer seed',outer_seed,flush=True)
    # The outer measurement record is first opened here, after saved decisions.
    outer=p.noisy(5,outer_seed);outer_points=[];outer_models=[]
    for g,idx in enumerate(indices):
        obs={v:y[idx] for v,y in outer.items()};model={v:R.WF.noise_model(y) for v,y in obs.items()}
        cov=O.Covariance(axlist[g],np.eye(len(idx)),model);tm=states[g]['old_map'];ocov=O.Covariance(vaxes,tm,model)
        native={k:convert(O.matrix(bank,obs,cov)) for k,(bank,convert) in states[g]['native_banks'].items()}
        oldpoint=oldbasis.convert(O.matrix(oldbank,O.map_values(tm,obs),ocov))
        outer_points.append(dict(native=native,old=oldpoint));outer_models.append(model)
    # Only diagnostic scoring now accesses clean fields and exact coefficients.
    grams=F.evaluation_grams(p,old,terms,opaque,labels)
    clean_native=[{k:convert(bank.matrix({v:y[indices[g]] for v,y in p.data.items()}))
                   for k,(bank,convert) in states[g]['native_banks'].items()} for g in range(3)]
    clean_old=oldbasis.convert(oldbank.matrix({v:y[vseg] for v,y in p.data.items()}))
    for g,fold in enumerate(folds):
        other=[h for h in range(3) if h!=g];st=states[g]
        for row in fold['rows']:
            if row['fit']['status']!='ok' or row['coefficients'] is None:continue
            c=by_id[row['candidate']];coef=np.asarray(row['coefficients']);v=vector(c,coef,len(labels))
            row['frozen_clean_energy']=float(v@grams[f'{g}_loo_oracle']@v/grams[f'{g}_loo_clean_scale'][c['yi']])
            row['coefficient_error']=coef-c['exact']
            for metric,meta in st['metrics'].items():
                if meta['weight'] is None:continue
                pp=[outer_points[h]['native'][st['vkey']] if metric=='native_broad' else outer_points[h]['old'] for h in other]
                cc=[clean_native[h][st['vkey']] if metric=='native_broad' else clean_old for h in other]
                row['outer_'+metric]=cross(pp,meta['weight'],meta['scale'],c,coef)
                row['clean_'+metric]=cross(cc,meta['weight'],meta['scale'],c,coef)
        for decision in fold['decisions']:
            if decision['status']!='ok':continue
            r=next(r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['bank']==decision['bank'] and r['method']=='irls')
            decision['evaluation']={k:v for k,v in r.items() if k.startswith(('inner_','outer_','clean_')) or k in ('frozen_clean_energy','coefficient_error','coefficients')}
    assert sha(freeze_path)==freeze_hash
    frozen.update(outer_data_sha256={v:R.digest(y) for v,y in outer.items()},outer_noise=outer_models,
                  selection_freeze_sha256=freeze_hash,elapsed_seconds=time.monotonic()-started)
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',frozen)
    print('DONE',name,seed,round(frozen['elapsed_seconds'],1),'s',flush=True)


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','kdv_cossin','wave','ks'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[6,7,8]);args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            assert seed in (6,7,8)
            if not (ROOT/'results'/f'{name}_seed{seed}.json').exists():run(name,seed)


if __name__=='__main__':main()
