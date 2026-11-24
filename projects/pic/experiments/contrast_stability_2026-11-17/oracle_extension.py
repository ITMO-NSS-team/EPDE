import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
from pathlib import Path
import argparse,json,time,subprocess,sys
from concurrent.futures import ThreadPoolExecutor


def run(name):
    start=time.monotonic();seed=21
    parent=ROOT/'results'/f'{name}_seed21.json';meta=json.loads(parent.read_text());a=np.load(parent.with_suffix('.npz'))
    frozen=json.loads((ROOT/'frozen'/parent.name).read_text());op=ROOT/'oracle_diagnostic'/f'{name}.json'
    oracle=json.loads(op.read_text());o=np.load(op.with_suffix('.npz'))
    p=R.datasets.load(name);cs,labels=R.panel(p);refs=[j for j,c in enumerate(cs) if c['label']=='reference'];assert len(refs)==1
    j=refs[0];ix=np.concatenate([np.arange(g['start'],g['stop']) for g in frozen['groups'] if g['candidate']==j])
    C.K.common_width(max(len(c['xi']) for c in cs));_,old=E.F.baseline(name,0);plan={k:tuple(v) for k,v in old['base_plan'].items()}
    terms,opaque=R.symbols(p,labels);bank=R.LocalBank(R.axes_of(p),terms,opaque,plan);ext=C.Extractor([cs[j]],plan['shape'])
    clean=ext.extract(bank.matrix(p.data))[0];np.testing.assert_array_equal(clean,a['clean_contrasts'][ix])
    observed,dyad,_=ext.extract(bank.matrix(p.noisy(5,21)))
    np.testing.assert_array_equal(observed,a['observed_contrasts'][ix]);np.testing.assert_array_equal(dyad,a['observed_dyad'][refs])
    models={'pilot_estimated':C.NoiseModel(ext,a['pilot_contrasts'][ix],a['calibration_contrasts'][:,ix])}
    radii={'pilot_estimated':float(np.quantile(a['bootstrap_noise_q'][:,j],.95))}
    for tag in ('clean_estimated_noise','clean_actual_noise'):
        models[tag]=C.NoiseModel(ext,clean,o[tag+'_calibration'][:,ix]);radii[tag]=float(np.quantile(o[tag+'_noise_q_check'][:,j],.95))
        np.testing.assert_allclose(models[tag].score(a['benchmark_contrasts'][:,ix]-clean)[:,0],o[tag+'_noise_q_actual'][:,j],rtol=2e-13,atol=2e-13)
    rng=np.random.default_rng(np.random.SeedSequence([720132,21]));cc=[];actual=[]
    for b in range(1024):
        field={v:y+.05*float(np.std(y))*rng.standard_normal(y.shape) for v,y in p.data.items()}
        cc.append(ext.extract(bank.matrix(field))[0]);actual.append(ext.extract(bank.matrix(p.noisy(5,950000+b)))[0])
    cc=np.stack(cc);actual=np.stack(actual);arrays=dict(check_contrasts=cc,actual_contrasts=actual);rows=[]
    for tag,model in models.items():
        q=model.score(actual-clean)[:,0];arrays[tag+'_actual_q']=q
        rows.append(dict(control=tag,radius_draws=128,radius=radii[tag],actual_draws=1024,coverage_count=int(np.sum(q<=radii[tag])),coverage=float(np.mean(q<=radii[tag]))))
    qcheck=models['clean_actual_noise'].score(cc-clean)[:,0];arrays['clean_actual_noise_check_q']=qcheck
    radius=float(np.quantile(qcheck,.95));q=arrays['clean_actual_noise_actual_q']
    rows.append(dict(control='clean_actual_noise_extended_check',radius_draws=1024,radius=radius,actual_draws=1024,coverage_count=int(np.sum(q<=radius)),coverage=float(np.mean(q<=radius))))
    dest=ROOT/'oracle_extension';dest.mkdir(exist_ok=True);arr=dest/f'{name}.npz';np.savez_compressed(arr,**arrays)
    result=dict(system=name,seed=21,reference=cs[j]['id'],rows=rows,reference_replay='PASS',
        source_sha256={str(p):E.S.sha(p) for p in [Path(__file__),Path(C.__file__),ROOT/'oracle_extension_protocol.txt']},
        input_sha256={str(p):E.S.sha(p) for p in [parent,parent.with_suffix('.npz'),ROOT/'frozen'/parent.name,op,op.with_suffix('.npz')]},
        array_sha256=E.S.sha(arr),actual_noise_seeds=list(range(950000,951024)),seconds=time.monotonic()-start)
    R.dump(arr.with_suffix('.json'),result);print(json.dumps(dict(system=name,seconds=result['seconds'],rows=rows)),flush=True)


def worker(name):
    log=ROOT/'logs'/f'oracle_extension_{name}.txt'
    with log.open('w',encoding='utf-8') as f:
        ret=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--system',name],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return dict(system=name,exit_code=ret.returncode)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--system');args=ap.parse_args()
    if args.system:run(args.system)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,['ac','kdv_cossin','ks','wave']))
        R.dump(ROOT/'oracle_extension_processes.json',rows);print(json.dumps(rows));sys.exit(any(r['exit_code'] for r in rows))
