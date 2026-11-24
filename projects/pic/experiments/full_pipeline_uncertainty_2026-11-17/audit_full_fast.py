from bridge import *
from full_pipeline_fast import Experiment,cutoff,MC
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor
import time,traceback,argparse
spec=importlib.util.spec_from_file_location('new_region_checks',S.ROOT/'audit_joint.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
spec=importlib.util.spec_from_file_location('saved_temporal_checks',PREVIOUS/'audit_time.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)

def close(x,y):
    scale=max(float(np.max(np.abs(y))),np.finfo(float).tiny);np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-10*scale)

def direct_metric(e,pilot,model):
    baseline=e.matrices(pilot,correct=False);rows={b:[] for b in e.banks}
    for j in range(MC):
        rng=np.random.default_rng(np.random.SeedSequence([984105,e.system_id,e.pid,j]));fields={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in pilot.items()}
        for b,m in e.matrices(fields,model).items():rows[b].append(m-baseline[b])
    return {b:np.einsum('nri,nrj->ij',x,x)/MC for b,rr in rows.items() for x in [np.stack(rr)]}

def one(path):
    path=Path(path);root=path.parent.parent;d=read(path);fpath=root/'frozen'/path.name;f=read(fpath);assert sha(fpath)==d['frozen_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256'];links=T.links(d)+T.links(f)+2;e=Experiment(d['system'],d['seed'],d['profile']);assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256']
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h;links+=1
    for v,y in e.raw.items():T.temporal(y,d['model'][v],d['fit_info'][v])
    e.model=d['model'];reconstructed,_=Pilot.reconstruct(e.raw,e.axes,e.model);saved={v:a['anchor_pilot_'+v] for v in e.raw};pilot_errors=[]
    for v,y in saved.items():
        assert R.digest(y)==d['pilot_sha256'][v];err=float(np.max(np.abs(reconstructed[v]-y))/max(np.max(np.abs(y)),np.finfo(float).tiny));assert err<=1e-9,('pilot',path,err);pilot_errors.append(err)
    e.pilot=saved;observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);clean=e.matrices(e.p.data,correct=False);cov=direct_metric(e,e.pilot,e.model);scores=regions=certificates=matrix_replays=refit_replays=0
    def eigen_score(g,c):
        ss=c[np.ix_(e.active,e.active)];gg=g[np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(ss));ss=ss/sc[:,None]/sc[None,:];gg=gg/sc[:,None]/sc[None,:];return float(eigvalsh((gg+gg.T)/2,(ss+ss.T)/2)[-1])
    for b in e.banks:
        for tag,m in [('observed',observed[b]),('pilot',pilot[b]),('clean',clean[b])]:close(m,a[b+'_'+tag]);matrix_replays+=1
        close(cov[b],a[b+'_anchor_covariance']);close(S.gram(observed[b]-clean[b]),a[b+'_error_gram']);close(eigen_score(a[b+'_error_gram'],a[b+'_anchor_covariance']),d['actual_scores'][b]);matrix_replays+=2
        for mode in ('fixed_pilot','refit_pilot'):
            qq=[eigen_score(g,c) for g,c in zip(a[b+'_bootstrap_grams'],a[mode+'_'+b+'_covariances'])];np.testing.assert_allclose(qq,a[mode+'_'+b+'_scores'],rtol=3e-7,atol=3e-9);scores+=len(qq);cut=cutoff(a[mode+'_'+b+'_scores']);assert cut==d['cuts'][mode][b];assert d['coverage'][mode][b]==(d['actual_scores'][b]<=cut);g=S.gram(observed[b]);sigma=a[b+'_anchor_covariance']
            for c,r in zip(e.cs,d['regions'][mode][b]):Q.region_check(r,g-cut*sigma,g,sigma,c,cut);regions+=1
            for r in d['references']:
                if (r['mode'],r['bank'])!=(mode,b):continue
                c=next(c for c in e.cs if c['id']==r['candidate']);v=S.E.vector(c,c['exact'],len(e.labels));assert r['noise_covered']==bool(v@a[b+'_error_gram']@v<=cut*float(v@sigma@v));assert r['zero_accepted']==bool(v@g@v<=cut*float(v@sigma@v))
    for mode in ('fixed_pilot','refit_pilot'):
        cut=cutoff(np.maximum(a[mode+'_local_scores'],a[mode+'_broad_scores']));assert cut==d['joint'][mode]['cutoff'];assert d['coverage'][mode]['both']==(max(d['actual_scores'].values())<=cut)
        for c,cert in zip(e.cs,d['joint'][mode]['certificates']):
            replay=B.C.J.certificate([S.gram(observed[b]) for b in e.banks],[a[b+'_anchor_covariance'] for b in e.banks],cut,c,e.seed);assert replay['geometry']==cert['geometry'];Q.close(replay['lambda_local'],cert['lambda_local']);certificates+=1
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]));assert len(d['refits'])==d['bootstrap'];field_hashes=0
    for j in range(d['bootstrap']):
        fields,_=e.draw(rng);assert {v:R.digest(y) for v,y in fields.items()}==d['bootstrap_field_sha256'][j];field_hashes+=len(fields)
        if j not in (0,d['bootstrap']-1):continue
        model=d['refits'][j]['model'];replayed,info=e.estimate(fields)
        for v,m in model.items():np.testing.assert_allclose(m[0],replayed[v][0],rtol=1e-8,atol=1e-12);np.testing.assert_allclose(m[1],replayed[v][1],rtol=1e-8,atol=1e-10);refit_replays+=1
        pp,_=Pilot.reconstruct(fields,e.axes,model);saved={v:a[f'bootstrap_pilot_{j}_'+v] for v in e.raw}
        for v,y in saved.items():
            assert R.digest(y)==d['refitted_pilot_sha256'][j][v];err=float(np.max(np.abs(pp[v]-y))/max(np.max(np.abs(y)),np.finfo(float).tiny));assert err<=1e-9,('refitted_pilot',path,j,err);pilot_errors.append(err)
        mat=e.matrices(fields,model);cc={'fixed_pilot':direct_metric(e,e.pilot,model),'refit_pilot':direct_metric(e,saved,model)}
        for b in e.banks:
            close(S.gram(mat[b]-pilot[b]),a[b+'_bootstrap_grams'][j]);matrix_replays+=1
            for mode in cc:close(cc[mode][b],a[mode+'_'+b+'_covariances'][j]);matrix_replays+=1
    warnings=[('observed',-1,v,i['status']) for v,i in d['fit_info'].items() if i['status']!='ok']
    warnings += [('bootstrap',j,v,i['status']) for j,r in enumerate(d['refits']) for v,i in r['info'].items() if i['status']!='ok'];assert sorted(warnings)==sorted((r['stream'],r['draw'],r['variable'],r['status']) for r in d['warnings'])
    return dict(status='PASS',record=path.stem,operator_scores=scores,regions=regions,certificates=certificates,matrix_replays=matrix_replays,refit_replays=refit_replays,field_hashes=field_hashes,hash_links=links,max_pilot_relative_difference=max(pilot_errors),result_sha256=sha(path))

def safe(path):
    try:r=one(path)
    except Exception:r=dict(status='FAIL',record=Path(path).stem,error=traceback.format_exc())
    root=Path(path).parent.parent;dump(root/'audit_records'/(r['record']+'.json'),r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--workers',type=int,default=1);args=ap.parse_args();root=ROOT/'preflight_fast' if args.preflight else ROOT;expected=4 if args.preflight else 69;pending={};done=set();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for path in sorted((root/'results').glob('*.json')):
                if path.stem not in done and len(pending)<args.workers:pending[pool.submit(safe,str(path))]=path;done.add(path.stem)
            for f in list(pending):
                if f.done():r=f.result();rows.append(r);pending.pop(f);print(r['record'],r['status'],flush=True)
            if len(done)==expected and not pending:break
            if not args.preflight and (ROOT/'processes_fast.json').exists() and not pending:break
            time.sleep(1)
    out=dict(status='PASS' if len(rows)==expected and all(r['status']=='PASS' for r in rows) else 'FAIL',records=len(rows),expected=expected,totals={k:sum(r.get(k,0) for r in rows) for k in ('operator_scores','regions','certificates','matrix_replays','refit_replays','field_hashes','hash_links')},rows=rows,source_sha256=sources('audit_full_fast.py','full_pipeline_fast.py','pilot_roundoff_protocol.txt'))
    dump(root/'audit.json',out);assert out['status']=='PASS',out
if __name__=='__main__':main()
