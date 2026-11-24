from common import *
import paired_study as E
from scipy.linalg import eigvalsh
import importlib.util,time,argparse,traceback
from concurrent.futures import ProcessPoolExecutor
spec=importlib.util.spec_from_file_location('region_checks',S.ROOT/'audit_joint.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
def close(x,y):
    scale=max(float(np.max(np.abs(y))),np.finfo(float).tiny);np.testing.assert_allclose(x,y,rtol=5e-10,atol=5e-12*scale)
def checklinks(d):
    n=0
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);n+=1
    return n
def one(path):
    path=Path(path);d=read(path);e=E.Study(d['system'],d['seed'],d['profile']);fp=ROOT/'frozen_fast'/path.name;f=read(fp)
    assert d['status'] in ('COMPLETE','COMPLETE_WITH_FIT_WARNINGS');assert sha(fp)==d['frozen_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256']
    counts=dict(hash_links=checklinks(d)+checklinks(f)+2,operator_scores=0,regions=0,certificates=0,matrix_replays=0,refit_replays=0)
    assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256'];assert {v:R.digest(y) for v,y in e.pilot.items()}==f['pilot_sha256']
    for v,m in e.model.items():np.testing.assert_array_equal(m[0],d['model'][v][0]);np.testing.assert_array_equal(m[1],d['model'][v][1])
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    assert all(np.isfinite(x).all() for x in a.values())
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h;counts['hash_links']+=1
    for mode in ('fixed','refit'):
        for bank in e.banks:
            pre=f'{mode}_{bank}';sigma=a[pre+'_sigma'];g=a[pre+'_observed_gram'];cut=d['metrics'][mode][bank]['cutoff'];np.testing.assert_array_equal(sigma,a[pre+'_calibration_grams'].mean(axis=0));np.testing.assert_array_equal(g,S.gram(a[bank+'_observed']))
            sub=sigma[np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(sub));sub=sub/sc[:,None]/sc[None,:]
            for tag,out in [('check','check'),('actual_error','actual')]:
                gg=a[pre+'_'+tag+'_grams'];small=gg[:,e.active,:][:,:,e.active]/sc[None,:,None]/sc[None,None,:];direct=np.array([eigvalsh((x+x.T)/2,sub)[-1] for x in small]);np.testing.assert_allclose(direct,a[pre+'_'+out+'_scores'],rtol=3e-7,atol=3e-9);counts['operator_scores']+=len(direct)
                assert np.all(gg[:,e.deterministic,:]==0) and np.all(gg[:,:,e.deterministic]==0)
            assert cut==np.quantile(a[pre+'_check_scores'],.95)
            for c,r in zip(e.cs,d['regions'][mode][bank]):Q.region_check(r,g-cut*sigma,g,sigma,c,cut);counts['regions']+=1
            stat=next(r for r in d['statistics'] if (r['mode'],r['bank'])==(mode,bank));assert stat['operator_coverage_count']==np.sum(a[pre+'_actual_scores']<=cut)
            for row in stat['references']:
                c=next(c for c in e.cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(e.labels));bound=cut*float(v@sigma@v);noise=np.einsum('i,bij,j->b',v,a[pre+'_actual_error_grams'],v);check=np.einsum('i,bij,j->b',v,a[pre+'_check_grams'],v)
                assert row['anchor_accepted']==bool(v@g@v<=bound);assert row['pointwise_coverage_count']==np.sum(noise<=np.quantile(check,.95));assert row['uniform_coverage_count']==np.sum(noise<=bound)
                assert row['zero_acceptance_count']==np.sum(np.einsum('i,bij,j->b',v,a[pre+'_actual_grams'],v)<=bound)
        joint=d['joint'][mode];assert joint['cutoff']==np.quantile(np.maximum(a[f'{mode}_local_check_scores'],a[f'{mode}_broad_check_scores']),.95)
        assert joint['operator_coverage_count']==np.sum(np.maximum(a[f'{mode}_local_actual_scores'],a[f'{mode}_broad_actual_scores'])<=joint['cutoff'])
        gg=[a[f'{mode}_{b}_observed_gram'] for b in e.banks];ss=[a[f'{mode}_{b}_sigma'] for b in e.banks]
        for c,cert in zip(e.cs,joint['certificates']):
            replay=C.J.certificate(gg,ss,joint['cutoff'],c,e.seed);assert cert['geometry']==replay['geometry'];Q.close(cert['lambda_local'],replay['lambda_local']);counts['certificates']+=1
    for tag,fields,correct in [('observed',e.raw,True),('pilot',e.pilot,False),('clean',e.p.data,False)]:
        for b,m in e.matrices(fields,correct=correct).items():close(m,a[b+'_'+tag]);counts['matrix_replays']+=1
    for stream,domain in [('calibration',980102),('check',980103),('actual',980104)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.pid,e.seed]));hashes=[]
        assert len(d['refits'][stream])==256
        for j in range(256):
            fields=e.true_field(j) if stream=='actual' else e.draw(rng)[0]
            if j not in (0,255):continue
            hashes.append({v:R.digest(y) for v,y in fields.items()});model,info=e.estimate(fields)
            for v,m in model.items():
                saved=d['refits'][stream][j]['model'][v];np.testing.assert_array_equal(m[0],saved[0]);np.testing.assert_array_equal(m[1],saved[1]);assert info[v]['status']==d['refits'][stream][j]['info'][v]['status'];counts['refit_replays']+=1
            for mode,m in [('fixed',e.model),('refit',model)]:
                for b,matrix in e.matrices(fields,m).items():
                    tag='actual_error' if stream=='actual' else stream;target=a[b+('_clean' if stream=='actual' else '_pilot')]
                    close(S.gram(matrix-target),a[f'{mode}_{b}_{tag}_grams'][j]);counts['matrix_replays']+=1
                    if stream=='actual':close(S.gram(matrix),a[f'{mode}_{b}_actual_grams'][j])
        assert hashes==d['field_hashes'][stream]
    expected=[]
    for v,i in d['fit_info'].items():
        if i['status']!='ok':expected.append(('anchor',-1,v,i['status']))
    for stream,rows in d['refits'].items():
        for j,row in enumerate(rows):
            for v,i in row['info'].items():
                if i['status']!='ok':expected.append((stream,j,v,i['status']))
    assert sorted(expected)==sorted((r['stream'],r['draw'],r['variable'],r['status']) for r in d['warnings'])
    return dict(status='PASS',record=path.stem,warnings=len(expected),result_sha256=sha(path),**counts)
def safe(path):
    try:r=one(path)
    except Exception:r=dict(status='FAIL',record=Path(path).stem,error=traceback.format_exc())
    dump(ROOT/'audit_records_fast'/(r['record']+'.json'),r);return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);args=p.parse_args();pending={};done=set();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for path in sorted((ROOT/'results').glob('*.json')):
                if path.stem not in done and len(pending)<args.workers:pending[pool.submit(safe,str(path))]=path;done.add(path.stem)
            for fut in list(pending):
                if fut.done():r=fut.result();rows.append(r);print(r['record'],r['status'],flush=True);pending.pop(fut)
            if len(done)==69 and not pending:break
            time.sleep(1)
    failures=[r for r in rows if r['status']!='PASS'];totals={k:sum(r.get(k,0) for r in rows) for k in ('hash_links','operator_scores','regions','certificates','matrix_replays','refit_replays')}
    dump(ROOT/'audit_fast.json',dict(status='FAIL' if failures else 'PASS',records=len(rows),totals=totals,rows=rows,source_sha256=hashes('audit_fast.py')));assert not failures,failures
if __name__=='__main__':main()
