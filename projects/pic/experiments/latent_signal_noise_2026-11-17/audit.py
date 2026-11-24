from pathlib import Path
import sys,json,importlib.util,traceback
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('spectral_coverage_audit',ROOT/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R,np=C.S,C.R,C.np
spec=importlib.util.spec_from_file_location('previous_region_checks',S.ROOT/'audit_joint.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache

@lru_cache(None)
def sha(p):return S.E.sha(p)

def links(d):
    count=0
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);count+=1
    return count

def one(path):
    path=Path(path);d=json.loads(path.read_text());counts=dict(hash_links=links(d),operator_scores=0,registered_scores=0,regions=0,intersection_certificates=0,matrix_replays=0)
    e=C.Study(d['system'],d['seed'],d['profile'],d['model_kind']);assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256']
    for v,m in e.model.items():
        if m is None:assert d['model'][v] is None;continue
        np.testing.assert_array_equal(m[0],d['model'][v][0]);np.testing.assert_array_equal(m[1],d['model'][v][1])
    if d['status']!='COMPLETE':
        assert not e.valid and d['status']=='incomplete_noise_model'
        return dict(record=path.stem,status='PASS',incomplete=True,**counts)
    assert e.valid;fp=ROOT/'frozen'/path.name;f=json.loads(fp.read_text());counts['hash_links']+=links(f)
    assert sha(fp)==d['frozen_sha256'] and sha(path.with_suffix('.npz'))==d['array_sha256'];counts['hash_links']+=2
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in a.values())
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h;counts['hash_links']+=1
    assert {v:R.digest(y) for v,y in e.pilot.items()}==f['pilot_sha256']
    for bank in e.banks:
        m=d['metrics'][bank];assert m['status']=='ok';cut=m['cutoff'];g=a[bank+'_observed_gram'];s=a[bank+'_sigma'];aa=g-cut*s
        np.testing.assert_array_equal(g,S.gram(a[bank+'_observed']));np.testing.assert_array_equal(s,a[bank+'_calibration_grams'].mean(axis=0))
        sub=s[np.ix_(e.active,e.active)];scale=np.sqrt(np.diag(sub));sub=sub/scale[:,None]/scale[None,:]
        for tag in ('check','actual_error'):
            gg=a[bank+'_'+tag+'_grams'];hh=gg[:,e.active,:][:,:,e.active]/scale[None,:,None]/scale[None,None,:]
            direct=np.array([eigvalsh((h+h.T)/2,sub)[-1] for h in hh]);stored=a[bank+('_check_operator_scores' if tag=='check' else '_actual_operator_scores')]
            np.testing.assert_allclose(direct,stored,rtol=3e-8,atol=3e-10);counts['operator_scores']+=len(direct)
            assert np.all(gg[:,e.deterministic,:]==0) and np.all(gg[:,:,e.deterministic]==0)
        assert cut==np.quantile(a[bank+'_check_operator_scores'],.95)
        for c,r in zip(e.cs,d['regions'][bank]):assert c['id']==r['candidate'];A.region_check(r,aa,g,s,c,cut);counts['regions']+=1
        for row in [r for r in d['statistics'] if r['bank']==bank]:
            c=next(c for c in e.cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(g));check=np.einsum('i,bij,j->b',v,a[bank+'_check_grams'],v)
            actual=np.einsum('i,bij,j->b',v,a[bank+'_actual_error_grams'],v);zero=np.einsum('i,bij,j->b',v,a[bank+'_actual_grams'],v)
            threshold=np.quantile(check,.95);bound=cut*float(v@s@v)
            assert row['pointwise_noise_coverage_count']==np.sum(actual<=threshold);assert row['uniform_noise_coverage_count']==np.sum(actual<=bound)
            assert row['pointwise_zero_acceptance_count']==np.sum(zero<=threshold);assert row['uniform_zero_acceptance_count']==np.sum(zero<=bound)
            assert row['operator_coverage_count']==np.sum(a[bank+'_actual_operator_scores']<=cut);assert row['anchor_exact_accepted']==bool(v@g@v<=bound)
            A.close(threshold,row['directional_radius']);A.close(bound,row['uniform_bound']);A.close(float(np.sum((a[bank+'_clean']@v)**2)),row['clean_residual']);counts['registered_scores']+=770
    joint=d['joint'];cut=joint['cutoff'];assert cut==np.quantile(np.maximum(a['local_check_operator_scores'],a['broad_check_operator_scores']),.95)
    assert joint['operator_coverage_count']==np.sum(np.maximum(a['local_actual_operator_scores'],a['broad_actual_operator_scores'])<=cut)
    gg=[a[b+'_observed_gram'] for b in e.banks];ss=[a[b+'_sigma'] for b in e.banks]
    for c,cert in zip(e.cs,joint['certificates']):
        replay=C.J.certificate(gg,ss,cut,c,e.seed);assert cert['geometry']==replay['geometry'];A.close(cert['lambda_local'],replay['lambda_local'])
        norm=cert['target_normalizers'];lam=cert['lambda_local'];g=lam*(gg[0]/norm[0])+(1-lam)*(gg[1]/norm[1]);s=lam*(ss[0]/norm[0])+(1-lam)*(ss[1]/norm[1])
        A.region_check(cert['outer_region'],g-cut*s,g,s,c,cut)
        if cert.get('feasible_witness') is not None:
            v=S.E.vector(c,cert['feasible_witness'],len(g));margins=[float(v@((x-cut*y)/n)@v) for x,y,n in zip(gg,ss,norm)];assert max(margins)<=0
        if cert['geometry']=='unbounded_certified':
            direction=np.asarray(cert['common_negative_direction']);assert all(direction@((x-cut*y)/n)[np.ix_(c['xi'],c['xi'])]@direction<0 for x,y,n in zip(gg,ss,norm))
        counts['intersection_certificates']+=1
    for row in joint['reference_statistics']:
        c=next(c for c in e.cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(gg[0]));accepted=[];anchor=[]
        for bank,g,s in zip(e.banks,gg,ss):
            bound=cut*float(v@s@v);accepted.append(np.einsum('i,bij,j->b',v,a[bank+'_actual_grams'],v)<=bound);anchor.append(float(v@g@v)<=bound)
        assert row['anchor_accepted']==all(anchor);assert row['actual_acceptance_count']==np.sum(accepted[0]&accepted[1])
    for tag,field,correct in [('observed',e.raw,True),('pilot',e.pilot,False),('clean',e.p.data,False)]:
        for bank,m in e.matrices(field,correct).items():np.testing.assert_array_equal(m,a[bank+'_'+tag]);counts['matrix_replays']+=1
    for tag,domain in [('calibration',960102),('check',960103)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.profile_id,e.seed]));hh=[]
        for b in range(256):
            noise={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
            if b in (0,255):
                hh.append({v:R.digest(y) for v,y in noise.items()})
                for bank,m in e.matrices({v:e.pilot[v]+y for v,y in noise.items()}).items():
                    np.testing.assert_array_equal(S.gram(m-a[bank+'_pilot']),a[bank+'_'+tag+'_grams'][b]);counts['matrix_replays']+=1
        assert hh==f['innovation_hashes'][tag]
    for j,b in enumerate((0,255)):
        fields=e.true_field(b);assert {v:R.digest(y) for v,y in fields.items()}==d['actual_field_hashes'][j]
        for bank,m in e.matrices(fields).items():
            np.testing.assert_array_equal(S.gram(m),a[bank+'_actual_grams'][b]);np.testing.assert_array_equal(S.gram(m-a[bank+'_clean']),a[bank+'_actual_error_grams'][b]);counts['matrix_replays']+=1
    return dict(record=path.stem,status='PASS',incomplete=False,**counts)

def safe_one(path):
    try:return one(path)
    except Exception:return dict(record=Path(path).stem,status='FAIL',error=traceback.format_exc())

def main():
    process=json.loads((ROOT/'processes.json').read_text())['rows'];assert len(process)==138 and all(r['exit_code']==0 for r in process)
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==138
    expected={f"{r['system']}_{r['profile']}_{r['model']}_seed{r['seed']}" for r in process};assert {p.stem for p in files}==expected
    paired={}
    for p in files:
        d=json.loads(p.read_text());key=(d['system'],d['profile'],d['seed']);hashes=(d['raw_sha256'],d.get('actual_field_hashes'))
        if key in paired:
            assert paired[key][0]==hashes[0]
            if paired[key][1] and hashes[1]:assert paired[key][1]==hashes[1]
        else:paired[key]=hashes
    rows=[]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(safe_one,str(p)) for p in files]):
            row=f.result();rows.append(row);print(row['record'],row['status'],flush=True)
    failed=[r for r in rows if r['status']!='PASS']
    if failed:R.dump(ROOT/'audit_failed.json',dict(rows=rows));raise AssertionError(f'{len(failed)} audit failures')
    totals={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('record','status','incomplete')}
    out=dict(status='PASS',records=138,incomplete=sum(r['incomplete'] for r in rows),paired_fields=True,totals=totals,rows=rows,
        source_sha256={str(Path(__file__)):S.E.sha(__file__),str(S.ROOT/'audit_joint.py'):S.E.sha(S.ROOT/'audit_joint.py')})
    R.dump(ROOT/'audit.json',out);print(totals,flush=True)

if __name__=='__main__':main()
