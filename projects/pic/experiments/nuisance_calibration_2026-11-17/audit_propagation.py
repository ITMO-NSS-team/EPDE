from propagate import *
from types import MethodType
import traceback


def one(key):
    e,d=getcase(key);p,a=primary_arrays(key);f=read(ROOT/'propagation_frozen'/(key+'.json'));result=read(ROOT/'propagation_results'/(key+'.json'));ref=next(c for c in e.cs if c['label']=='reference');e.matrices=MethodType(lambda obj,fields,model=None,correct=True:Old.moments(obj,fields,model,correct),e);cv=A.V.Legacy.direct_metric(e,e.pilot,e.model);obs=e.matrices(e.raw);clean=e.matrices(e.p.data,correct=False);actual={};scores=0;certificates=0;bounds=[];covering=[];best=float('inf');accepted=False;v=S.E.vector(ref,ref['exact'],len(e.labels))
    for b in e.banks:
        A.close(cv[b],a[b+'_covariance']);A.close(obs[b],a[b+'_observed']);actual[b]=A.eigen_score(S.gram(obs[b]-clean[b]),cv[b],e.active)
        for mode in p['cuts']:
            ccc=a[b+'_bootstrap_covariances'] if mode=='studentized' else [a[b+'_covariance']]*128;qs=[A.eigen_score(g,s,e.active) for g,s in zip(a[b+'_bootstrap_grams'],ccc)];A.close(qs,a[mode+'_'+b+'_scores']);scores+=128
    for mode,cut in p['cuts'].items():
        assert sorted(np.maximum(a[mode+'_local_scores'],a[mode+'_broad_scores']))[122]==cut;assert result['coverage'][mode]==(max(actual.values())<=cut)
        for c,cert in zip(e.cs,p['certificates'][mode]):A.certcheck(cert,[S.gram(obs[b]) for b in e.banks],[cv[b] for b in e.banks],cut,c);certificates+=1
        accepted=accepted or all(v@S.gram(obs[b])@v<=cut*float(v@cv[b]@v) for b in e.banks)
    ellipse=read(ROOT/'bootstrap_frozen'/(key+'.json'));center=np.array(ellipse['basic_center']);covellipse=np.array(ellipse['covariance'])
    for j,item in enumerate(p['points']):
        t=np.array(item['transformed']);A.close(np.log(item['model']['u'][0]),t[0]);A.close(np.arctanh(item['model']['u'][1][0]),t[1])
        if j==0:A.close(t,center)
        elif j==1:A.close(t,ellipse['theta'])
        else:
            delta=t-center;A.close(float(delta@np.linalg.solve(covellipse,delta)),ellipse['empirical_cutoff']*(.25 if j<34 else 1.))
    seen=[]
    for start in range(0,66,3):
        path=ROOT/'propagation_chunks'/f'{key}_start{start:03d}.npz';meta=read(path.with_suffix('.json'));assert sha(path)==meta['array_sha256']
        with np.load(path) as z:
            for row in meta['rows']:
                idx=row['index'];seen.append(idx);assert row==f['points'][idx]
                if row['status']!='ok':continue
                pos=list(z['indices']).index(idx);model=row['model'];direct=e.matrices(e.raw,model);qq={};ratio=[]
                for b in e.banks:
                    A.close(direct[b],z[b+'_observed'][pos]);qq[b]=[A.eigen_score(g,z[b+'_covariance'][pos],e.active) for g in z[b+'_grams'][pos]];A.close(qq[b],z[b+'_scores'][pos]);scores+=128;A.close(S.gram(obs[b]-direct[b]),z[b+'_correction_gram'][pos]);proof=row['bounds'][b];tau=proof['tau'];h=(1+tau)*z[b+'_correction_gram'][pos]+(1+1/tau)*row['conditional_cutoff']*z[b+'_covariance'][pos];A.close(A.eigen_score(h,a[b+'_covariance'],e.active),proof['bound']);bounds.append(proof['bound']);ratio.append(A.eigen_score(S.gram(direct[b]-clean[b]),z[b+'_covariance'][pos],e.active)/row['conditional_cutoff'])
                A.close(sorted(np.maximum(qq['local'],qq['broad']))[125],row['conditional_cutoff']);A.certcheck(row['reference_certificate'],[S.gram(direct[b]) for b in e.banks],[z[b+'_covariance'][pos] for b in e.banks],row['conditional_cutoff'],ref);certificates+=1
                best=min(best,max(ratio))
                if max(ratio)<=1:covering.append(idx)
                accepted=accepted or all(v@S.gram(direct[b])@v<=row['conditional_cutoff']*float(v@z[b+'_covariance'][pos]@v) for b in e.banks)
                if idx in (0,1,34,65):
                    cc=A.V.Legacy.direct_metric(e,e.pilot,model)
                    for b in e.banks:A.close(cc[b],z[b+'_covariance'][pos])
                    for hi,j in enumerate((0,127)):
                        rng=np.random.default_rng(np.random.SeedSequence([985802,e.system_id,e.pid,e.seed,j]));field={vv:yy+R.WF.ar_noise(rng,yy.shape,*model[vv][:2]) for vv,yy in e.pilot.items()};assert {vv:R.digest(yy) for vv,yy in field.items()}==row['field_hashes'][hi]
                        for b,mm in e.matrices(field,model).items():A.close(S.gram(mm-a[b+'_pilot']),z[b+'_grams'][pos,j])
    assert seen==list(range(66));assert max(bounds+list(p['cuts'].values()))==f['scalar_cutoff'];A.certcheck(f['scalar_certificate'],[S.gram(obs[b]) for b in e.banks],[cv[b] for b in e.banks],f['scalar_cutoff'],ref);certificates+=1
    assert result['coverage']['scalar']==(max(actual.values())<=f['scalar_cutoff']);assert result['coverage']['finite_union']==(bool(covering) or any(result['coverage'][m] for m in p['cuts']));assert result['covering_conditional_points']==covering;A.close(best,result['best_conditional_ratio']);assert bool(accepted)==result['reference_accepted']
    if f['geometry']=='bounded_outer':
        assert all(c['geometry']=='bounded_outer' for c in f['components']);A.close(np.min([c['outer_region']['lower'] for c in f['components']],axis=0),f['outer_box']['lower']);A.close(np.max([c['outer_region']['upper'] for c in f['components']],axis=0),f['outer_box']['upper'])
    elif f['geometry']=='unbounded_certified':assert any(c['geometry']=='unbounded_certified' for c in f['components'])
    return dict(status='PASS',key=key,generalized_scores=scores,certificates=certificates,loewner_bounds=len(bounds),source_sha256=psrc(),input_sha256={str(ROOT/'propagation_results'/(key+'.json')):sha(ROOT/'propagation_results'/(key+'.json'))})


def safe(key):
    try:r=one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    dump(ROOT/'propagation_audits'/(key+'.json'),r);return r


def main():
    assert read(ROOT/'propagation_process.json')['status']=='COMPLETE';rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(safe,bkey(*j)) for j in JOBS]):r=f.result();rows.append(r);print('propagation audit',r['key'],r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if len(rows)==24 and all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_propagation.py','propagate.py','propagation_protocol.txt'));dump(ROOT/'propagation_audit.json',out);assert out['status']=='PASS'


if __name__=='__main__':main()
