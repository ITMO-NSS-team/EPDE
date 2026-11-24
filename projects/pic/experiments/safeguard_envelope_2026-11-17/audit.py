"""Independent Python likelihood and generalized-eigenvalue replay."""
from pipeline import *
import audit_new as A
import native_score as Native
from types import MethodType
from scipy.optimize import linprog
import traceback
Native.restore()


def arrays(path,meta):
    path=Path(path);assert sha(path)==meta['array_sha256']
    with np.load(path) as z:a={k:z[k] for k in z.files}
    for k,h in meta.get('array_hashes',{}).items():assert R.digest(a[k])==h
    return a


def one(key):
    A.V.proofs.clear();e,d,anchor=load_case(key)
    ap=ROOT/'anchors'/(key+'.json');arrays(ap.with_suffix('.npz'),d)
    pp=ROOT/'point_frozen'/(key+'.json');p=read(pp);a=arrays(pp.with_suffix('.npz'),p)
    rp=ROOT/'results'/(key+'.json');r=read(rp)
    ep=ROOT/'envelope_frozen'/(key+'.json');env=read(ep)
    e.matrices=MethodType(lambda obj,fields,model=None,correct=True:moments(obj,fields,model,correct),e)
    assert F.State.axis_score is Native.ORIGINAL
    assert {v:R.digest(y) for v,y in e.pilot.items()}==d['pilot_sha256']
    reconstructed,_=Pilot.reconstruct(e.raw,e.axes,e.model)
    for v,y in e.raw.items():
        A.exactfit(y,e.model[v],d['fit_info'][v]);A.close(reconstructed[v],e.pilot[v])
    obs=moments(e,e.raw);pilot=moments(e,e.pilot,correct=False);clean=moments(e,e.p.data,correct=False)
    cov=A.V.Legacy.direct_metric(e,e.pilot,e.model);scores=regions=certs=fitcount=conditional=proofcount=0
    actual={b:A.eigen_score(S.gram(obs[b]-clean[b]),cov[b],e.active) for b in e.banks}
    for b in e.banks:
        A.close(obs[b],a[b+'_observed']);A.close(pilot[b],a[b+'_pilot']);A.close(cov[b],a[b+'_covariance']);A.close(actual[b],r['actual_scores'][b])
        for mode in ('studentized','fixed_metric'):
            cc=a[b+'_bootstrap_covariances'] if mode=='studentized' else [a[b+'_covariance']]*128
            qs=[A.eigen_score(g,c,e.active) for g,c in zip(a[b+'_bootstrap_grams'],cc)]
            A.close(qs,a[mode+'_'+b+'_scores']);scores+=128
            cut=sorted(a[mode+'_'+b+'_scores'])[122];assert cut==p['cuts'][mode][b]
            assert r['coverage'][mode][b]==(actual[b]<=cut)
            g=S.gram(obs[b])
            for c,reg in zip(e.cs,p['regions'][mode][b]):A.V.verified(reg,g-cut*cov[b],g,cov[b],c,cut);regions+=1
    for mode,item in p['joint'].items():
        cut=sorted(np.maximum(*[a[mode+'_'+b+'_scores'] for b in e.banks]))[122]
        assert cut==item['cutoff'];assert r['coverage'][mode]['joint']==(max(actual.values())<=cut)
        for c,cert in zip(e.cs,item['certificates']):A.certcheck(cert,[S.gram(obs[b]) for b in e.banks],[cov[b] for b in e.banks],cut,c);certs+=1
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]))
    assert len(p['refits'])==128
    for j,row in enumerate(p['refits']):
        fields,_=e.draw(rng);assert row['draw']==j;assert {v:R.digest(y) for v,y in fields.items()}==row['field_sha256']
        for v,y in fields.items():A.exactfit(y,row['model'][v],row['info'][v]);fitcount+=1
        if j in (0,127):
            mat=moments(e,fields,row['model']);cv=A.V.Legacy.direct_metric(e,e.pilot,row['model'])
            for b in e.banks:A.close(S.gram(mat[b]-pilot[b]),a[b+'_bootstrap_grams'][j]);A.close(cv[b],a[b+'_bootstrap_covariances'][j])
    for start in range(0,128,8):
        path=ROOT/'chunks'/f'{key}_start{start:03d}.json';chunk=read(path);aa=arrays(path.with_suffix('.npz'),chunk)
        assert chunk['rows']==p['refits'][start:start+8]
        for b in e.banks:
            assert np.array_equal(aa[b+'_grams'],a[b+'_bootstrap_grams'][start:start+8])
            assert np.array_equal(aa[b+'_covariances'],a[b+'_bootstrap_covariances'][start:start+8])
    points=read(ROOT/'nuisance_points'/(key+'.json'))['points']
    theta=np.array([np.log(e.model['u'][0]),np.arctanh(e.model['u'][1][0])])
    A.close(points[0]['transformed'],theta)
    for item,fit in zip(points[1:],p['refits']):
        A.close(item['transformed'],2*theta-np.array([np.log(fit['model']['u'][0]),np.arctanh(fit['model']['u'][1][0])]))
    seen=[];bounds=[];bad=[]
    worst=max((row for row in env['points'] if row['status']=='ok'),key=lambda row:max(v['bound'] for v in row['bounds'].values()))['index']
    for start in range(0,129,4):
        path=ROOT/'envelope_chunks'/f'{key}_start{start:03d}.json';chunk=read(path);aa=arrays(path.with_suffix('.npz'),chunk)
        for row in chunk['rows']:
            idx=row['index'];seen.append(idx);assert row==env['points'][idx]
            if row['status']!='ok':bad.append(idx);continue
            pos=list(aa['indices']).index(idx);item=points[idx];model=item['model'];assert model==row['model']
            A.close(model['u'][0],np.exp(item['transformed'][0]));A.close(model['u'][1][0],np.tanh(item['transformed'][1]))
            mm=moments(e,e.raw,model);qq={}
            for b in e.banks:
                cv=aa[b+'_covariance'][pos];gg=aa[b+'_grams'][pos]
                qq[b]=[A.eigen_score(g,cv,e.active) for g in gg];A.close(qq[b],aa[b+'_scores'][pos]);conditional+=len(qq[b])
                A.close(S.gram(obs[b]-mm[b]),aa[b+'_correction_gram'][pos])
            A.close(sorted(np.maximum(qq['local'],qq['broad']))[125],row['conditional_cutoff'])
            for b in e.banks:
                proof=row['bounds'][b];tau=proof['tau'];assert np.exp(-10)<=tau<=np.exp(10)
                h=(1+tau)*aa[b+'_correction_gram'][pos]+(1+1/tau)*row['conditional_cutoff']*aa[b+'_covariance'][pos]
                bound=A.eigen_score(h,a[b+'_covariance'],e.active);A.close(bound,proof['bound']);bounds.append(proof['bound']);proofcount+=1
            if idx in (0,1,128,worst):
                cv=A.V.Legacy.direct_metric(e,e.pilot,model)
                for b in e.banks:A.close(cv[b],aa[b+'_covariance'][pos])
                for hi,j in enumerate((0,127)):
                    rngj=np.random.default_rng(np.random.SeedSequence([985802,e.system_id,e.pid,e.seed,j]));fields={v:y+R.WF.ar_noise(rngj,y.shape,*model[v][:2]) for v,y in e.pilot.items()}
                    assert {v:R.digest(y) for v,y in fields.items()}==row['field_hashes'][hi]
                    for b,m in moments(e,fields,model).items():A.close(S.gram(m-pilot[b]),aa[b+'_grams'][pos,j])
    assert seen==list(range(129));assert bad==env['unresolved_points']
    if not bad:
        cut=max(bounds+[item['cutoff'] for item in p['joint'].values()]);assert cut==env['cutoff']
        for b in e.banks:assert r['coverage']['envelope'][b]==(actual[b]<=cut)
        assert r['coverage']['envelope']['joint']==(max(actual.values())<=cut)
        for c,cert in zip(e.cs,env['certificates']):A.certcheck(cert,[S.gram(obs[b]) for b in e.banks],[cov[b] for b in e.banks],cut,c);certs+=1
    else:assert env['status']=='UNRESOLVED' and r['coverage']['envelope'] is None
    for row in r['references']:
        b=row['bank'];c=next(c for c in e.cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(e.labels))
        cut=env['cutoff'] if row['mode']=='envelope' else p['cuts'][row['mode']][b];bound=cut*float(v@cov[b]@v)
        assert row['noise_covered']==(v@S.gram(obs[b]-clean[b])@v<=bound);assert row['zero_accepted']==(v@S.gram(obs[b])@v<=bound)
    return dict(status='PASS',key=key,original_python_fits=fitcount+len(e.raw),bootstrap_field_hashes=128,primary_scores=scores,conditional_scores=conditional,regions=regions,certificates=certs,loewner_bounds=proofcount,high_precision_checks=list(A.V.proofs),source_sha256=src('audit.py'),input_sha256={str(rp):sha(rp),str(ep):sha(ep)})


def safe(key):
    try:r=one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    path=ROOT/'audits'/(key+'.json')
    if path.exists() and read(path)['status']=='FAIL':dump(path.with_name(key+'_previous_failure.json'),read(path))
    dump(path,r);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();assert read(ROOT/'evaluation_process.json')['status']=='COMPLETE'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[]
        for f in as_completed([pool.submit(safe,key_of(*j)) for j in JOBS]):
            r=f.result();rows.append(r);print('audit',r['key'],r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) and len(rows)==12 else 'FAIL',rows=rows,source_sha256=src('audit.py'))
    dump(ROOT/'audit.json',out);assert out['status']=='PASS'


if __name__=='__main__':main()
