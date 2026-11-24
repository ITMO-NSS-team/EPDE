from design_engine import *
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import MethodType
import argparse,traceback
import audit_verified as V

def close(x,y):np.testing.assert_allclose(x,y,rtol=3e-8,atol=1e-10*max(float(np.max(np.abs(y))),1e-300))

def exactfit(y,model,info,nuisance_only=False):
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;sc=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/sc]
    u=np.array(info['transformed']);u=np.r_[np.arctanh(info['rho']),u] if nuisance_only else u;t,jac=F.P.inverse(u,1)
    val,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);g*=jac
    close(val,info['objective']);close(sc*np.sqrt(q/cc.count),model[0]);close(t[0],model[1][0])
    lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND]);pgrad=abs(u-np.clip(u-g,lo,hi));pg=max(pgrad[1:] if nuisance_only else pgrad)
    close(pg,info['projected_gradient']);assert pg<=2e-5 or info['status']!='ok'

def eigen_score(g,c,active):
    ix=np.ix_(active,active);s=c[ix];h=g[ix];sc=np.sqrt(np.diag(s));s=s/sc[:,None]/sc;h=h/sc[:,None]/sc
    return float(eigvalsh((h+h.T)/2,(s+s.T)/2)[-1])

def certcheck(cert,grams,cv,cut,c):
    norms=np.array(cert['target_normalizers']);close(norms,[g[c['yi'],c['yi']]+cut*s[c['yi'],c['yi']] for g,s in zip(grams,cv)])
    lam=cert['lambda_local'];assert 0<=lam<=1;gg=lam*grams[0]/norms[0]+(1-lam)*grams[1]/norms[1];ss=lam*cv[0]/norms[0]+(1-lam)*cv[1]/norms[1]
    V.verified(cert['outer_region'],gg-cut*ss,gg,ss,c,cut);aa=[(g-cut*s)/n for g,s,n in zip(grams,cv,norms)]
    if cert.get('feasible_witness') is not None:
        v=S.E.vector(c,cert['feasible_witness'],len(gg));m=[float(v@a@v) for a in aa];assert max(m)<=0;close(m,cert['witness_margins'])
    if cert['geometry']=='unbounded_certified':
        v=np.array(cert['common_negative_direction']);q=[float(v@a[np.ix_(c['xi'],c['xi'])]@v) for a in aa];assert max(q)<0;close(q,cert['curvatures'])

def result(path):
    path=Path(path);d=read(path);repair=path.parent.name=='repair_results';fp=ROOT/('repair_frozen' if repair else 'frozen')/path.name;f=read(fp)
    assert sha(fp)==d['frozen_sha256'] and sha(path.with_suffix('.npz'))==d['array_sha256']
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h
    e,_,_=primary(d['system'],'iid',51);e.profile=d['profile'];e.seed=d['seed'];e.pid,e.rho=S.PROFILES[e.profile];e.model=d['model'];e.raw=e.true_field(0,True);e.pilot={v:a['anchor_pilot_'+v] for v in e.raw}
    if not repair:
        build(e,d['specs']);e.matrices=MethodType(lambda obj,fields,model=None,correct=True:matrices(obj,fields,model,correct),e)
    assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256'];assert {v:R.digest(y) for v,y in e.pilot.items()}==d['pilot_sha256']
    for v,y in e.raw.items():exactfit(y,e.model[v],d['fit_info'][v])
    reconstructed,_=Pilot.reconstruct(e.raw,e.axes,e.model)
    for v,y in e.pilot.items():assert np.max(abs(reconstructed[v]-y))<=1e-9*max(abs(y).max(),1e-300)
    observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);clean=e.matrices(e.p.data,correct=False);cov=V.Legacy.direct_metric(e,e.pilot,e.model);ns=nr=nc=nf=0
    for b in e.banks:
        for tag,m in [('observed',observed[b]),('pilot',pilot[b]),('clean',clean[b])]:close(m,a[b+'_'+tag])
        close(cov[b],a[b+'_covariance']);close(S.gram(observed[b]-clean[b]),a[b+'_error_gram']);q=[eigen_score(g,c,e.active) for g,c in zip(a[b+'_bootstrap_grams'],a[b+'_bootstrap_covariances'])];close(q,a[b+'_scores']);ns+=len(q)
        actual=eigen_score(a[b+'_error_gram'],a[b+'_covariance'],e.active);close(actual,d['actual_scores'][b]);cut=sorted(a[b+'_scores'])[122];assert cut==d['cuts'][b];assert d['coverage'][b]==(actual<=cut)
        if not repair:
            g=S.gram(observed[b]);s=cov[b]
            for c,r in zip(e.cs,d['regions'][b]):V.verified(r,g-cut*s,g,s,c,cut);nr+=1
    pairs={'joint':dict(d['joint'],banks=['local','broad'])} if repair else d['joint']
    for mode,details in pairs.items():
        bb=details['banks'];cut=sorted(np.maximum(*[a[b+'_scores'] for b in bb]))[122];assert cut==details['cutoff'];assert d['coverage'][mode]==(max(d['actual_scores'][b] for b in bb)<=cut)
        for c,cert in zip(e.cs,details['certificates']):certcheck(cert,[S.gram(observed[b]) for b in bb],[a[b+'_covariance'] for b in bb],cut,c);nc+=1
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]))
    for j,row in enumerate(d['refits']):
        fields,_=e.draw(rng);assert {v:R.digest(y) for v,y in fields.items()}==d['bootstrap_field_sha256'][j];nf+=len(fields)
        if repair:
            for v,y in fields.items():exactfit(y,row['model'][v],row['info'][v]);assert row['info'][v]['improvement']>=-1e-10
        if j not in (0,127):continue
        mat=e.matrices(fields,row['model']);cv=V.Legacy.direct_metric(e,e.pilot,row['model'])
        for b in e.banks:close(S.gram(mat[b]-pilot[b]),a[b+'_bootstrap_grams'][j]);close(cv[b],a[b+'_bootstrap_covariances'][j])
        if not repair:
            for v,y in fields.items():exactfit(y,row['model'][v],row['info'][v])
    return dict(status='PASS',record=str(path.relative_to(ROOT)),operator_scores=ns,regions=nr,certificates=nc,field_hashes=nf,high_precision_fallbacks=len(V.proofs),input_sha256={str(path):sha(path)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['design','repair','profiles']);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();rows=[]
    if a.kind=='profiles':
        for p in (ROOT/'noise_profiles').glob('*.json'):
            d=read(p)
            with np.load(ROOT/'noise_inputs'/(d['key']+'.npz')) as z:y=z['field']
            exactfit(y,[d['sigma'],[d['rho']]],d,True);rows.append(dict(record=p.name,status='PASS'))
    else:
        paths=list((ROOT/('results' if a.kind=='design' else 'repair_results')).glob('*.json'));assert len(paths)==(9 if a.kind=='design' else 3)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(result,str(p)) for p in paths]):r=f.result();rows.append(r);print(r,flush=True)
    dump(ROOT/('audit_'+a.kind+'.json'),dict(status='PASS',rows=rows,source_sha256=src('audit_new.py')))

if __name__=='__main__':main()
