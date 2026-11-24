from context import *
import audit_new as A
from scipy.linalg import eigvalsh
from types import MethodType
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,traceback,argparse

def one(path):
    path=Path(path);e,d,a=load_saved(path);fp=ROOT/'frozen'/path.name;f=read(fp);assert sha(fp)==d['frozen_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256']
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h
    e.matrices=MethodType(lambda obj,fields,model=None,correct=True:matrices(obj,fields,model,correct),e)
    assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256'];assert {v:R.digest(y) for v,y in e.pilot.items()}==d['pilot_sha256']
    for v,y in e.raw.items():A.exactfit(y,e.model[v],d['fit_info'][v])
    obs=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);clean=e.matrices(e.p.data,correct=False);cov=A.V.Legacy.direct_metric(e,e.pilot,e.model);ns=nr=nc=nf=nt=0
    for b in e.banks:
        for tag,m in [('observed',obs[b]),('pilot',pilot[b]),('clean',clean[b])]:A.close(m,a[b+'_'+tag])
        A.close(cov[b],a[b+'_covariance']);A.close(S.gram(obs[b]-clean[b]),a[b+'_error_gram']);A.close(A.eigen_score(a[b+'_error_gram'],cov[b],e.active),d['actual_scores'][b])
        for mode in ('studentized','fixed_metric'):
            cc=a[b+'_bootstrap_covariances'] if mode=='studentized' else [cov[b]]*128
            qs=[A.eigen_score(g,c,e.active) for g,c in zip(a[b+'_bootstrap_grams'],cc)];A.close(qs,a[mode+'_'+b+'_scores']);ns+=128;cut=sorted(a[mode+'_'+b+'_scores'])[122];assert cut==d['cuts'][mode][b];assert d['coverage'][mode][b]==(d['actual_scores'][b]<=cut)
            g=S.gram(obs[b])
            for c,r in zip(e.cs,d['regions'][mode][b]):A.V.verified(r,g-cut*cov[b],g,cov[b],c,cut);nr+=1
            for row in d['references']:
                if (row['mode'],row['bank'])!=(mode,b):continue
                c=next(c for c in e.cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(e.labels));bound=cut*float(v@cov[b]@v);assert row['noise_covered']==(v@a[b+'_error_gram']@v<=bound);assert row['zero_accepted']==(v@g@v<=bound)
    for mode,pairs in d['joint'].items():
        for pair,item in pairs.items():
            bb=item['banks'];cut=sorted(np.maximum(*[a[mode+'_'+b+'_scores'] for b in bb]))[122];assert cut==item['cutoff'];assert d['coverage'][mode][pair]==(max(d['actual_scores'][b] for b in bb)<=cut)
            for c,cert in zip(e.cs,item['certificates']):A.certcheck(cert,[S.gram(obs[b]) for b in bb],[cov[b] for b in bb],cut,c);nc+=1
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]))
    for j,row in enumerate(d['refits']):
        fields,_=e.draw(rng);assert {v:R.digest(y) for v,y in fields.items()}==d['bootstrap_field_sha256'][j];nf+=len(fields)
        for v,y in fields.items():A.exactfit(y,row['model'][v],row['info'][v]);nt+=1
        if j in (0,127):
            mat=e.matrices(fields,row['model']);cv=A.V.Legacy.direct_metric(e,e.pilot,row['model'])
            for b in e.banks:A.close(S.gram(mat[b]-pilot[b]),a[b+'_bootstrap_grams'][j]);A.close(cv[b],a[b+'_bootstrap_covariances'][j])
    return dict(status='PASS',record=path.name,operator_scores=ns,regions=nr,certificates=nc,field_hashes=nf,exact_fit_replays=nt,input_sha256={str(path):sha(path)})

def safe(path):
    try:r=one(path)
    except Exception:r=dict(status='FAIL',record=Path(path).name,error=traceback.format_exc())
    dump(ROOT/'audits'/Path(path).name,r);return r

def main():
    pending={};done=set();rows=[]
    with ProcessPoolExecutor(max_workers=1) as pool:
        while len(done)<18 or pending:
            for path in sorted((ROOT/'results').glob('*.json')):
                if path.name not in done and len(pending)<1:pending[pool.submit(safe,str(path))]=path;done.add(path.name)
            for f in list(pending):
                if f.done():r=f.result();rows.append(r);pending.pop(f);print('audit',r['record'],r['status'],r.get('error',''),flush=True)
            time.sleep(1)
    out=dict(status='PASS' if len(rows)==18 and all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=sources('audit.py'))
    dump(ROOT/'audit.json',out);assert out['status']=='PASS'

if __name__=='__main__':main()
