from envelope import *
from scipy.linalg import eigvalsh

def score(e,g,c):
    ix=np.ix_(e.active,e.active);a=g[ix];b=c[ix];sc=np.sqrt(np.diag(b));a=a/sc[:,None]/sc;b=b/sc[:,None]/sc;return float(eigvalsh((a+a.T)/2,(b+b.T)/2)[-1])

def main():
    rows=[]
    for key in ('ode_iid_seed60','ode_positive_seed57'):
        e,d,a=load_case(key)
        for index,model in enumerate((e.model,{'u':[.08,[-.4],[2]]},{'u':[.2,[.8],[2]]})):
            meta,arrays=point(e,a,dict(index=index,status='ok',model=model));assert meta['status']=='ok';direct={b:[] for b in e.banks}
            for j in range(16):
                rng=np.random.default_rng(np.random.SeedSequence([984105,e.system_id,e.pid,j]));fields={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in e.pilot.items()}
                for b,m in moments(e,fields,model).items():direct[b].append(m-a[b+'_pilot'])
            for b in e.banks:
                x=np.stack(direct[b]);cv=np.einsum('bri,brj->ij',x,x)/16;np.testing.assert_allclose(cv,arrays[b+'_covariance'],rtol=1e-8,atol=1e-12)
                q=[score(e,g,cv) for g in arrays[b+'_grams']];np.testing.assert_allclose(q,arrays[b+'_scores'],rtol=3e-8,atol=1e-10)
                proof=meta['bounds'][b];h=(1+proof['tau'])*arrays[b+'_correction_gram']+(1+1/proof['tau'])*meta['conditional_cutoff']*cv;np.testing.assert_allclose(score(e,h,a[b+'_covariance']),proof['bound'],rtol=3e-8)
            for j in (0,127):
                rng=np.random.default_rng(np.random.SeedSequence([985802,e.system_id,e.pid,e.seed,j]));fields={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in e.pilot.items()}
                for b,m in moments(e,fields,model).items():np.testing.assert_allclose(S.gram(m-a[b+'_pilot']),arrays[b+'_grams'][j],rtol=1e-8,atol=1e-12)
            rows.append(dict(key=key,index=index,conditional_cutoff=meta['conditional_cutoff'],bounds=meta['bounds']))
    dump(ROOT/'envelope_checks.json',dict(status='PASS',rows=rows,source_sha256=src('envelope.py','envelope_batch.py','check_envelope.py')));print('ENVELOPE CHECKS PASS',len(rows),flush=True)

if __name__=='__main__':main()
