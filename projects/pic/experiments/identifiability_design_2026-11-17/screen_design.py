from design_engine import *
import time

def main():
    start=time.monotonic();e,d,path=primary('kdv_cossin','iid',51)
    original=e.matrices(e.pilot,correct=False);oldcov=Legacy.paired(e,{'fixed':e.pilot},e.model)['fixed']
    build(e,menu(e));pm=matrices(e,e.pilot,correct=False);cv=all_metrics(e,{'fixed':e.pilot},e.model)['fixed'];rows=[];arrays={}
    for b in ('local','broad'):
        np.testing.assert_allclose(S.gram(pm[b]),S.gram(original[b]),rtol=2e-10,atol=1e-12)
        np.testing.assert_allclose(cv[b],oldcov[b],rtol=2e-10,atol=1e-12)
    for b,m in pm.items():
        arrays[b+'_pilot']=m;arrays[b+'_covariance']=cv[b];w,info=S.metric(cv[b],e.active)
        if w is None:rows.append(dict(bank=b,status='ineligible',metric=info));continue
        x=m[:,e.active];det=m[:,e.deterministic]
        if len(e.deterministic):x=x-det@np.linalg.lstsq(det,x,rcond=1e-12)[0]
        info_gram=S.gram(x@w);eig=np.linalg.eigvalsh((info_gram+info_gram.T)/2)
        assert min(eig)>=-1e-8*max(1.,max(eig))
        proxy=float(np.sum(np.log1p(np.maximum(eig,0.))))
        rows.append(dict(bank=b,status='ok',score=proxy,eigenvalues=eig,complex_rows=int(np.prod(e.banks[b].shape)),metric=info))
    ranking=sorted((r for r in rows if r['status']=='ok' and r['bank']!='local'),key=lambda r:-r['score']);selected=ranking[0]['bank']
    ap=ROOT/'design_screen.npz';np.savez_compressed(ap,**arrays)
    out=dict(status='FROZEN',selected=selected,rows=rows,ranking=[r['bank'] for r in ranking],specs=e.specs,
             input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))},array_sha256=sha(ap),source_sha256=src('screen_design.py','protocol.txt'),seconds=time.monotonic()-start)
    dump(ROOT/'design_screen.json',out);print('FROZEN',selected,[(r['bank'],r.get('score')) for r in rows],flush=True)
    clean=matrices(e,e.p.data,correct=False);diagnostics=[]
    for c in e.cs:
        if c['label']!='reference':continue
        v=S.E.vector(c,c['exact'],len(e.labels))
        for b,m in clean.items():
            x=m[:,c['xi']];norm=np.sqrt(np.sum(x*x,axis=0));corr=S.gram(x/norm)
            diagnostics.append(dict(bank=b,candidate=c['id'],clean_feature_correlation_eigenvalues=np.linalg.eigvalsh(corr),clean_reference_residual=float(np.linalg.norm(m@v)),clean_target_norm=float(np.linalg.norm(m[:,c['yi']]))))
    dump(ROOT/'design_diagnostic.json',dict(status='COMPLETE',frozen_screen_sha256=sha(ROOT/'design_screen.json'),diagnostics=diagnostics,source_sha256=src('screen_design.py')))

if __name__=='__main__':main()
