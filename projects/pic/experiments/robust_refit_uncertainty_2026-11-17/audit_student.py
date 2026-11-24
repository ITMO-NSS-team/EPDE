from common import *
import paired_study_fast as E
from studentize import Studentizer
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util,argparse
spec=importlib.util.spec_from_file_location('student_region_checks',S.ROOT/'audit_joint.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
def one(path):
    path=Path(path);d=read(path);fpath=ROOT/'studentized_all_frozen'/path.name;f=read(fpath);primary=ROOT/'results'/path.name;pd=read(primary);e=E.Study(d['system'],d['seed'],d['profile']);links=0
    assert sha(fpath)==d['frozen_sha256'] and sha(path.with_suffix('.npz'))==d['array_sha256']
    for dd in (d,f):
        for key in ('source_sha256','input_sha256'):
            for p,h in dd[key].items():assert sha(p)==h;links+=1
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    with np.load(primary.with_suffix('.npz')) as z:b={k:z[k] for k in z.files}
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h;links+=1
    scores=regions=0;approx_errors=[]
    for bank in e.banks:
        for stream,tag in [('check','check'),('actual','actual_error')]:
            gg=b[f'refit_{bank}_{tag}_grams'];covs=a[f'{bank}_{stream}_covariances'];qq=[]
            assert len(covs)==256 and np.isfinite(covs).all()
            for gram,cov in zip(gg,covs):
                ss=cov[np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(ss));ss=ss/sc[:,None]/sc[None,:];g=gram[np.ix_(e.active,e.active)]/sc[:,None]/sc[None,:];qq.append(eigvalsh((g+g.T)/2,(ss+ss.T)/2)[-1])
            np.testing.assert_allclose(qq,a[f'{bank}_{stream}_scores'],rtol=5e-7,atol=5e-9);scores+=256
        cut=np.quantile(a[bank+'_check_scores'],.95);assert cut==d['metrics'][bank]['cutoff'];assert d['coverage_counts'][bank]==np.sum(a[bank+'_actual_scores']<=cut)
        sigma=a[bank+'_anchor_sigma'];np.testing.assert_array_equal(sigma,a[bank+'_response_cross'][0,0]);g=b[f'refit_{bank}_observed_gram']
        for c,r in zip(e.cs,d['regions'][bank]):Q.region_check(r,g-cut*sigma,g,sigma,c,cut);regions+=1
    # Recompute direct 32-field metrics at a check draw and an actual draw.
    student=Studentizer(e)
    for stream,j in [('check',0),('actual',255)]:
        t=student.encode(pd['refits'][stream][j]['model']);direct=student.direct(t)
        for bank,g in direct.items():
            ss=g[np.ix_(e.active,e.active)];approx=a[f'{bank}_{stream}_covariances'][j][np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(ss));ss=ss/sc[:,None]/sc[None,:];approx=approx/sc[:,None]/sc[None,:]
            eig=eigvalsh((approx+approx.T)/2,(ss+ss.T)/2);error=float(np.max(np.abs(eig-1)));assert error<=.011,(path,stream,bank,error);approx_errors.append(error)
    core=ROOT/'studentized'/path.name
    if core.exists():
        prior=read(core);assert prior['coverage_counts']==d['coverage_counts']
        with np.load(core.with_suffix('.npz')) as z:
            for bank in e.banks:
                for stream in ('check','actual'):np.testing.assert_allclose(a[bank+'_'+stream+'_scores'],z[bank+'_'+stream+'_scores'],rtol=2e-12,atol=2e-12)
    out=dict(status='PASS',record=path.stem,operator_scores=scores,regions=regions,direct_metric_checks=4,max_direct_metric_relative_error=max(approx_errors),hash_links=links,result_sha256=sha(path));dump(ROOT/'student_audit_records'/path.name,out);return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=7);args=p.parse_args();files=list((ROOT/'studentized_all').glob('*.json'));assert len(files)==69
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,str(p)) for p in files]):r=f.result();rows.append(r);print(r['record'],'PASS',flush=True)
    totals={k:sum(r[k] for r in rows) for k in ('operator_scores','regions','direct_metric_checks','hash_links')};dump(ROOT/'audit_student.json',dict(status='PASS',records=69,totals=totals,max_direct_metric_relative_error=max(r['max_direct_metric_relative_error'] for r in rows),rows=rows,source_sha256=hashes('audit_student.py')));print(totals,flush=True)
if __name__=='__main__':main()
