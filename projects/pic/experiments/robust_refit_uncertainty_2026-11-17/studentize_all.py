from common import *
import paired_study_fast as E
from scipy.linalg import eigvalsh
import argparse,time
MC=32
H=.001
TRUST=.05
from patch_studentizer import Patches as Studentizer
def run(name,profile,seed):
    start=time.monotonic();key=f'{name}_{profile}_seed{seed}';path=ROOT/'results'/(key+'.json');d=read(path);e=E.Study(name,seed,profile);student=Studentizer(e);student.build();params=[student.encode(r['model']) for r in d['refits']['check']];validation=student.validate(params);anchor={b:g[0,0] for b,g in student.cross.items()};arrays={b+'_response_cross':g for b,g in student.cross.items()};scores={tag:{b:[] for b in e.banks} for tag in ('check','actual')}
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    metrics={};regions={};counts={};covariances={tag:{b:[] for b in e.banks} for tag in ('check','actual')}
    for j,t in enumerate(params):
        covs=student.evaluate(t)
        for b,cov in covs.items():
            covariances['check'][b].append(cov);w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['check'][b].append(float(S.operator_scores(a[f'refit_{b}_check_grams'][j:j+1],w,e.active)[0]))
    for b in e.banks:
        cut=float(np.quantile(scores['check'][b],.95));metrics[b]=dict(cutoff=cut);g=a[f'refit_{b}_observed_gram'];regions[b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*anchor[b],g,anchor[b],c,cut)) for c in e.cs];arrays[b+'_anchor_sigma']=anchor[b];arrays[b+'_check_scores']=np.array(scores['check'][b])
    for b in e.banks:arrays[b+'_check_covariances']=np.stack(covariances['check'][b])
    fp=ROOT/'studentized_all_frozen'/(key+'.json');src=hashes('studentize_all.py','studentize.py','patch_studentizer.py','studentization_protocol.txt','studentization_all_protocol.txt','paired_study_fast.py','local6_fit.py','fast_covariance.py');dump(fp,dict(system=name,profile=profile,seed=seed,validation=validation,metrics=metrics,regions=regions,center=student.center,check_models_sha256=R.digest(np.stack(params)),array_hashes={k:R.digest(v) for k,v in arrays.items()},source_sha256=src,input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))}));fhash=sha(fp)
    for j,row in enumerate(d['refits']['actual']):
        covs=student.evaluate(student.encode(row['model']))
        for b,cov in covs.items():
            covariances['actual'][b].append(cov);w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['actual'][b].append(float(S.operator_scores(a[f'refit_{b}_actual_error_grams'][j:j+1],w,e.active)[0]))
    for b in e.banks:arrays[b+'_actual_scores']=np.array(scores['actual'][b]);counts[b]=int(np.sum(arrays[b+'_actual_scores']<=metrics[b]['cutoff']))
    for b in e.banks:arrays[b+'_actual_covariances']=np.stack(covariances['actual'][b])
    ap=ROOT/'studentized_all'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays)
    result=dict(system=name,profile=profile,seed=seed,status='COMPLETE',coverage_counts=counts,validation=validation,exact_metric_calls=student.exact_calls,response_metric_calls=student.response_calls,metrics=metrics,regions=regions,seconds=time.monotonic()-start,frozen_sha256=fhash,array_sha256=sha(ap),source_sha256=src,input_sha256={str(path):sha(path)})
    dump(ap.with_suffix('.json'),result);print(key,counts,validation['status'],'exact',student.exact_calls,'seconds',round(result['seconds'],1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system');p.add_argument('profile');p.add_argument('seed',type=int);a=p.parse_args();run(a.system,a.profile,a.seed)
