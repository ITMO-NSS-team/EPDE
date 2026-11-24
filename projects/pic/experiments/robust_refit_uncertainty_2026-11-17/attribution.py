from common import *
import paired_study_fast as E
from scipy.linalg import eigvalsh
import time
def one(name,profile,seed):
    key=f'{name}_{profile}_seed{seed}';path=ROOT/'results'/(key+'.json');d=read(path);e=E.Study(name,seed,profile)
    true={v:[.05*np.std(y),[e.rho]*y.ndim,[3]*y.ndim] for v,y in e.p.data.items()};clean=e.matrices(e.p.data,correct=False)
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    oracle={b:[] for b in e.banks};means={(mode,b):np.zeros_like(clean[b]) for mode in ('fixed','refit','oracle') for b in e.banks}
    for j in range(256):
        fields=e.true_field(j)
        for mode,m in [('oracle',true),('fixed',e.model),('refit',d['refits']['actual'][j]['model'])]:
            for b,mat in e.matrices(fields,m).items():
                err=mat-clean[b];means[mode,b]+=err/256
                if mode=='oracle':oracle[b].append(S.gram(err))
    rows=[];arrays={}
    for b in e.banks:
        og=np.stack(oracle[b]);arrays[b+'_oracle_error_grams']=og
        for mode in ('fixed','refit'):
            pre=f'{mode}_{b}';metric=a[pre+'_white'];cut=d['metrics'][mode][b]['cutoff'];scores=S.operator_scores(og,metric,e.active);arrays[pre+'_oracle_scores']=scores;arrays[pre+'_mean_error']=means[mode,b]
            sigma=a[pre+'_sigma'][np.ix_(e.active,e.active)];g=og.mean(axis=0)[np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(sigma));ss=sigma/sc[:,None]/sc[None,:];gg=g/sc[:,None]/sc[None,:];eig=eigvalsh((gg+gg.T)/2,ss)
            original=next(r['operator_coverage_count'] for r in d['statistics'] if (r['mode'],r['bank'])==(mode,b))
            rows.append(dict(system=name,profile=profile,seed=seed,mode=mode,bank=b,original_coverage_count=original,oracle_coverage_count=int(np.sum(scores<=cut)),empirical_mean_score_cutoff_ratio=float(S.operator_scores(S.gram(means[mode,b])[None],metric,e.active)[0]/cut),oracle_mean_score_cutoff_ratio=float(S.operator_scores(S.gram(means['oracle',b])[None],metric,e.active)[0]/cut),second_moment_min=float(eig[0]),second_moment_max=float(eig[-1])))
        arrays[b+'_oracle_mean_error']=means['oracle',b]
    ap=ROOT/'attribution'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays);dump(ap.with_suffix('.json'),dict(rows=rows,array_sha256=sha(ap),source_sha256=hashes('attribution.py','attribution_protocol.txt','paired_study_fast.py','fast_covariance.py'),input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))}))
    print(key,[(r['mode'],r['bank'],r['original_coverage_count'],r['oracle_coverage_count']) for r in rows],flush=True)
def main():
    jobs=[(n,p,s) for n,p in [('ns','iid'),('pde_compound','iid'),('ks','iid'),('ks','positive'),('ks','negative'),('ac','positive')] for s in (42,43,44)]
    pending=list(jobs)
    while pending:
        found=False
        for job in list(pending):
            n,p,s=job
            if (ROOT/'results'/f'{n}_{p}_seed{s}.json').exists():one(*job);pending.remove(job);found=True
        if not found:time.sleep(2)
    rows=[r for p in (ROOT/'attribution').glob('*.json') for r in read(p)['rows']];dump(ROOT/'attribution_summary.json',dict(status='COMPLETE',records=18,rows=rows,source_sha256=hashes('attribution.py')))
if __name__=='__main__':main()
