import moments as M
O,R,A,np=M.O,M.R,M.A,M.np
from pathlib import Path
import json,csv,hashlib,argparse


def write(name,rows):
    with (M.ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['targeted','core'],default='targeted');args=ap.parse_args()
    names=['wave','ks','ac','kdv_cossin'] if args.suite=='targeted' else R.datasets.names('core')
    sources=[M.ROOT/'run_broad_validation.py',Path(M.__file__),Path(O.__file__),Path(M.Q.__file__),Path(M.B.__file__),Path(M.V.__file__)]
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    retention=[];ranking=[];options=[];uncertainty=[];failures=[];case_hashes={};weighting=[];baseline_counts=0;baseline_cases={}
    for name in names:
        for seed in range(3):
            path=M.ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());cs=d['candidates']
            assert d['source_sha256']==expected
            assert d['input_sha256']=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in d['input_sha256']}
            case_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            prior=json.loads((M.J.L.ROOT/'results'/path.name).read_text())
            for slot in sorted({c['slot'] for c in cs}):
                kept=bool(A.frontier(prior['aggregates']['local_continuous']['cross']['scores'],d['instability'],cs,slot))
                baseline_cases[name,seed,slot]=kept;baseline_counts+=kept
            for e in d['infeasible']:failures.append(dict(system=name,seed=seed,**e))
            for bank,gg in d['weighting'].items():
                for g,aa in gg.items():
                    for method,meta in aa.items():
                        weighting.append(dict(system=name,seed=seed,bank=bank,training=int(g),method=method,rank=meta['rank'],rows=meta['rows'],
                            feasible=meta['feasible'],condition=meta['eigenvalues'][-1]/meta['eigenvalues'][0] if meta['eigenvalues'][0]>0 else None,
                            identity_error=meta.get('whitened_covariance_max_error')))
            with np.load(path.with_name(path.stem+'_samples.npz')) as z:
                for arm,rows in d['rows'].items():
                    noise=z[arm+'_128']
                    for kind,agg in d['aggregates'][arm].items():
                        for slot in sorted({c['slot'] for c in cs}):
                            row=dict(system=name,seed=seed,slot=slot,arm=arm,fit=kind,old_local_retained=baseline_cases[name,seed,slot])
                            for metric in ('cross','ordinary','clean_full','clean_subsample'):
                                row[metric+'_retained']=bool(A.frontier(agg[metric]['scores'],d['instability'],cs,slot))
                            retention.append(row)
                    for slot in sorted({c['slot'] for c in cs}):
                        oi=[i for i,r in enumerate(rows) if r['kind']=='option' and r['slot']==slot]
                        ri=next(i for i,r in enumerate(rows) if r['kind']=='baseline' and r['slot']==slot and r['label']=='reference')
                        ei=next(i for i,r in enumerate(rows) if r['kind']=='exact' and r['slot']==slot)
                        best=min(oi,key=lambda i:rows[i]['cross']);bc=min(oi,key=lambda i:rows[i]['frozen_clean_energy']);bb=min(oi,key=lambda i:rows[i]['bank_clean_energy'])
                        ref,choice,cleanbest=rows[ri],rows[best],rows[bc]
                        rank=dict(system=name,seed=seed,slot=slot,arm=arm,lowest_score_option=choice['option'],
                            frozen_clean_best=cleanbest['option'],bank_clean_best=rows[bb]['option'],
                            baseline_clean_energy=ref['frozen_clean_energy'],lowest_score_clean_energy=choice['frozen_clean_energy'],
                            accuracy_ratio=choice['frozen_clean_energy']/ref['frozen_clean_energy'],
                            agrees_with_frozen_best=choice['option']==cleanbest['option'],
                            choice_vs_baseline_gap=choice['cross']-ref['cross'],
                            choice_vs_baseline_clean_gap=choice['clean_full']-ref['clean_full'],
                            choice_vs_baseline_sd=float(np.std(noise[:,best]-noise[:,ri],ddof=1)),
                            frozen_best_vs_baseline_gap=cleanbest['cross']-ref['cross'],
                            frozen_best_vs_baseline_sd=float(np.std(noise[:,bc]-noise[:,ri],ddof=1)))
                        ranking.append(rank)
                        exact=rows[ei];uncertainty.append(dict(system=name,seed=seed,slot=slot,arm=arm,exact_score=exact['cross'],
                            exact_clean=exact['clean_full'],exact_subsample_clean=exact['clean_subsample'],sd128=exact['sd128'],sd64=exact['sd64']))
                        for i in oi:options.append(dict(system=name,seed=seed,slot=slot,arm=arm,option=rows[i]['option'],cross=rows[i]['cross'],
                            bank_clean_energy=rows[i]['bank_clean_energy'],frozen_clean_energy=rows[i]['frozen_clean_energy'],sd128=rows[i]['sd128']))
    summary={}
    for arm in sorted({r['arm'] for r in retention}):
        summary[arm]={}
        for kind in ('baseline','joint'):
            rr=[r for r in retention if r['arm']==arm and r['fit']==kind]
            summary[arm][kind]=dict(evaluated=len(rr),retained=sum(r['cross_retained'] for r in rr),
                old_local_retained_on_same_cases=sum(r['old_local_retained'] for r in rr),
                gains=sum(r['cross_retained'] and not r['old_local_retained'] for r in rr),
                losses=sum(not r['cross_retained'] and r['old_local_retained'] for r in rr))
    prefix=args.suite+'_'
    for name,rows in [('retention.csv',retention),('ranking.csv',ranking),('options.csv',options),('uncertainty.csv',uncertainty),('weighting.csv',weighting)]:write(prefix+name,rows)
    R.dump(M.ROOT/(prefix+'summary.json'),summary);R.dump(M.ROOT/(prefix+'infeasible.json'),failures)
    audit=dict(system_runs=len(case_hashes),equation_seed_cases=len(baseline_cases),old_local_retained=baseline_counts,
        infeasible_fold_arms=len(failures),hashes_verified=True,result_sha256=case_hashes,
        max_whitening_identity_error=max(r['identity_error'] for r in weighting if r['identity_error'] is not None))
    R.dump(M.ROOT/(prefix+'audit.json'),audit)
    print(json.dumps(summary,indent=2));print('Audit:',{k:v for k,v in audit.items() if k!='result_sha256'})
    for name in ['wave','ks','ac','kdv_cossin']:
        for arm in ['q0_plain','q0_white128','q1_plain','q1_white128']:
            rr=[r for r in ranking if r['system']==name and r['arm']==arm]
            print(name,arm,'lowest-score options',[(r['seed'],r['lowest_score_option']) for r in rr],
                'median frozen energy',float(np.median([r['lowest_score_clean_energy'] for r in rr])))


if __name__=='__main__':main()
