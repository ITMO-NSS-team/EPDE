"""Post-hoc diagnosis; no fitted coefficient or selection decision is changed."""
import run_outer as S
np,R,ROOT=S.np,S.R,S.ROOT
import json,csv
from scipy.linalg import eigvalsh


def main():
    rows=[];sensitivity=[];eiv=[]
    for seed in (6,7,8):
        path=ROOT/'sensitivity'/f'seed{seed}.json';d=json.loads(path.read_text());c=d['candidate']
        with np.load(ROOT/'training_arrays'/f'kdv_cossin_seed{seed}.npz') as z:
            for row in d['rows']:
                for i,truth in enumerate(c['exact']):
                    entry=dict(seed=seed,training=row['training'],bank=row['bank'],method=row['method'],index=i,exact=truth,
                               bootstrap_successes=row['bootstrap_successes'],bootstrap_total=64)
                    for key in ('observed_coefficients','pilot_coefficients','native_clean_ols_coefficients','error',
                                'bootstrap_mean','bootstrap_sd','bootstrap_drift_from_pilot','error_in_conditional_sd'):
                        if row.get(key) is not None:entry[key]=row[key][i]
                    if row.get('linearized',{}).get('status')=='ok':
                        entry['conditional_sd']=row['linearized']['conditional_sd'][i]
                        entry['normalized_design_eigen_ratio']=row['linearized']['normalized_design_eigen_ratio']
                    sensitivity.append(entry)
                if row['method']!='irls' or row['pilot_status']!='ok':continue
                tag=f"g{row['training']}_{row['bank']}";pilot=z[tag+'_pilot'];draws=z[tag+'_draws'];coef=np.asarray(row['pilot_coefficients'])
                delta=draws-draws.mean(axis=0);dx=delta[:,:,c['xi']];dy=delta[:,:,c['yi']]
                residual=dy-np.einsum('brj,j->br',dx,coef)
                cov=residual.T@residual/(len(draws)-1);ev,u=np.linalg.eigh((cov+cov.T)/2)
                assert ev[0]>S.E.RCOND*ev[-1]
                w=(u/np.sqrt(ev)).T;x=w@pilot[:,c['xi']]
                wx=np.einsum('rs,bsj->brj',w,dx);wr=residual@w.T
                signal=x.T@x;noise=np.einsum('brj,brk->jk',wx,wx)/(len(draws)-1)
                score=np.einsum('brj,br->j',wx,wr)/(len(draws)-1)
                shift=np.linalg.solve(signal+noise,score)
                ratios=eigvalsh(noise,signal)
                eiv.append(dict(seed=seed,training=row['training'],bank=row['bank'],pilot_coefficients=coef,
                    conditional_expected_normal_score=score,fixed_weight_expected_coefficient_shift=shift,
                    actual_bootstrap_mean_shift=row.get('bootstrap_drift_from_pilot'),
                    generalized_noise_to_signal_eigenvalues=ratios))
    for bank in ('local','broad','mixed'):
        for method in ('ols','irls'):
            rr=[r for r in sensitivity if r['bank']==bank and r['method']==method]
            for i in range(3):
                chosen=[r for r in rr if r['index']==i];entry=dict(bank=bank,method=method,index=i)
                for key in ('observed_coefficients','pilot_coefficients','native_clean_ols_coefficients',
                            'bootstrap_mean','bootstrap_sd','bootstrap_drift_from_pilot','error_in_conditional_sd',
                            'conditional_sd','normalized_design_eigen_ratio'):
                    vals=[r[key] for r in chosen if key in r]
                    if vals:entry['median_'+key]=float(np.median(vals))
                entry['observed_complete_folds']=sum('observed_coefficients' in r for r in chosen)
                rows.append(entry)
    for name,rr in [('coefficient_sensitivity.csv',sensitivity),('sensitivity_summary.csv',rows)]:
        keys=list(dict.fromkeys(k for r in rr for k in r))
        with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rr)
    R.dump(ROOT/'eiv_normal_score_diagnostic.json',dict(post_hoc=True,no_refitting_or_selection=True,
        description='Conditional fixed-weight expected normal-equation displacement; dynamic IRLS bootstrap may differ.',rows=eiv,
        source_sha256=S.sha(__file__),input_sha256={str(p):S.sha(p) for p in sorted((ROOT/'sensitivity').glob('*.json'))}))
    print('IRLS conditional expected coefficient shift at frozen pilot weights:')
    for bank in ('local','broad','mixed'):
        rr=[r for r in eiv if r['bank']==bank]
        print(bank,'cases',len(rr),'normal shift',np.median([r['fixed_weight_expected_coefficient_shift'] for r in rr],axis=0),
              'bootstrap shift',np.median([r['actual_bootstrap_mean_shift'] for r in rr if r['actual_bootstrap_mean_shift'] is not None],axis=0),
              'noise/signal eigenvalues',np.median([r['generalized_noise_to_signal_eigenvalues'] for r in rr],axis=0))


if __name__=='__main__':main()
