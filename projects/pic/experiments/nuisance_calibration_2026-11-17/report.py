from calibration_context import *
from summarize_profiles import stats,coverage
from collections import Counter
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    for name in ('audit.json','bootstrap_audit.json','propagation_audit.json'):assert read(ROOT/name)['status']=='PASS'
    prof=read(ROOT/'profile_summary.json');boot=[read(p) for p in sorted((ROOT/'bootstrap_results').glob('*.json'))];prop=[read(p) for p in sorted((ROOT/'propagation_results').glob('*.json'))];paired=[read(p) for p in (ROOT/'paired_profiles').glob('*.json')];assert len(boot)==len(prop)==len(paired)==24;profiles=('iid','positive','negative');methods=('uncorrected','basic_nominal','basic_empirical','finite_hull');bcov={};pcov={};geometry={};bias={};widths=[];warnings=[];paired_cov={}
    for profile in profiles:
        paired_cov[profile]={kind:{method:coverage([d['coverage'][kind][method] for d in paired if d['profile']==profile]) for method in ('lr','wald')} for kind in ('nominal','model_calibrated')}
        bb=[d for d in boot if d['profile']==profile];pp=[d for d in prop if d['profile']==profile];bcov[profile]={m:coverage([d['coverage'][m] for d in bb]) for m in methods};pcov[profile]={m:coverage([d['coverage'][m] for d in pp]) for m in pp[0]['coverage']};geometry[profile]={m:dict(Counter(d['primary_geometry'][m] for d in pp)) for m in ('studentized','fixed_metric')};geometry[profile].update(finite_union=dict(Counter(d['finite_union_geometry'] for d in pp)),scalar=dict(Counter(d['scalar_geometry'] for d in pp)));bias[profile]=dict(original_sigma_ratio=stats([d['sigma_ratio'] for d in bb]),basic_center_sigma_ratio=stats([d['basic_center_sigma_ratio'] for d in bb]),original_rho_error=stats([d['rho_error'] for d in bb]),basic_center_rho_error=stats([d['basic_center_rho_error'] for d in bb]))
    for d in prop:
        key=d['key'];p=read(ROOT/'propagation_primary'/(key+'.json'));u=read(ROOT/'propagation_frozen'/(key+'.json'));base=next(c for c in p['certificates']['studentized'] if c['label']=='reference');union=u['outer_box'];scalar=u['scalar_certificate'];b=read(ROOT/'bootstrap_frozen'/(key+'.json'));anchor=read(ROOT/'bootstrap_anchors'/(key+'.json'));warnings += [dict(key=key,draw=-1,status=anchor['fit_info']['status'])] if anchor['fit_info']['status']!='ok' else [];warnings += [dict(key=key,draw=r['draw'],status=r['info']['status']) for r in b['refits'] if r['info']['status']!='ok']
        for j in range(3):
            bw=base['outer_region']['upper'][j]-base['outer_region']['lower'][j] if base['geometry']=='bounded_outer' else None;uw=union['upper'][j]-union['lower'][j] if union is not None else None;sw=scalar['outer_region']['upper'][j]-scalar['outer_region']['lower'][j] if scalar['geometry']=='bounded_outer' else None;widths.append(dict(key=key,profile=d['profile'],coefficient=j,primary_width=bw,union_width=uw,scalar_width=sw,union_ratio=uw/bw if bw is not None and uw is not None else None,scalar_ratio=sw/bw if bw is not None and sw is not None else None))
    wr={m:stats([r[m+'_ratio'] for r in widths if r[m+'_ratio'] is not None]) if any(r[m+'_ratio'] is not None for r in widths) else None for m in ('union','scalar')};out=dict(status='COMPLETE',profile_study=prof,bootstrap_noise_coverage=bcov,bootstrap_bias=bias,equation_error_coverage=pcov,reference_geometry=geometry,outer_width_ratios_bounded_only=wr,width_coordinate_counts={m:sum(r[m+'_ratio'] is not None for r in widths) for m in ('union','scalar')},reference_acceptance=sum(d['reference_accepted'] for d in prop),warnings=warnings,source_sha256=src('report.py','summarize_profiles.py'))
    stress=read(ROOT/'profile_stress.json');out['profile_stress']={k:stress[k] for k in ('max_lr_improvement','nominal_membership_changes','calibrated_membership_changes')};out['posthoc_repair']=read(ROOT/'posthoc_repair.json')['rows'];out['paired_likelihood_coverage']=paired_cov;dump(ROOT/'summary.json',out);rows=[]
    for c,pp in prof['coverage'].items():
        for profile,kk in pp.items():
            for kind,mm in kk.items():
                for method,x in mm.items():rows.append(dict(cohort=c,profile=profile,method=kind+'_'+method,n=x['n'],covered=x['covered'],unresolved=x['unresolved'],lower95=x['binomial95_interval'][0],upper95=x['binomial95_interval'][1]))
    for profile,mm in bcov.items():
        for method,x in mm.items():rows.append(dict(cohort='fresh_bootstrap',profile=profile,method=method,n=x['n'],covered=x['covered'],unresolved=x['unresolved'],lower95=x['binomial95_interval'][0],upper95=x['binomial95_interval'][1]))
    for name,rr in [('noise_coverage.csv',rows),('coefficient_outer_widths.csv',widths)]:
        with (ROOT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
    fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained');x=np.arange(3);ax=axs[0,0]
    for cohort,label,offset,color in [('model_validation','Controlled Gaussian model',-.18,'#4477aa'),('ode','Fixed ODE signal',.18,'#cc6677')]:
        rr=[prof['coverage'][cohort][p]['model_calibrated']['lr'] for p in profiles];ys=[d['covered']/d['n'] for d in rr];err=np.array([[ys[j]-d['binomial95_interval'][0] for j,d in enumerate(rr)],[d['binomial95_interval'][1]-ys[j] for j,d in enumerate(rr)]]);ax.bar(x+offset,ys,.36,label=label,color=color,yerr=err,capsize=3)
        for j,d in enumerate(rr):ax.text(j+offset,ys[j]-.065,f"{d['covered']}/64",ha='center',color='white',fontsize=9)
    ax.axhline(.975,color='black',ls='--',lw=1);ax.set(xticks=x,xticklabels=profiles,ylim=(0,1.07),ylabel='True noise pair covered',title='Likelihood calibration does not transfer');ax.legend(fontsize=8,loc='lower left')
    ax=axs[0,1]
    for cohort,label,color in [('model_validation','Controlled model','#4477aa'),('ode','ODE','#cc6677')]:
        rr=[read(p) for p in (ROOT/cohort).glob('*.json')];rr=[r for r in rr if r['profile']=='positive'];ax.scatter([r['model'][0]/np.exp(r['true_noise'][0]) for r in rr],[r['model'][1][0] for r in rr],s=21,alpha=.7,label=label,c=color)
    ax.scatter([1],[.4],marker='*',s=130,c='black',label='Truth');ax.set(xlabel='Estimated / true noise amplitude',ylabel='Estimated noise correlation',title='The positive-noise bias is model dependent');ax.legend(fontsize=8)
    ax=axs[1,0]
    for k,(method,label) in enumerate([('uncorrected','No bias correction'),('basic_nominal','Basic / chi-square'),('basic_empirical','Basic / bootstrap')]):
        yy=[bcov[p][method]['covered']/8 for p in profiles];ax.bar(x+(k-1)*.24,yy,.23,label=label)
        for j,p in enumerate(profiles):ax.text(j+(k-1)*.24,yy[j]+.02,str(bcov[p][method]['covered'])+'/8',ha='center',fontsize=8)
    ax.axhline(.975,c='black',ls='--',lw=1);ax.set(xticks=x,xticklabels=profiles,ylim=(0,1.2),ylabel='True noise pair covered',title='Independent bootstrap test:24 fresh fields');ax.legend(fontsize=8,loc='lower left')
    ax=axs[1,1]
    for k,(method,label) in enumerate([('studentized','Primary'),('finite_union','Separate scenarios'),('scalar','Scalar compression')]):
        yy=[geometry[p][method].get('bounded_outer',0) for p in profiles];ax.bar(x+(k-1)*.24,yy,.23,label=label)
    ax.set(xticks=x,xticklabels=profiles,ylim=(0,9),ylabel='Bounded reference coefficient regions /8',title='Precision under66 ellipse scenarios');ax.legend(fontsize=8,loc='lower left')
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    boot_text='\n'.join(p+': '+', '.join(f"{m} {bcov[p][m]['covered']}/8" for m in methods) for p in profiles)
    paired_text='\n'.join(p+': '+', '.join(f"{m} {paired_cov[p]['model_calibrated'][m]['covered']}/8" for m in ('lr','wald')) for p in profiles)
    weak_text='\n'.join(p+': '+', '.join(f"{m} {pcov[p][m]['covered']}/8" for m in pcov[p]) for p in profiles)
    geometry_text='\n'.join(p+': '+', '.join(f"{m} bounded {geometry[p][m].get('bounded_outer',0)}/8" for m in geometry[p]) for p in profiles)
    text=['Noise-region calibration and solver-free sensitivity','===================================================','',
    '1. Controlled likelihood calibration versus the ODE benchmark',
    '384 controlled calibration fields(128 per profile),192 independent',
    'controlled validation fields(64 per profile),192 fresh ODE fields',
    '(seeds200..263,64 per profile). All selected original and restricted',
    'fits were replayed with the original Python likelihood. Numerical',
    'convergence does not imply statistical confidence validity.',
    'The controlled signal has stationary Gaussian AR2 curvature independent',
    'of AR1 measurement noise. It matches the assumed contrast likelihood.',
    'The ODE has a fixed deterministic signal. Controlled curvature parameters',
    'were fixed from prior fits; no new validation outcome tunes them.',
    'Target nuisance coverage97.5%. Nominal chi-square2 threshold7.37776.',
    'Controlled calibration uses rank126/128, frozen before validation.',
    'Model-calibrated profile-LR coverage:',
    '\n'.join(f"{cohort}/{p}: {prof['coverage'][cohort][p]['model_calibrated']['lr']}" for cohort in ('model_validation','ode') for p in profiles),
    'Wald coverage and nominal-cutoff comparisons are in summary.json/CSV.',
    'For positive correlation: controlled64/64 versus ODE7/64 covered.',
    'The ODE coverage95% binomial interval is approximately4.5%..21.2%,',
    'far below the97.5% target. Slight cutoff adjustment does not fix this.',
    'Median positive noise amplitude/truth: controlled0.9998, ODE1.4242.',
    'Median positive rho error: controlled-0.0038, ODE+0.2810.',
    'This is strong evidence that the assumed signal/noise model does not',
    'provide reliable nuisance uncertainty for this ODE. It is not a proof',
    'that all Gaussian curvature models or all solver-free estimators fail.',
    'Controlled calibration at one fixed nuisance configuration is not a',
    'uniform or deployable calibration for arbitrary signals/parameters.', '',
    'A posthoc80-start restricted-likelihood search on all64 positive ODE',
    'fields and all24 paired fields found better nuisance optima in some',
    'cases(maximum LR reduction6.0042), but changed ZERO nominal or calibrated',
    'membership decisions. Original results are preserved. This is a finite',
    'optimizer stress test, not a proof of global optimality.', '',
    'One already-covered field,positive/234, exposed a new unrestricted basin:',
    'the better restricted solution had lower objective than the stored full',
    'fit, producing a negative retrospective LR. Releasing that solution',
    'improved-2loglik by25.4127: sigma0.10054,rho0.33935 became',
    'sigma0.13023,rho0.60022(true sigma0.10301,rho0.4). Better optimization',
    'therefore increased the noise bias. The repaired diagnostic LR20.0859',
    'rejects truth; the original frozen positive count is7/64, while replacing',
    'this one result posthoc would give6/64. No replacement was made because',
    'the starting point came from truth-based validation. This neither explains',
    'away the57 original misses nor establishes a deployable optimizer repair.', '',
    '2. Continuous basic-bootstrap regions: independent follow-up',
    'The previous12-case ellipsoid comparison was archival, not validation.',
    'All12 empirical basic ellipsoids contained truth, motivating this frozen',
    '24-field follow-up(seeds264..271,8 per profile).128 refits each; all retained.',
    'Coordinates theta=(log sigma,atanh rho). Center2theta_hat-mean(theta_star),',
    'sample covariance V, and either chi-square2(.975) or empirical rank126 of',
    'centered bootstrap Mahalanobis radii. The uncorrected comparator uses',
    'theta_hat and the same bootstrap covariance. Constants are retained.',
    'Fresh noise-pair coverage:',boot_text,
    'Already-frozen model-calibrated likelihood methods on those SAME24 fields:',paired_text,
    'Observed estimate versus basic-center bias:',str(bias),
    'These ellipsoids are continuous regions. Their empirical calibration',
    'must still be checked:8 outer fields per profile cannot prove97.5%',
    'coverage. The finite reflected-point hull remains a descriptive comparator.', '',
    '3. Propagating uncertainty while preserving scenario geometry',
    'Frozen probe: ellipse center, observed estimate,32 angles on each of',
    'half and full radius(66 points). Each has128 conditional known-noise',
    'fields, MC16 normalization and97.5% joint rank126 cutoff. Both original',
    '95% primary regions are included in the finite union. The scalar',
    'triangle/Loewner compression is a paired comparator. No point is dropped.',
    'The66-point scan does NOT certify the continuum of the ellipse. A noise',
    'ellipse containing truth does not by itself validate downstream equation',
    'confidence coverage. This remains a finite sensitivity experiment.',
    'Full-library weak-error coverage:',weak_text,
    'The fixed-metric primary already covered24/24 in this cohort. No extra',
    'coverage gain over it is demonstrated here; the test measures precision',
    'when explicitly retaining additional noise scenarios.',
    'Reference geometry:',geometry_text,
    f"Reference equation accepted in {out['reference_acceptance']}/24 finite unions.",
    'Outer-width ratios versus primary studentized enclosure, bounded only:',str(wr),
    'Coordinates contributing to those ratios:',str(out['width_coordinate_counts']),
    'These are outer enclosure widths, not exact joint coefficient intervals.',
    'Boundedness alone also does not establish correct structure or exclude',
    'zero coefficients. All cutoffs were frozen before clean-moment scoring.', '',
    '4. Implementation and audit',
    'The original validated guarded fitter is used throughout. Substituting',
    'the exact native likelihood at initialization missed a better archived',
    'basin(1/54); adding3 starts still missed2/144. Those attempts were rejected',
    'and their failed records retained. The native final likelihood arithmetic',
    'remains independently checked against original Python. No ODE/PDE solver.',
    'Likelihood, gradient, Hessian, bootstrap draws, noise-region membership,',
    'generalized scores, scalar bounds and coefficient certificates audited.',
    'All prior experiment manifests remain unchanged; complete.py checks this.',
    'Selected-fit warnings in the24-case bootstrap:',str(warnings), '',
    '5. Interpretation and next step',
    'Do not promote the AR2 model-based likelihood regions: the positive-noise',
    'ODE coverage failure is substantial. The basic bootstrap tests whether',
    'reproducing the deterministic pilot can compensate for this bias.',
    'Keep any improved coverage separate from coefficient precision and from',
    'a formal confidence guarantee. The next methodological comparison should',
    'change the signal representation in noise estimation and validate on more',
    'independent signals, rather than trusting convergence or one likelihood',
    'model. To obtain continuous robust equation regions, finite nuisance scans',
    'must also be replaced by bounds over the full noise region.',
    'This round concerns ODE noise identification; the prior KdV weak-direction',
    'failure is unresolved. Dyad remains provisional. No production edit.', '']
    (ROOT/'findings.txt').write_text('\n'.join(text),encoding='utf-8');print('REPORT',json.dumps(dict(bootstrap=bcov,propagation=pcov,geometry=geometry,widths=wr)),flush=True)


if __name__=='__main__':main()
