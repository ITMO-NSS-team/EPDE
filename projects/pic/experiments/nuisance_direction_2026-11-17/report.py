from context import *
from collections import Counter
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def counts(rows):return dict(Counter(rows))
def stats(values):return dict(min=float(min(values)),median=float(np.median(values)),max=float(max(values)))

def main():
    data=[read(p) for p in sorted((ROOT/'results').glob('*.json'))];survey=[read(p) for p in sorted((ROOT/'survey').glob('*.json'))];guard=[read(p) for p in sorted((ROOT/'guard_results').glob('*.json'))];local=[read(p) for p in sorted((ROOT/'local_comparison').glob('*.json'))]
    assert len(data)==18 and len(survey)==96 and len(guard)==144 and len(local)==96
    summary={};geometry={};cutratios={};failures=[];extreme=[]
    for name in ('ode','kdv_cossin'):
        summary[name]={};geometry[name]={};cutratios[name]={}
        for profile in ('iid','positive','negative'):
            rr=[d for d in data if (d['system'],d['profile'])==(name,profile)];assert len(rr)==3
            summary[name][profile]={m:{b:sum(d['coverage'][m][b] for d in rr) for b in rr[0]['coverage'][m]} for m in ('studentized','fixed_metric')}
            geometry[name][profile]={m:{pair:counts(c['geometry'] for d in rr for c in d['joint'][m][pair]['certificates'] if c['label']=='reference') for pair in rr[0]['joint'][m]} for m in ('studentized','fixed_metric')}
            cutratios[name][profile]={b:stats([d['cuts']['fixed_metric'][b]/d['cuts']['studentized'][b] for d in rr]) for b in rr[0]['cuts']['studentized']}
    for d in data:
        for m,cover in d['coverage'].items():
            for b,covered in cover.items():
                if not covered:failures.append(dict(system=d['system'],profile=d['profile'],seed=d['seed'],mode=m,bank=b))
        fits=[d['model']]+[f['model'] for f in d['refits']]
        extreme.append(dict(system=d['system'],profile=d['profile'],seed=d['seed'],anchor_rho=d['model']['u'][1][0],extreme_fits=sum(abs(mm['u'][1][0])>.95 for mm in fits),fit_warnings=len(d['warnings'])))
    noise={};gs={};ls={}
    for profile in ('iid','positive','negative'):
        rr=[d for d in survey if d['profile']==profile];noise[profile]=dict(n=len(rr),sigma_ratio=stats([d['sigma_ratio']['u'] for d in rr]),rho_error=stats([d['rho_error']['u'] for d in rr]),near_unit=sum(abs(d['model']['u'][1][0])>.95 for d in rr),statuses=counts(d['fit_info']['u']['status'] for d in rr),score_ratio={b:stats([d['scores'][b]['fitted']/d['scores'][b]['oracle_noise'] for d in rr]) for b in ('local','broad')})
        rr=[d for d in local if d['profile']==profile];ls[profile]=dict(n=len(rr),sigma_ratio=stats([d['sigma_ratio'] for d in rr]),rho_error=stats([d['rho_error'] for d in rr]),near_unit=sum(abs(d['model'][1][0])>.95 for d in rr),statuses=counts(d['info']['status'] for d in rr))
    for cohort in ('archived','fresh'):
        gs[cohort]={}
        for profile in ('iid','positive','negative'):
            rr=[d for d in guard if (d['cohort'],d['profile'])==(cohort,profile)];gs[cohort][profile]=dict(n=len(rr),old_near_unit=sum(abs(d['old_model'][1][0])>.95 for d in rr),new_near_unit=sum(abs(d['model'][1][0])>.95 for d in rr),triggered=sum(d['info']['guard_triggered'] for d in rr),sigma_ratio=stats([d['sigma_ratio'] for d in rr]),rho_error=stats([d['rho_error'] for d in rr]),innovation_sigma_ratio=stats([d['innovation_sigma_ratio'] for d in rr]),statuses=counts(d['info']['status'] for d in rr),likelihood_gains=stats([d['info']['guard_improvement']*d['info']['contrast_count'] for d in rr]))
    out=dict(status='COMPLETE',coverage=summary,reference_geometry=geometry,cutoff_ratios=cutratios,failures=failures,primary_extreme_fits=extreme,survey=noise,guard=gs,local6=ls,retrospective=read(ROOT/'retrospective.json')['summary'],source_sha256=sources('report.py'))
    dump(ROOT/'summary.json',out)
    rows=[]
    for d in data:
        for mode,cover in d['coverage'].items():
            for bank,ok in cover.items():rows.append(dict(system=d['system'],profile=d['profile'],seed=d['seed'],mode=mode,bank=bank,covered=ok))
    with (ROOT/'coverage.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    fig,grid=plt.subplots(2,2,figsize=(12,9),layout='constrained');axs=grid.ravel();profiles=['iid','positive','negative'];x=np.arange(3)
    for ax,cohort,title in [(axs[0],'archived','Same 96 fields'),(axs[1],'fresh','Independent guard test: 48 fields')]:
        old=[np.median([d['old_sigma_ratio'] for d in guard if d['cohort']==cohort and d['profile']==p]) for p in profiles];new=[gs[cohort][p]['sigma_ratio']['median'] for p in profiles];ax.bar(x-.18,old,.36,label='all-start refinement');ax.bar(x+.18,new,.36,label='conditional profile starts');ax.axhline(1,color='black',ls='--',lw=1);ax.set(xticks=x,xticklabels=profiles,yscale='log',ylabel='Median estimated / true noise amplitude',title=title);ax.legend(fontsize=8)
    info=read(ROOT/'contrast_information.json')['rows'];ax=axs[2]
    for degree,label in [(-1,'Signal known'),(1,'Local linear removal'),(5,'Local quintic removal')]:
        r=[q for q in info if q['degree_removed']==degree];ax.plot([q['rho'] for q in r],[q['local_cramer_rao_rho'] for q in r],'o-',label=label)
    ax.set(xlabel='True noise correlation',ylabel='Local Cramer-Rao scale for rho',yscale='log',title='Detrending loses correlation information');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axs[3];ss=[r for r in read(ROOT/'design_screen.json')['rows'] if r['bank'] in ('local','broad','full_k4')];ax.bar(range(3),[r['score'] for r in ss],color=['#777777','#4477aa','#228833']);ax.set(xticks=range(3),xticklabels=[r['bank'] for r in ss],ylabel='Lower quartile of log(1 + weakest signal/noise)',title='Frozen KdV design proxy');ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    coverage_text='\n'.join(f'{name}/{p}: {summary[name][p]}' for name in summary for p in summary[name])
    geometry_text='\n'.join(f'{name}/{p}: {geometry[name][p]}' for name in geometry for p in geometry[name])
    guard_text='\n'.join(f"{p}: n={gs['fresh'][p]['n']}, near-unit {gs['fresh'][p]['old_near_unit']} -> {gs['fresh'][p]['new_near_unit']}, sigma/truth {gs['fresh'][p]['sigma_ratio']}, rho error {gs['fresh'][p]['rho_error']}, statuses {gs['fresh'][p]['statuses']}" for p in profiles)
    local_text='\n'.join(f"{p}: n={ls[p]['n']}, near-unit={ls[p]['near_unit']}, sigma/truth {ls[p]['sigma_ratio']}, rho error {ls[p]['rho_error']}, statuses {ls[p]['statuses']}" for p in profiles)
    changed=sum(d['coverage']['studentized'][b]!=d['coverage']['fixed_metric'][b] for d in data for b in d['coverage']['studentized'])
    lines=['Solver-free nuisance normalization and weak-direction research','=============================================================', '',
    'Frozen primary comparison:18 fresh datasets,128 refits each, MC16.',
    'Both methods correct moments using refitted noise. Studentized normalization',
    'uses each replicate covariance; fixed_metric uses observed covariance.',
    'This isolates normalization randomness; it is NOT confidence-set propagation',
    'over noise parameters. All cutoffs were frozen before clean-data evaluation.', '',
    '1. Coverage and precision (counts out of3 for every system/profile)',
    coverage_text, f'Paired normalization coverage disagreements: {changed}.',
    'Reference coefficient geometry:',geometry_text,
    'Coverage is full-library weak-error containment, not exact-equation recovery.',
    'No bootstrap draw was removed. All warnings are retained in result files.',
    'Retrospective72-data comparison is exploratory and has different estimator',
    'versions in its69 historical and3 repaired-ODE cohorts; do not pool them.',
    str(out['retrospective']),'',
    '2. Noise identification and optimizer safeguards',
    'In96 independent ODE fields, the repaired temporal fitter still produced',
    f"near-unit estimates: {[ (p,noise[p]['near_unit'],noise[p]['n']) for p in profiles ]}.",
    'Profile checks recovered better moderate-rho optima in the three extreme',
    'positive-correlation cases inspected. Polishing the original starts was not',
    'sufficient: new basins must also be searched. The IID112 moderate-rho case',
    'was different: its old fit had slightly BETTER likelihood than the lower-rho',
    'profile solution (difference0.475 in -2loglik). Not every biased estimate is',
    'an optimizer failure.',
    'The new guard triggers extra fixed-rho starts at[-.8,-.4,0,.4,.8] only if',
    '|rho|>.95 or fit status is non-ok, then releases rho. It retains the old fit',
    'as a candidate and selects the best exact objective. The trigger is not a cap.',
    'Fixed-rho intermediate records store the full gradient; their unconstrained',
    'status may be nonstationary by construction. Only released fits compete.',
    'Independent48-field guard evaluation (32 positive,8 IID,8 negative):',
    guard_text,
    'The guard does not remove moderate amplitude/correlation bias. It has been',
    'tested for noise fitting here; full bootstrap uncertainty WITH this new guard',
    'is not validated. The original18-case run was preserved rather than replaced.', '',
    '3. Existing local-polynomial alternative (exploratory96-field comparison)',
    local_text,
    'Local6 is the unchanged Gaussian-consistent robust estimator after removing',
    'quintic polynomials in short blocks. This compares signal/noise assumptions.',
    'It is not newly tuned, not a validated global replacement, and not calibrated',
    'as part of the present fresh bootstrap procedure.',
    'Gaussian information diagnostic: with320 points and true rho=.4, the local',
    'Cramer-Rao scale for rho rises0.05128->0.31399 after removing local quintic',
    'polynomials. The inferred log-sigma/rho parameter correlation becomes0.9841.',
    'For rho=-.4 the same contrast scale is0.08362. This asymmetry is consistent',
    'with losing the slow noise components when removing a smooth signal.',
    'The calculation retains cross-block covariance and assumes the signal lies',
    'exactly in the removed polynomial space. These are local information bounds,',
    'not confidence intervals or predictions for biased/boundary estimators.',
    'Independent Gaussian score simulations verified the Fisher matrices within',
    '1.5% relative error. This diagnostic explains a limitation of detrending;',
    'it does not prove the cause of every temporal AR2-model bias.', '',
    '4. Weak-direction design',
    'The fixed12-bank menu was ranked by weak coefficient directions across every',
    'eligible fixed candidate support, projecting only its deterministic feature',
    'columns. Purely deterministic candidates were retained in equation inference.',
    'The selected bank was full_k4. Its proxy score1.05783 exceeds broad0.90009.',
    'The reference support minimum pilot signal/noise eigenvalue improves only',
    '1.82278->1.89319; the main proxy gain comes from other, more weakly identified',
    'supports. This helps explain why a better aggregate criterion can leave the',
    'main coefficient ambiguity unresolved. Reference labels/exact coefficients',
    'did not enter selection; the benchmark candidate panel itself is fixed and',
    'reference-derived, so these tests do not establish adaptive EPDE-search validity.', '',
    '5. Limits and decision',
    'Keep dyad provisional and retain separate weak accuracy and precision checks.',
    'Numerical optimizer repair and statistical noise identification are distinct.',
    'Do not interpret apparent coverage with unstable nuisance fits as a solution:',
    'very large uncertainty regions can cover well while conveying little precision.',
    'Conditional profile starts deserve broader end-to-end tests; moderate bias',
    'requires comparing the signal/noise model assumptions and nuisance uncertainty.',
    'Three fully bootstrapped datasets per case cannot certify95% coverage.',
    'Constants/valid identities are retained. No solver or production change.',
    'All raw records, frozen outcomes, numerical audits and manifests are saved.', '']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('SUMMARY',summary,'GUARD',gs['fresh'],'LOCAL',ls,flush=True)

if __name__=='__main__':main()
