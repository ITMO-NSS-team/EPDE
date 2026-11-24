from compat_study import *
from collections import Counter
from scipy.stats import beta
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stats(x):
    x=np.asarray(x,dtype=float);return dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),min=float(min(x)),max=float(max(x)),q95=float(np.quantile(x,.95)),q99=float(np.quantile(x,.99)))

def count(x):
    n=len(x);k=sum(bool(v) for v in x);return dict(n=n,count=k,fraction=k/n,binomial95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.])

def summarize(rows):
    return dict(n=len(rows),selection_changes=sum(r['selected_p']['cv']!=r['selected_p']['compatible'] for r in rows),methods={m:dict(sigma_ratio=stats([r['errors'][m]['sigma_ratio'] for r in rows]),absolute_sigma_error=stats([abs(r['errors'][m]['sigma_ratio']-1) for r in rows]),absolute_rho_error=stats([abs(r['errors'][m]['rho_error']) for r in rows]),selected_status=dict(Counter(r['selected_status'][m] for r in rows)),dimensions=dict(Counter(r['selected_p'][m] for r in rows))) for m in ('cv','compatible','rich')},rich_region={m:dict(coverage=count([r['coverage'][m] for r in rows]),unbounded=count([r['unbounded_certified'][m] for r in rows])) for m in ('nominal','grid_calibrated')})

def main():
    audit=read(ROOT/'audit.json');assert audit['status']=='PASS';diag=read(ROOT/'dimension_diagnostic.json');assert diag['status']=='PASS';rr=[read(p) for p in sorted((ROOT/'signal_results').glob('*.json'))];assert len(rr)==1288;cal=read(ROOT/'calibration_frozen.json');controls=[d for p in sorted((ROOT/'control_validation').glob('*.json')) if (d:=read(p))];summary={};table=[]
    for name in ('ode','vdp','duffing'):
        summary[name]={}
        for profile in PROFILES:
            a=[r for r in rr if r['cohort']=='primary' and r['name']==name and r['profile']==profile];s=summarize(a);summary[name][profile]=s
            for m,z in s['methods'].items():table.append(dict(system=name,profile=profile,method=m,n=len(a),median_sigma_ratio=z['sigma_ratio']['median'],median_absolute_sigma_error=z['absolute_sigma_error']['median'],q99_sigma_ratio=z['sigma_ratio']['q99'],max_sigma_ratio=z['sigma_ratio']['max'],selected_warnings=len(a)-z['selected_status'].get('ok',0)))
    stress=summarize([r for r in rr if r['cohort']=='stress']);cv={}
    for n in (320,1001):
        cv[str(n)]={}
        for k,rho in enumerate(RHOS):
            a=[r for d in controls if d['n']==n and d['rho_index']==k for r in d['rows']];assert len(a)==64;cv[str(n)][str(rho)]=dict(methods={m:dict(coverage=count([r['lr']<=cut for r in a]),unbounded=count([r['limit_lr']<cut-1e-6 for r in a])) for m,cut in [('nominal',H.CUTOFF),('grid_calibrated',cal['cutoffs'][str(n)])]},selected_status=dict(Counter(r['fit']['status'] for r in a)))
    dimensions={}
    for cohort,name,profile in [('control','320','positive'),('control','1001','positive')]+[('primary',n,p) for n in ('ode','vdp','duffing') for p in PROFILES]+[('stress','ode','positive')]:
        rows=[r for r in diag['rows'] if r['cohort']==cohort and r['name']==name and r['profile']==profile];key='/'.join((cohort,name,profile));dimensions[key]={}
        for p in (32,48,64):
            a=[next(x for x in r['rows'] if x['p']==p) for r in rows];dimensions[key][str(p)]=dict(coverage=count([r['nominal_covered'] for r in a]),unbounded=count([r['unbounded_certified'] for r in a]),sigma_absolute_error=stats([abs(r['sigma_ratio']-1) for r in a]),selected_status=dict(Counter(r['fit_status'] for r in a)))
    changes=[r for r in rr if r['selected_p']['cv']!=r['selected_p']['compatible']];warnings=[r for r in rr if any(s!='ok' for s in r['selected_status'].values())];fine=[r for r in audit['rows'] if r['fine_grid_improvements']];out=dict(status='COMPLETE',primary=summary,stress=stress,controlled_validation=cv,calibration=cal['cutoffs'],changes=changes,selected_warnings=warnings,fine_grid_improvements=fine,posthoc_dimensions=dimensions,source_sha256=src('compat_report.py','protocol.txt'),input_sha256={str(ROOT/'audit.json'):sha(ROOT/'audit.json'),str(ROOT/'dimension_diagnostic.json'):sha(ROOT/'dimension_diagnostic.json')});dump(ROOT/'summary.json',out)
    with (ROOT/'point_accuracy.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
    fig,axs=plt.subplots(2,2,figsize=(12.4,9),layout='constrained');x=np.arange(6)
    for j,n in enumerate((320,1001)):
        vals=[cv[str(n)][str(r)]['methods']['nominal']['unbounded']['fraction'] for r in RHOS];axs[0,0].bar(x+(j-.5)*.35,vals,.34,label=f'n={n}')
    axs[0,0].set(xticks=x,xticklabels=RHOS,ylabel='Fraction with no finite upper noise bound',title='Controlled model: 64 fields per configuration');axs[0,0].legend()
    x=np.arange(3)
    for j,mode in enumerate(('nominal','grid_calibrated')):
        ys=[stress['rich_region'][mode][metric]['fraction'] for metric in ('coverage','unbounded')];axs[0,1].bar(np.arange(2)+(j-.5)*.35,ys,.34,label=mode.replace('_',' '))
        for k,v in enumerate(ys):axs[0,1].text(k+(j-.5)*.35,v+.02,f'{v:.1%}',ha='center',fontsize=9)
    axs[0,1].axhline(.975,c='black',ls='--',lw=1);axs[0,1].set(xticks=[0,1],xticklabels=['Truth covered','Unbounded amplitude'],ylim=(0,1.14),title='1,024 fresh positive-noise ODE fields');axs[0,1].legend(loc='lower left',fontsize=8)
    for j,metric in enumerate(('coverage','unbounded')):
        ys=[dimensions['stress/ode/positive'][str(p)][metric]['fraction'] for p in (32,48,64)];axs[1,0].bar(x+(j-.5)*.35,ys,.34,label=metric)
        for k,v in enumerate(ys):axs[1,0].text(k+(j-.5)*.35,v+.02,f'{v:.1%}',ha='center',fontsize=8)
    axs[1,0].set(xticks=x,xticklabels=[32,48,64],xlabel='Fixed spline dimension',ylim=(0,1.13),title='Posthoc dimension diagnostic: same 1,024 fields');axs[1,0].legend(loc='lower left',fontsize=8)
    rows=[r for r in rr if r['cohort']=='stress'];a=[r['errors']['cv']['sigma_ratio'] for r in rows];b=[r['errors']['compatible']['sigma_ratio'] for r in rows];axs[1,1].scatter(a,b,s=12,alpha=.45);axs[1,1].plot([.7,1000],[.7,1000],c='black',ls='--',lw=1);axs[1,1].set(xscale='log',yscale='log',xlabel='Original CV: estimated / true amplitude',ylabel='Compatibility rule: estimated / true amplitude',title='Rare boundary failures are only partly repaired')
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    s=stress;d=dimensions['stress/ode/positive'];lines=['Common-contrast compatibility: result and information loss','=========================================================','',
    'The p64 compatibility rule is not a complete repair. It fixed both archived',
    'development failures, but only one of two boundary failures in1024 fresh',
    'positive-noise ODE fields. A very flexible mean also makes noise amplitude',
    'unbounded in many otherwise well-calibrated likelihood regions.','',
    '1. What was frozen before fresh validation',
    'Original conditional-CV estimator, compatible-CV estimator, and fixed-p64',
    'noise estimator. All candidate noise models are compared on the SAME p64',
    'error-contrast space. Compatibility means LR<=chi-square2(.975); select the',
    'best original CV score among compatible models. No parameter is clipped,',
    'no truth is used for selection, and no field or selected warning is dropped.',
    'This is separate from the p64 likelihood-region calibration comparison.',
    '264 fresh signals:192 ODE,48 VDP,24 Duffing; then1024 positive ODE stress',
    'fields. Controlled calibration1536 fields and independent validation768',
    'fields span n320/1001 and rho=-.8,-.4,0,.4,.8,.95. Zero-mean Gaussian AR1',
    'simulation needs no equation solver. Source and RNG details are frozen.',
    'Calibration takes the maximum of six rank126/128 LR quantiles for each n.',
    'These are finite-grid cutoffs, NOT uniform-in-rho confidence guarantees:',
    str(cal['cutoffs']),'',
    '2. Fresh results for the compatibility rule',
    'The rule changed no selection in the264-field primary cohort. In the1024',
    'positive-noise stress fields it changed five selections. One catastrophic',
    'boundary fit was repaired(seed1283), another survived(seed1334), and one',
    'ordinary change worsened amplitude accuracy(seed1055). Median absolute',
    'amplitude error is essentially unchanged. Complete results retain tails.',
    'Stress summary:',json.dumps(s,indent=2),'',
    'Remaining boundary seed1334: original and compatible sigma/truth784.0,',
    'rho0.999999. Its rich-space limiting LR6.1008 is below cutoff7.3778, so',
    'this safeguard has insufficient evidence to reject the divergent model.',
    'Fixed-p64 estimation itself also hit a boundary at seed1456, while the',
    'ordinary CV p32 estimate stayed finite. More flexible is not always safer.','',
    '3. Coverage and finite precision are different requirements',
    f"Nominal p64 region: truth covered {s['rich_region']['nominal']['coverage']['count']}/1024; amplitude unbounded in {s['rich_region']['nominal']['unbounded']['count']}/1024.",
    f"Grid-calibrated p64 region: truth covered {s['rich_region']['grid_calibrated']['coverage']['count']}/1024; amplitude unbounded in {s['rich_region']['grid_calibrated']['unbounded']['count']}/1024.",
    'In exact controlled data with n320,rho.4, the nominal region was amplitude-',
    'unbounded27/64 times. This cannot be blamed on the ODE signal, an equation',
    'solver, or an incorrect simulation noise family. The rich mean removes',
    'low-frequency information that would otherwise distinguish correlated',
    'measurement noise from deterministic signal. It is a property of this',
    'mean/noise model, not a proof that the original data cannot identify noise',
    'under stronger, justified signal assumptions.','',
    'The unboundedness test is analytic: as rho->1 with sigma^2(1-rho)->eta,',
    'contrast covariance tends to eta*K, K=-Q_perp*|i-j|*Q_perp. Constants are',
    'in the spline space, so the divergent common offset disappears. Profiling',
    'eta gives a limiting likelihood. If its LR is strictly below the cutoff,',
    'there is an escaping sequence with sigma->infinity inside the region.',
    'This is not extrapolation from a finite rho grid. A failed sufficient',
    'test is not reported as a certified compact region. Neither does an',
    'unbounded noise region alone prove every coefficient region unbounded.','',
    '4. Posthoc diagnosis of mean complexity (not new-method validation)',
    'After seeing p64 unboundedness, the SAME frozen fields were compared at',
    'fixed p32,p48,p64. These comparisons did not alter any primary selection',
    'or cutoff. They do not validate an adaptive dimension-selection rule.',
    json.dumps(d,indent=2),
    'Controlled and all-signal dimension comparisons are in summary.json.',
    'A useful next rule must check both signal adequacy and retained noise',
    'information. Always using the largest spline is not a sufficient repair.',
    'Fixed p32 is a diagnostic benchmark here; selecting it because these ODE',
    'results look good would require independent validation on other signals.','',
    '5. Audit and scope',
    'Every raw field was regenerated. Every spline selection was replayed.',
    'All true-parameter likelihoods were checked by dense orthogonal contrasts;',
    'selected likelihoods also use independent whitened QR and dense checks.',
    'All rich fits were stress-tested with129 rho grid points. The analytic',
    'boundary limit was checked independently through first differences and',
    'a singular-value decomposition of the differentiated spline design.',
    'Fine-grid improvements beyond tolerance: '+str(len(fine)),
    'Numerical checks passing does not imply statistical validity. Boundary',
    'warnings remain in the results. All earlier artifact manifests are checked.',
    'No production edits, solver calls, removed constants, or lost identities.',
    'This round studies nuisance inference; it does not rerun weak coefficient',
    'propagation, validate adaptive equation search, or settle the dyad/KdV issue.','']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8');print('REPORT COMPLETE',dict(stress_changes=s['selection_changes'],coverage=s['rich_region'],fine_grid_warnings=len(fine),dimensions=d),flush=True)

if __name__=='__main__':main()
