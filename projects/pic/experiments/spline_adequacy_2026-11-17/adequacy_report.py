from adequacy_study import *
from collections import Counter
from scipy.stats import beta
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRIMARY='adj_cv_a01'
METHODS=('cv','fixed32',*[kind+'_'+m for kind in ('own','adj') for m in A.METHODS])

def stats(x):
    if not x:return dict(n=0)
    a=np.asarray(x);return dict(n=len(a),median=float(np.median(a)),mean=float(np.mean(a)),max=float(max(a)),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)))

def event(k,n):return dict(count=k,n=n,fraction=k/n,binomial95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.])

def summarize(rows):
    out={};n=len(rows)
    for m in METHODS:
        rr=[r['methods'][m] for r in rows];valid=[r for r in rr if r['status']=='ok'];returned=[r for r in rr if r['selected_p'] is not None];covered=sum(r['coverage'] for r in valid);out[m]=dict(n=n,valid=len(valid),abstentions=sum(r['selected_p'] is None for r in rr),fit_warnings=sum(r['selected_p'] is not None and r['status']!='ok' for r in rr),usable_covered=event(covered,n),coverage_among_valid=event(covered,len(valid)) if valid else None,diagnostic_covered_returned=sum(r['coverage'] for r in returned),unbounded_returned=sum(r['unbounded_certified'] for r in returned),dimensions=dict(Counter(r['selected_p'] for r in valid)),valid_sigma_absolute_error=stats([abs(r['sigma_ratio']-1) for r in valid]),valid_rho_absolute_error=stats([abs(r['rho_error']) for r in valid]),all_returned_sigma_ratio=stats([r['sigma_ratio'] for r in returned]))
    return out

def main():
    audit=read(ROOT/'audit.json');assert audit['status']=='PASS';rows=[read(p) for p in sorted((ROOT/'results').glob('*.json'))];assert len(rows)==1832;groups={};cohorts={c:summarize([r for r in rows if r['job']['cohort']==c]) for c in ('primary','stress','analytic','control')};csvrows=[]
    for key in sorted({(r['job']['cohort'],r['job']['name'],r['job']['profile']) for r in rows}):
        rr=[r for r in rows if (r['job']['cohort'],r['job']['name'],r['job']['profile'])==key];name='/'.join(key);groups[name]=summarize(rr)
        for m,d in groups[name].items():csvrows.append(dict(group=name,method=m,n=d['n'],valid=d['valid'],covered=d['usable_covered']['count'],abstentions=d['abstentions'],fit_warnings=d['fit_warnings'],unbounded_returned=d['unbounded_returned'],median_sigma_abs_error=d['valid_sigma_absolute_error'].get('median'),max_returned_sigma_ratio=d['all_returned_sigma_ratio'].get('max')))
    analytic={str(f):summarize([r for r in rows if r['job']['cohort']=='analytic' and r['job']['frequency']==f]) for f in FREQUENCIES};controls={};abstentions=[];changes=[];failures=[]
    for r in rows:
        d=read(ROOT/'frozen'/(r['key']+'.json'));j=r['job'];m=r['methods'][PRIMARY]
        if j['cohort']=='control':
            for kind in ('own','adj'):
                for s in d[kind+'_screens']:
                    key='/'.join((j['name'],j['profile'],kind,'p'+str(s['p'])));controls.setdefault(key,[]).append(s['pvalue'])
        if m['selected_p'] is None:
            ss=d['adj_screens'];abstentions.append(dict(key=r['key'],job=j,ok_fit_dimensions=[s['p'] for s in ss if s['fit_status']=='ok'],boundary_pass_dimensions=[s['p'] for s in ss if s['boundary_excluded']],adequacy_pass_dimensions=[s['p'] for s in ss if s['pvalue']>=.01],screens=ss))
        if m['selected_p']!=r['methods']['cv']['selected_p']:changes.append(dict(key=r['key'],cohort=j['cohort'],cv=r['methods']['cv'],primary=m))
        if m['status']=='ok':failures.append(dict(key=r['key'],job=j,primary=m,cv=r['methods']['cv'],selected_screen=next(s for s in d['adj_screens'] if s['p']==m['selected_p'])))
    controls={k:dict(n=len(v),reject_a01=sum(x<.01 for x in v),reject_a05=sum(x<.05 for x in v)) for k,v in controls.items()};failures=sorted(failures,key=lambda x:abs(np.log(x['primary']['sigma_ratio'])),reverse=True)[:24];fine=[r for r in audit['rows'] if r['fine_grid_improvements']]
    failure=read(ROOT/'failure_analysis.json');assert failure['status']=='PASS';boot=read(ROOT/'bootstrap_summary.json');assert boot['status']=='COMPLETE'
    out=dict(status='COMPLETE',primary_method=PRIMARY,cohorts=cohorts,groups=groups,analytic_frequency=analytic,control_adequacy_rejections=controls,primary_abstentions=abstentions,primary_changes=changes,largest_primary_errors=failures,fine_grid_improvements=fine,failure_diagnostic=failure['groups'],bootstrap_followup={k:boot[k] for k in ('general','profiles','target_conditional','target_scan_count','target_eligible_count')},source_sha256=src('adequacy_report.py','protocol.txt'),input_sha256={str(ROOT/(n+'.json')):sha(ROOT/(n+'.json')) for n in ('audit','failure_analysis','bootstrap_summary')});dump(ROOT/'summary.json',out)
    with (ROOT/'method_results.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    fig,axs=plt.subplots(2,2,figsize=(12.8,9.4),layout='constrained');x=np.arange(6)
    for k,m in enumerate(('cv',PRIMARY)):
        d=cohorts['stress'][m];vals=[d['dimensions'].get(p,0) for p in G.SIZES];axs[0,0].bar(x+(k-.5)*.35,vals,.34,label='CV' if m=='cv' else 'Screened CV')
    axs[0,0].set(xticks=x,xticklabels=G.SIZES,xlabel='Selected spline dimension',ylabel='Resolved cases',title='Fresh ODE stress: 1,024 positive-noise fields');axs[0,0].legend()
    ms=('cv','own_cv_a01',PRIMARY,'adj_smallest_a01','adj_cv_a05');x=np.arange(len(ms));covered=[cohorts['stress'][m]['usable_covered']['fraction'] for m in ms];absent=[cohorts['stress'][m]['abstentions']/1024 for m in ms];axs[0,1].bar(x,covered,label='Resolved and truth covered');axs[0,1].bar(x,absent,bottom=covered,label='Abstained',color='#cc8844');axs[0,1].axhline(.975,c='black',ls='--',lw=1);axs[0,1].set(xticks=x,xticklabels=['CV','Own noise\nCV .01','Adjacent\nCV .01','Adjacent\nsmallest .01','Adjacent\nCV .05'],ylim=(0,1.08),ylabel='Fraction of all 1,024 cases',title='Uncertainty after model selection');axs[0,1].tick_params(axis='x',labelsize=8);axs[0,1].legend(loc='lower left',fontsize=8)
    x=np.array(FREQUENCIES)
    for m,label in [('cv','CV'),('fixed32','Fixed 32'),(PRIMARY,'Screened CV')]:
        ys=[analytic[str(f)][m]['valid_sigma_absolute_error'].get('median',float('nan')) for f in FREQUENCIES];axs[1,0].plot(x,ys,'o-',label=label)
    axs[1,0].set(yscale='log',xticks=x,xlabel='Frequency parameter of analytic signal',ylabel='Median relative amplitude error, valid cases',title='Independent analytic signals: 48 fields per frequency');axs[1,0].legend(fontsize=8)
    xx=np.arange(len(FREQUENCIES));vals=[analytic[str(f)][PRIMARY]['usable_covered']['fraction'] for f in FREQUENCIES];ab=[analytic[str(f)][PRIMARY]['abstentions']/48 for f in FREQUENCIES];axs[1,1].bar(xx,vals,label='Resolved and truth covered');axs[1,1].bar(xx,ab,bottom=vals,label='Abstained',color='#cc8844');axs[1,1].plot(xx,[analytic[str(f)]['cv']['usable_covered']['fraction'] for f in FREQUENCIES],'ko--',label='Original CV coverage');axs[1,1].set(xticks=xx,xticklabels=FREQUENCIES,ylim=(0,1.08),xlabel='Frequency parameter',ylabel='Fraction of all 48 cases',title='Primary rule: full denominator retained');axs[1,1].legend(loc='lower left',fontsize=8)
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    s=cohorts['stress'];a=cohorts['analytic'];lines=['Automatic spline adequacy and noise-information screening','=========================================================','',
    'The primary rule eliminated boundary outputs on1024 fresh positive-noise',
    'ODE fields and improved usable nominal noise coverage911 ->949/1024.',
    'It is not ready for production: nonrejection of extra signal does not',
    'establish negligible signal bias. Bootstrap calibration subsequently',
    'covered48/48 general cases but returned the entire parameter space in10.',
    'All8 targeted fresh p24 cases also produced whole-space regions.','',
    'This experiment tests a frozen automatic selection rule, including',
    'abstention. No method is promoted based on these results. Noise-region',
    'coverage after selection is empirical: the nominal chi-square cutoff',
    'does not account for selection. Complete denominators are retained.','',
    '1. Frozen rule',
    'Candidates p8,16,24,32,48,64. Compare each mean space with its union with',
    'the next richer spline (p96 only diagnoses p64). A GLS F statistic tests',
    'additional mean structure. Correlation is estimated, so its p-value is',
    'a diagnostic rather than an exact level-alpha test. Own-noise and next-',
    'richer-noise versions are compared. Candidate noise likelihood must also',
    'exclude the previously established rho->1,sigma->infinity escaping path.',
    'Excluding this path is not a proof of whole-region compactness. Fits and',
    'screening-noise fits must be numerically resolved. All candidate fits stay',
    'in the record, including rejected and boundary fits. If none is eligible,',
    'the method abstains instead of silently substituting a spline.',
    'Eight predeclared rules: own/adjacent noise x CV/smallest ordering x',
    'alpha.01/.05. Primary rule is adjacent-noise CV alpha.01. Unchanged CV',
    'and fixed32 are comparators. No threshold is tuned on validation results.','',
    '2. Independent observations',
    '1832 fresh noisy fields:264 benchmark fields on ODE/VDP/Duffing,1024 ODE',
    'positive-noise stress realizations,288 analytic oscillatory signals,',
    '256 exact-model controls. Analytic frequencies1,2,4,6,8,12 vary required',
    'mean complexity; each uses16 realizations for iid,+.4,-.4 noise. These',
    'signals are evaluated directly, with no ODE/PDE integration. Controls',
    'are zero-mean AR1 with n320/1001 and rho=-.4,0,.4,.8. All fit and selection',
    'records were frozen before evaluating truth. Constants remain available.','',
    '3. ODE positive-noise stress results (n1024)',
    '\n'.join(m+': '+json.dumps(d) for m,d in s.items()),'',
    '4. Analytic signal transfer (not just more noise on the same ODE)',
    '\n'.join('frequency '+str(f)+': '+json.dumps({m:analytic[str(f)][m] for m in ('cv','fixed32',PRIMARY)}) for f in FREQUENCIES),'',
    'The benchmark, controlled and threshold/order ablations are in summary.json',
    'and method_results.csv. Error summaries among valid outputs do not count',
    'abstentions; their counts are therefore always shown separately. Boundary',
    'fits are retained in all-returned amplitude tails and warning counts.',
    'Coverage among valid outputs is conditional on passing the data-dependent',
    'screens and cannot alone establish unconditional confidence coverage.','',
    '5. Why failures remain',
    'On the primary ODE stress sample the rule chose p24 in42 cases; only5',
    'nominal regions covered truth. Their median sigma error was+22.1% and',
    'rho error+.2004, despite adequacy p-values as high as.638. Nonrejection',
    'of a test for extra mean structure is not proof of negligible bias.',
    'Posthoc grouping by selected dimension:',json.dumps(failure['groups']['adj_cv_a01'],indent=2),
    'The p24 approximation alone adds14.26% to expected variance at known',
    'true rho; p32 adds0.359%. Selection and joint noise fitting amplify the',
    'practical error. This diagnostic uses truth only after all choices freeze.','',
    '6. Independent selection-aware bootstrap follow-up',
    '48 new ODE fields,128 complete refits each. All observed choices werep32.',
    'Then scan256 further positive fields, freeze fits, select the first8 p24',
    'cases by observed dimension only, and bootstrap them. This second cohort',
    'is a conditional stress test, not unconditional population coverage.',
    'An abstaining bootstrap draw retains infinite statistic mass. Rank126/128',
    'becomes infinite when at least3 draws abstain. That gives a whole-space',
    'noise region, which must not be counted as useful precision.',
    json.dumps({k:boot[k] for k in ('general','profiles','target_conditional','target_scan_count','target_eligible_count')},indent=2),
    'See bootstrap_findings.txt and bootstrap_overview.pdf for full details.',
    'Calibrating the complete selector exposed its instability under the fitted',
    'model; it did not make the biased p24 uncertainty useful.','',
    '7. Retained failure records',
    'Primary abstentions: '+str(len(abstentions)),
    'Primary selection changes: '+str(len(changes)),
    'Largest primary amplitude errors (all are retained):',
    json.dumps(failures,indent=2),'',
    'No fine-grid improvement is silently substituted.129-grid audit warning',
    'records: '+str(len(fine)),
    'All fields/hashes and selection decisions were replayed; selected true-',
    'parameter likelihoods and boundary limits use independent dense contrasts.',
    'GLS adequacy calculations were checked densely on representative fields',
    'and every abstention. Prior artifact manifests remain unchanged.',
    'This round evaluates noise inference, not downstream coefficient regions',
    'or adaptive equation discovery. No production, dyad, KdV or full15-system',
    'replacement is claimed. The next change must address approximation bias',
    'in mean/noise selection, not merely tighten a p-value threshold or inflate',
    'a confidence cutoff. In particular, accepting a smaller mean because an',
    'extra-structure test is nonsignificant is insufficient. Quantified bounds',
    'on residual mean error, or justified signal regularity, are candidates',
    'for a subsequent frozen comparison, not established fixes from this study.','']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8');print('REPORT COMPLETE',json.dumps(dict(stress={m:s[m] for m in ('cv',PRIMARY,'adj_smallest_a01')},analytic_primary=a[PRIMARY],abstentions=len(abstentions),fine_grid_warnings=len(fine))),flush=True)

if __name__=='__main__':main()
