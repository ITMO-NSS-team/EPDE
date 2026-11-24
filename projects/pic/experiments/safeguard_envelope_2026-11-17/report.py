from study_context import *
from collections import Counter
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stats(x):return dict(min=float(min(x)),median=float(np.median(x)),max=float(max(x)))
def counts(x):return dict(Counter(x))
def width(reg):return np.array(reg['upper'])-reg['lower']


def main():
    data=[read(p) for p in sorted((ROOT/'results').glob('*.json'))];assert len(data)==12
    data.sort(key=lambda d:((0 if d['cohort']=='paired_previous' else {'iid':1,'positive':2,'negative':3}[d['profile']]),d['seed']))
    point={d['key']:read(ROOT/'point_frozen'/(d['key']+'.json')) for d in data};env={d['key']:read(ROOT/'envelope_frozen'/(d['key']+'.json')) for d in data};union={d['key']:read(ROOT/'union_diagnostic'/(d['key']+'.json')) for d in data}
    fitinfo=[i for d in point.values() for i in list(d['fit_info'].values())+[i for r in d['refits'] for i in r['info'].values()]]
    coverage={};geometry={};nuisance={};noise={}
    for group in ('paired_previous','fresh_iid','fresh_positive','fresh_negative'):
        rr=[d for d in data if d['cohort']=='paired_previous'] if group=='paired_previous' else [d for d in data if d['cohort']=='fresh' and d['profile']==group[6:]]
        coverage[group]={m:{b:sum(d['coverage'][m][b] for d in rr) for b in ('local','broad','joint')} for m in ('studentized','fixed_metric','envelope')}
        geometry[group]={m:counts(c['geometry'] for d in rr for c in (d['envelope_geometry'] if m=='envelope' else d['point_geometry'][m])) for m in ('studentized','fixed_metric','envelope')}
        geometry[group]['scenario_union']=counts(union[d['key']]['geometry'] for d in rr)
        nuisance[group]=dict(n=len(rr),truth_in_coordinate_box=sum(d['nuisance_truth_in_basic_coordinate_box'] for d in rr),truth_in_convex_hull=sum(d['nuisance_truth_in_basic_convex_hull'] for d in rr))
        noise[group]=dict(sigma_ratio=stats([d['sigma_truth_ratio'] for d in rr]),rho_error=stats([d['rho_error'] for d in rr]))
    widths=[];cases=[];worst=[]
    for d in data:
        key=d['key'];u=union[key];ec=d['envelope_geometry'][0];base=d['point_geometry']['studentized'][0]['outer_region'];uw=width(u['outer_box']) if u['geometry']=='bounded_outer' else None;ew=width(ec['outer_region']) if ec['geometry']=='bounded_outer' else None
        for j in range(3):
            widths.append(dict(key=key,coefficient=j,exact=point[key]['candidates'][0]['exact'][j],primary_lower=base['lower'][j],primary_upper=base['upper'][j],envelope_lower=ec['outer_region']['lower'][j] if ew is not None else None,envelope_upper=ec['outer_region']['upper'][j] if ew is not None else None,union_lower=u['outer_box']['lower'][j] if uw is not None else None,union_upper=u['outer_box']['upper'][j] if uw is not None else None,scalar_width_ratio=float(ew[j]/width(base)[j]) if ew is not None else None,union_width_ratio=float(uw[j]/width(base)[j]) if uw is not None else None))
        worstpoint=max(env[key]['points'],key=lambda row:max(v['bound'] for v in row['bounds'].values()));worst.append(dict(key=key,index=worstpoint['index'],model=worstpoint['model'],bounds=worstpoint['bounds'],conditional_cutoff=worstpoint['conditional_cutoff']))
        cases.append(dict(key=key,cohort=d['cohort'],profile=d['profile'],sigma_ratio=d['sigma_truth_ratio'],rho_error=d['rho_error'],studentized_joint_ratio=max(d['actual_scores'].values())/d['point_joint_cutoffs']['studentized'],fixed_joint_ratio=max(d['actual_scores'].values())/d['point_joint_cutoffs']['fixed_metric'],envelope_joint_ratio=max(d['actual_scores'].values())/d['envelope_cutoff'],envelope_cutoff_ratio=d['envelope_cutoff']/max(d['point_joint_cutoffs'].values()),scalar_geometry=ec['geometry'],union_geometry=u['geometry'],truth_in_hull=d['nuisance_truth_in_basic_convex_hull']))
    arithmetic=read(ROOT/'native_arithmetic_checks.json');checks=read(ROOT/'native_fit_checks.json');audit=read(ROOT/'audit.json');assert audit['status']=='PASS'
    out=dict(status='COMPLETE',coverage=coverage,reference_geometry=geometry,noise=noise,nuisance_set_diagnostics=nuisance,fit_diagnostics=dict(total=len(fitinfo),guard_triggers=sum(i['guard_triggered'] for i in fitinfo),guard_triggers_observed=sum(i['guard_triggered'] for p in point.values() for i in p['fit_info'].values()),statuses=counts(i['status'] for i in fitinfo),near_unit=sum(abs(i['rhos'][0])>.95 for i in fitinfo),original_python_polishes=sum(i['native_certification']['original_polish'] for i in fitinfo),max_original_value_discrepancy=max(i['native_certification']['original_value_error'] for i in fitinfo)),cases=cases,scalar_width_ratio_bounded_only=stats([w['scalar_width_ratio'] for w in widths if w['scalar_width_ratio'] is not None]),union_width_ratio=stats([w['union_width_ratio'] for w in widths]),worst_envelope_points=worst,all_reference_noise_covered=all(r['noise_covered'] for d in data for r in d['references']),all_reference_zero_accepted=all(r['zero_accepted'] for d in data for r in d['references']),native_arithmetic_cases=len(arithmetic['rows']),native_complete_fit_cases=len(checks['rows']),audit_totals={k:sum(r[k] for r in audit['rows']) for k in ('original_python_fits','bootstrap_field_hashes','primary_scores','conditional_scores','regions','certificates','loewner_bounds')},source_sha256=sources('report.py'))
    dump(ROOT/'summary.json',out)
    for name,rows in [('case_summary.csv',cases),('coefficient_outer_enclosures.csv',widths)]:
        with (ROOT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained');ax=axs[0,0];paired=[d for d in data if d['cohort']=='paired_previous'];true_sigma=.10301126988150873;x=np.arange(3)
    ax.bar(x-.18,[d['old_comparison']['old_model']['u'][0]/true_sigma for d in paired],.36,label='Previous fit');ax.bar(x+.18,[d['sigma_truth_ratio'] for d in paired],.36,label='Guarded fit');ax.axhline(1,c='k',ls='--',lw=1);ax.set(yscale='log',xticks=x,xticklabels=['+0.4 / '+str(d['seed']) for d in paired],ylabel='Estimated / true noise amplitude',title='Guard fixes the three catastrophic fits');ax.legend(fontsize=9)
    ax=axs[0,1];x=np.arange(12);labels=[('old ' if d['cohort']=='paired_previous' else '')+{'iid':'iid','positive':'+.4','negative':'−.4'}[d['profile']]+' / '+str(d['seed']) for d in data]
    for column,label,offset in [('studentized_joint_ratio','Studentized',-.18),('fixed_joint_ratio','Fixed metric',0),('envelope_joint_ratio','Scalar envelope',.18)]:ax.scatter(x+offset,[r[column] for r in cases],s=24,label=label)
    ax.axhline(1,c='k',ls='--',lw=1);ax.set(xticks=x,xticklabels=labels,ylabel='Observed joint error score / cutoff',title='One fresh miss; envelope covers it');ax.tick_params(axis='x',rotation=65,labelsize=8);ax.legend(fontsize=8)
    ax=axs[1,0]
    for i,d in enumerate(data):
        rr=[w for w in widths if w['key']==d['key']];uv=np.median([w['union_width_ratio'] for w in rr]);ax.scatter(i+.12,uv,c='#228833',s=25,label='Scenario union' if i==0 else None)
        if rr[0]['scalar_width_ratio'] is None:ax.text(i-.12,.97,'unbounded',transform=ax.get_xaxis_transform(),rotation=90,ha='center',va='top',fontsize=8,color='#cc6677')
        else:ax.scatter(i-.12,np.median([w['scalar_width_ratio'] for w in rr]),c='#cc6677',s=25,label='Scalar envelope' if i==1 else None)
    ax.axhline(1,c='k',ls='--',lw=1);ax.set(xticks=x,xticklabels=labels,ylabel='Median outer width / primary outer width',title='Keeping scenarios separate preserves precision');ax.tick_params(axis='x',rotation=65,labelsize=8);ax.legend(fontsize=8)
    ax=axs[1,1];p=read(ROOT/'nuisance_points/ode_positive_seed57.json')['points'];ax.scatter([r['model']['u'][0]/true_sigma for r in p[1:]],[r['model']['u'][1][0] for r in p[1:]],s=12,alpha=.55,label='128 reflected noise scenarios');ax.scatter([p[0]['model']['u'][0]/true_sigma],[p[0]['model']['u'][1][0]],marker='x',s=90,c='k',label='Observed estimate');ax.scatter([1],[.4],marker='*',s=150,c='#cc3311',label='Benchmark truth');ax.set(xlabel='Noise amplitude / benchmark truth',ylabel='Noise correlation',title='Positive/57: reflected noise scenarios');ax.legend(fontsize=8)
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    lines=['Guarded full bootstrap and finite nuisance sensitivity','====================================================','',
    'Scope: ODE positive seeds57..59 paired with prior observed fields; fresh',
    'seeds60..62 under iid,+.4,-.4. 12 observed fits,1536 bootstrap refits,',
    'MC16 covariance. Guard applied throughout. The same observed raw fields',
    'are paired in the historical reruns; bootstrap fields change when the',
    'fitted noise model and reconstructed pilot change. Do not call those',
    'bootstrap samples paired. All cutoffs frozen before clean-field scoring.', '',
    '1. Full procedure with the profile-start guard',
    'All3 historically unbounded reference regions become bounded under both',
    'bootstrap normalizations. Every fresh reference region is also bounded.',
    'Fresh full-library error coverage:8/9 for either normalization, with the',
    'same negative/60 miss in both banks. Historical reruns:3/3. Each profile',
    'has only3 fresh outer datasets; this cannot establish95% calibration.',
    'All reference-equation noise projections are covered and the true',
    'reference equation remains accepted in every case and method.',
    'Grouped coverage counts (each group has3 datasets):',str(coverage),
    'Selected fit diagnostics:',str(out['fit_diagnostics']),
    'Observed noise-estimation errors:',str(noise),
    'The guard removes catastrophic optimization basins; positive-correlation',
    'amplitude/correlation bias remains. Convergence is not identifiability.', '',
    '2. Why the remaining fresh failure occurs (posthoc diagnostic)',
    'negative/60: fitted sigma0.10151 versus true0.10301; fitted rho−0.48649',
    'versus true−0.4. Broad full-library score9.598 exceeds cutoff6.072',
    '(studentized) or5.513(fixed). Using true noise only in the moment',
    'correction barely changes the score:9.623. Using it only in covariance',
    'lowers the score to7.427; changing both gives7.447. Thus the covariance',
    'normalization matters more here than the small correction shift.',
    'A separate known-noise conditional97.5% cutoff is7.195: the broad score',
    'still exceeds it by about3.5%. Parameter error does not explain the',
    'entire miss; finite-sample tails/pilot approximation remain possible.',
    'This one case cannot distinguish a chance tail from systematic undercoverage.',
    'The reference equation itself is accepted, so full-library failure must',
    'not be presented as failure to recover this reference equation.', '',
    '3. Finite scalar sensitivity envelope',
    '129 nuisance points per dataset: observed estimate plus all128 basic',
    'bootstrap reflections in(log sigma,atanh rho). 128 known-noise conditional',
    'fields per point;198144 conditional fields total. No point discarded.',
    'All129 scenarios completed in every dataset. Joint conditional rank126',
    '(97.5%) and triangle/Loewner bounds produce one scalar cutoff per dataset.',
    'The envelope covers all12 datasets but positive/57 becomes unbounded.',
    'This trades precision for robustness to the considered finite scenarios.',
    'Its finite noise set is NOT a95% nuisance confidence region. Convex-hull',
    'inclusion is only a diagnostic and does not mean a finite point equals truth.',
    'Nuisance truth in coordinate box / convex hull:',str(nuisance),
    'Finite scalar outer-width ratios versus studentized primary enclosure',
    '(33 coordinates in11 bounded cases; the unbounded case excluded):',str(out['scalar_width_ratio_bounded_only']), '',
    '4. A useful geometric fix: retain separate scenarios',
    'Exploratory diagnostic after primary evaluation, on the fixed reference',
    'support only. Keep each nuisance model own corrected moments, covariance',
    'and frozen conditional cutoff. UNION the129 two-bank feasible sets, then',
    'include both primary sets. An intersection across nuisance models would',
    'be unjustified because only one model needs to describe the data.',
    'All131 components in each of12 datasets have bounded outer certificates.',
    'Thus all12 finite unions are bounded, including positive/57 where scalar',
    'compression was provably unbounded. The scalar construction discards',
    'directional information and can cause avoidable loss of precision.',
    'Union outer boxes are coordinate envelopes of component ellipsoids, not',
    'exact coefficient intervals. Width ratios over36 coordinates:',str(out['union_width_ratio']),
    'Reoptimized outer enclosures need not be nested even when feasible sets',
    'are nested; compare these widths as enclosure sizes, not exact-set widths.',
    'This geometric fix does not calibrate the underlying nuisance set and',
    'was not validated for an adaptive EPDE candidate search.', '',
    '5. Verification and implementation',
    'New C backend implements the same statistical stationary innovations',
    'likelihood/Joseph covariance recurrence. No ODE/PDE solver is called.',
    '104 arithmetic/gradient cases and54 complete-fit comparisons passed.',
    'Every selected fit was checked with the original Python likelihood.',
    'Independent audit totals:',str(out['audit_totals']),
    'Each scenario-union certificate also passed independent region, witness',
    'and curvature checks. Source/input hashes and old manifests are checked',
    'by complete.py. Compiler and preflight issues are retained in histories.',
    'Deterministic-column batch roundoff was checked then canonicalized before',
    'envelope freezing. Constants and valid identities remain in the library.', '',
    '6. Decision and next experiment',
    'Keep the profile-start guard as the tested experimental numerical fix.',
    'Prefer preserving scenario geometry to one global scalar inflation when',
    'investigating nuisance sensitivity. Before calling this uncertainty',
    'calibrated, construct/test a nuisance region under a stated signal model',
    'and validate it on substantially more fresh outer fields. The positive',
    'noise bias remains the main statistical obstacle. The separate KdV weak',
    'direction problem has not been retested here and remains unresolved.',
    'Dyad remains provisional; keep weak accuracy and coefficient precision',
    'as separate checks. No production changes, solver or omitted failures.', '']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('SUMMARY',json.dumps({k:out[k] for k in ('coverage','nuisance_set_diagnostics','fit_diagnostics','scalar_width_ratio_bounded_only','union_width_ratio','audit_totals')}),flush=True)


if __name__=='__main__':main()
