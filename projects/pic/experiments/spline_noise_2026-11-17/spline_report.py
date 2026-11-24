from spline_context import *
from collections import Counter
from scipy.stats import beta
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROFILES=('iid','positive','negative')
METHODS=('guarded','spline')

def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return dict(n=0)
    return dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),min=float(min(x)),max=float(max(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))

def coverage(x):
    n=len(x);unresolved=sum(v is None for v in x);k=sum(v is True for v in x);den=n-unresolved
    return dict(n=n,covered=k,unresolved=unresolved,binomial95=[float(beta.ppf(.025,k,den-k+1)) if k else 0.,float(beta.ppf(.975,k+1,den-k)) if k<den else 1.])

def csvsave(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def width(cert):
    if cert['geometry']!='bounded_outer':return None
    return np.array(cert['outer_region']['upper'])-cert['outer_region']['lower']

def main():
    for name in ('point_audit','bootstrap_audit','propagation_audit'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    point=[read(p) for p in sorted((ROOT/'point_results').glob('*.json'))];boot=[read(p) for p in sorted((ROOT/'bootstrap_results').glob('*.json'))];prop=[read(p) for p in sorted((ROOT/'propagation_results').glob('*.json'))];assert len(point)==264 and len(boot)==len(prop)==24
    assert read(ROOT/'union_review.json')['status']=='PASS'
    for r in prop:r['finite_union_geometry']=read(ROOT/'union_reviews'/(r['key']+'.json'))['geometry']
    for r in boot:
        if r['numerical_status']!='FROZEN':r['coverage']={m:None for m in r['coverage']}
    for r in prop:
        if r['numerical_status']!='FROZEN':
            r['primary_geometry']={m:r['numerical_status'] for m in r['primary_geometry']};r['finite_union_geometry']=r['scalar_geometry']=r['numerical_status']
    accuracy={};point_rows=[];pairing={}
    for name in ('ode','vdp','duffing'):
        accuracy[name]={};pairing[name]={}
        for profile in PROFILES:
            rr=[r for r in point if r['name']==name and r['profile']==profile];mm={}
            for method in ('guarded','spline','p8','p16','p24','p32','p48','p64'):
                mm[method]={q:stats([r['errors'][method][q] for r in rr]) for q in ('sigma_ratio','rho_error')};mm[method]['near_unit']=sum(r['errors'][method]['near_unit'] for r in rr)
                point_rows.append(dict(system=name,profile=profile,method=method,n=len(rr),sigma_ratio_median=mm[method]['sigma_ratio']['median'],rho_error_median=mm[method]['rho_error']['median'],near_unit=mm[method]['near_unit']))
            mm['selected_dimensions']=dict(Counter(r['selected_p'] for r in rr));mm['selected_status']=dict(Counter(r['selected_status'] for r in rr));accuracy[name][profile]=mm
            pairing[name][profile]={q:dict(spline_better=sum(abs(r['errors']['spline'][q]-target)<abs(r['errors']['guarded'][q]-target) for r in rr),n=len(rr),guarded_median_absolute_error=float(np.median([abs(r['errors']['guarded'][q]-target) for r in rr])),spline_median_absolute_error=float(np.median([abs(r['errors']['spline'][q]-target) for r in rr]))) for q,target in [('sigma_ratio',1),('rho_error',0)]}
    bcov={};pcov={};geometry={};warnings=[];width_rows=[];allwidths={};noise_area={};selection={}
    for method in METHODS:
        bcov[method]={};pcov[method]={};geometry[method]={};noise_area[method]={};selection[method]={}
        for profile in PROFILES:
            bb=[r for r in boot if r['method']==method and r['profile']==profile];pp=[r for r in prop if r['method']==method and r['profile']==profile]
            bcov[method][profile]={mode:coverage([r['coverage'][mode] for r in bb]) for mode in bb[0]['coverage']};pcov[method][profile]={mode:coverage([r['coverage'][mode] for r in pp]) for mode in pp[0]['coverage']};geometry[method][profile]={mode:dict(Counter(r['primary_geometry'][mode] for r in pp)) for mode in ('studentized','fixed_metric')};geometry[method][profile].update(finite_union=dict(Counter(r['finite_union_geometry'] for r in pp)),scalar=dict(Counter(r['scalar_geometry'] for r in pp)));noise_area[method][profile]=stats([r['regions']['basic_empirical']['area_transformed'] for r in bb if r['numerical_status']=='FROZEN'])
        for r in [x for x in boot if x['method']==method]:
            key=r['key'];b=read(ROOT/'bootstrap_frozen'/(key+'.json'));a=read(ROOT/'bootstrap_anchors'/(key+'.json'));rows=[dict(draw=-1,info=a['fit_info'])]+b['refits'];warnings.extend(dict(key=key,draw=x['draw'],status=x['info']['status']) for x in rows if x['info']['status']!='ok')
            if method=='spline':selection[key]=dict(Counter(x['info']['selected_p'] for x in rows))
    for r in prop:
        key=r['key'];p=read(ROOT/'propagation_primary'/(key+'.json'));u=read(ROOT/'propagation_frozen'/(key+'.json'));w={mode:width(next(c for c in p['certificates'][mode] if c['label']=='reference')) for mode in ('studentized','fixed_metric')};w['scalar']=width(u['scalar_certificate']);w['finite_union']=np.array(u['outer_box']['upper'])-u['outer_box']['lower'] if u['geometry']=='bounded_outer' else None;allwidths[key]=w
        reviewed=read(ROOT/'union_reviews'/(key+'.json'));w['finite_union']=np.array(reviewed['outer_box']['upper'])-reviewed['outer_box']['lower'] if reviewed['geometry']=='bounded_outer' else None
        if r['numerical_status']!='FROZEN':w={mode:None for mode in w};allwidths[key]=w
        for mode,x in w.items():
            for j in range(3):width_rows.append(dict(key=key,method=r['method'],profile=r['profile'],seed=r['seed'],mode=mode,coefficient=j,width=None if x is None else float(x[j])))
    comparisons={};expansion={}
    for mode in ('studentized','fixed_metric','finite_union','scalar'):
        rr=[];pairs=0
        for r in [x for x in prop if x['method']=='spline']:
            x=allwidths[r['key']][mode];y=allwidths[r['key'].replace('_spline','_guarded')][mode]
            if x is not None and y is not None:rr.extend(x/y);pairs+=1
        comparisons[mode]=dict(**stats(rr),bounded_pairs=pairs,total_pairs=12,profiles={})
        for profile in PROFILES:
            vals=[];n=0
            for r in [x for x in prop if x['method']=='spline' and x['profile']==profile]:
                a=allwidths[r['key']][mode];b=allwidths[r['key'].replace('_spline','_guarded')][mode]
                if a is not None and b is not None:vals.extend(a/b);n+=1
            comparisons[mode]['profiles'][profile]=dict(**stats(vals),bounded_pairs=n,total_pairs=4)
    for method in METHODS:
        expansion[method]={}
        for mode in ('finite_union','scalar'):
            rr=[]
            for key,w in allwidths.items():
                if key.endswith('_'+method) and w[mode] is not None and w['studentized'] is not None:rr.extend(w[mode]/w['studentized'])
            expansion[method][mode]=stats(rr)
    failures=[]
    for r in sorted(point,key=lambda r:abs(np.log(r['errors']['spline']['sigma_ratio'])),reverse=True)[:12]:
        d=read(ROOT/'point_frozen'/(r['key']+'.json'));sel=d['info']['selection'];chosen=next(v for v in sel if v['p']==r['selected_p']);p32=next(v for v in sel if v['p']==32)
        failures.append(dict(key=r['key'],selected_p=r['selected_p'],spline=r['errors']['spline'],fixed_p32=r['errors']['p32'],cv_p32_minus_selected=p32['score']-chosen['score'],mean_rmse_over_noise=r['mean_rmse_over_noise']))
    leakage=read(ROOT/'leakage_diagnostic.json');assert leakage['status']=='PASS'
    summary=dict(status='COMPLETE',point_accuracy=accuracy,paired_absolute_errors=pairing,noise_coverage=bcov,weak_error_coverage=pcov,reference_geometry=geometry,paired_outer_width_ratio_spline_over_guarded=comparisons,finite_expansion_over_primary=expansion,noise_region_area=noise_area,spline_bootstrap_dimensions=selection,selected_fit_warnings=warnings,largest_point_errors=failures,signal_leakage=leakage['rows'],reference_acceptance={m:sum(r['reference_accepted'] and r['numerical_status']=='FROZEN' for r in prop if r['method']==m) for m in METHODS},source_sha256=sources('spline_report.py','protocol.txt','warning_policy.txt'))
    dump(ROOT/'summary.json',summary);csvsave('point_accuracy.csv',point_rows);csvsave('coefficient_outer_widths.csv',width_rows)
    fig,axs=plt.subplots(2,2,figsize=(12.5,9),layout='constrained');x=np.arange(3);colors={'guarded':'#c56a51','spline':'#267f9b'}
    for j,method in enumerate(METHODS):
        ys=[accuracy['ode'][p][method]['sigma_ratio']['median'] for p in PROFILES];axs[0,0].bar(x+(j-.5)*.35,ys,.34,label=method,color=colors[method]);rr=[r for r in point if r['name']=='ode' and r['profile']=='positive'];axs[0,1].scatter([r['errors'][method]['sigma_ratio'] for r in rr],[r['errors'][method]['rho_error']+.4 for r in rr],s=17,alpha=.65,label=method,c=colors[method])
    axs[0,0].axhline(1,c='black',ls='--',lw=1);axs[0,0].set(xticks=x,xticklabels=PROFILES,ylabel='Median estimated / true noise amplitude',title='Fresh ODE fields: 64 per noise profile');axs[0,0].legend()
    axs[0,1].scatter([1],[.4],marker='*',s=130,c='black',label='truth');axs[0,1].set(xlabel='Estimated / true noise amplitude',ylabel='Estimated correlation',title='Positive correlation: paired estimates');axs[0,1].legend()
    for j,method in enumerate(METHODS):
        ys=[bcov[method][p]['basic_empirical']['covered'] for p in PROFILES];axs[1,0].bar(x+(j-.5)*.35,ys,.34,label=method,color=colors[method])
        for i,k in enumerate(ys):
            unresolved=bcov[method][PROFILES[i]]['basic_empirical']['unresolved'];label=str(k)+'/4'+(f'\n{unresolved} unresolved' if unresolved else '');axs[1,0].text(i+(j-.5)*.35,k+.07,label,ha='center',fontsize=8)
    axs[1,0].set(xticks=x,xticklabels=PROFILES,ylim=(0,4.7),ylabel='True noise pair covered / 4',title='Fresh bootstrap test: 12 paired fields');axs[1,0].legend(loc='lower left')
    ax=axs[1,1];positions=[];distributions=[];names=[]
    for i,mode in enumerate(('studentized','fixed_metric','finite_union','scalar')):
        ratios=[]
        for r in [x for x in prop if x['method']=='spline']:
            a=allwidths[r['key']][mode];b=allwidths[r['key'].replace('_spline','_guarded')][mode]
            if a is not None and b is not None:ratios.extend(a/b)
        if ratios:positions.append(i);distributions.append(ratios);names.append(mode.replace('_',' ')+'\n'+str(len(ratios))+'/36 coords')
    ax.boxplot(distributions,positions=positions,tick_labels=names,showfliers=True);ax.axhline(1,c='black',ls='--',lw=1);ax.set(ylabel='Spline / guarded outer width',title='Precision: bounded, resolved pairs only');ax.tick_params(axis='x',labelsize=8)
    fig.savefig(ROOT/'results_overview.png',dpi=180);fig.savefig(ROOT/'results_overview.pdf');plt.close(fig)
    lines=['Spline signal means for solver-free noise estimation','===================================================','',
    'A change in the signal representation greatly reduces the positive-noise',
    'bias on the forced ODE benchmark, but is not ready to replace the fitter:',
    'one of12 bootstrap cases is unresolved and another noise region misses',
    'truth after bias correction. It does not establish uniform confidence',
    'coverage or settle the dyad objective. This is the dataset named ode,',
    'not ode_simple. No equation solver or clean/reference-based fitting is used.','',
    '1. Frozen method and independent data',
    'Raw data = deterministic cubic spline mean + Gaussian AR1 measurement',
    'noise. Candidate dimensions:8,16,24,32,48,64. Choose dimension by four',
    'held-out predictive scores, refitting noise in each fold. Fit each fixed',
    'dimension by full-covariance restricted likelihood. Retain all grid minima',
    'and endpoints; verify selected arithmetic with whitened QR. The finite',
    'search is not a proof of global optimality. Held-out Student-t densities',
    'plug in rho and are selection scores, not coverage claims. Constants and',
    'linear terms are present in the spline; the equation library is unchanged.',
    '264 fresh paired fields: ODE192, VDP48, Duffing24, 5% noise; iid,+.4,-.4.',
    'All estimates were frozen before scoring against clean data.','',
    'Median noise amplitude / truth and correlation error (guarded -> spline):']
    for name in ('ode','vdp','duffing'):
        for p in PROFILES:
            a=accuracy[name][p];lines.append(f"{name}/{p}: n={a['guarded']['sigma_ratio']['n']}, sigma {a['guarded']['sigma_ratio']['median']:.5f} -> {a['spline']['sigma_ratio']['median']:.5f}; rho error {a['guarded']['rho_error']['median']:+.5f} -> {a['spline']['rho_error']['median']:+.5f}; dimensions {a['selected_dimensions']}")
    lines += ['', '2. Remaining point-estimation failures',
    'Positive ODE data selected p24 in5/64 cases. The largest amplitude error',
    'was +27.8%(seed354); small splines can still assign signal residuals to',
    'noise. Cross-validation optimizes prediction, not true-noise recovery.',
    'The fixed-p32 results below are posthoc diagnostics, not a retuned method.',
    json.dumps(failures,indent=2),'',
    'Independent posthoc approximation diagnostic (not used for any choice):',
    'Hold rho and spline dimension fixed. The expected restricted variance is',
    'sigma_true^2 + signal_residual_quadratic/(n-p). These are square roots',
    'of expected estimated variance divided by true sigma, not expected sigma:',
    '\n'.join(f"{r['system']}/{r['profile']}/p{r['p']}: {r['root_expected_sigma_squared_over_true_sigma']:.5f}" for r in leakage['rows'] if r['p'] in (16,24,32)), '',
    '3. Independent bootstrap and weak-error regions',
    'Fresh ODE seeds400..403:12 paired raw observations,128 refits per method',
    '(3072 total). Spline selection is repeated in every draw. Both procedures',
    'use the same correlated-GCV pilot rule; pilots depend on estimated noise.',
    'Bootstrap fields share normals but differ because estimated noise/pilots',
    'differ. No bootstrap replicate is removed. Noise coordinates are log sigma',
    'and atanh rho; basic center2theta_hat-mean(theta_star), covariance from128',
    'draws and empirical rank126 Mahalanobis radius. Two-bank primary weak-error',
    'cutoffs use rank123 and MC16 covariance with the observed pilot held fixed.',
    'Both studentized and fixed-metric cutoffs are tested.','',
    'Noise-pair coverage (basic empirical):']
    for method in METHODS:
        lines.append(method+': '+', '.join(p+' '+str(bcov[method][p]['basic_empirical']['covered'])+'/4, unresolved '+str(bcov[method][p]['basic_empirical']['unresolved']) for p in PROFILES))
    lines += ['','Full-library weak-error coverage:']
    for method in METHODS:
        for p in PROFILES:lines.append(method+'/'+p+': '+', '.join(m+' '+str(c['covered'])+'/4, unresolved '+str(c['unresolved']) for m,c in pcov[method][p].items()))
    lines += ['', 'Reference coefficient-region geometry:',json.dumps(geometry,indent=2),'',
    '4. Finite nuisance propagation and precision',
    '66 points per noise ellipse: center, estimate,32 half-radius angles and32',
    'full-radius angles.128 known-noise conditional fields per point; rank126',
    'joint cutoff, MC16 covariance. The union also retains both primary regions.',
    'This finite probe DOES NOT certify the continuous ellipse or adaptive',
    'equation search. Scalar triangle/Loewner compression is a comparator.',
    'Paired outer-width ratios, spline / guarded (bounded pairs only):',
    json.dumps(comparisons,indent=2),
    'Within-method finite/scalar outer width divided by studentized primary:',
    json.dumps(expansion,indent=2),
    'Widths describe outer enclosures of the reference-support intersection,',
    'not exact confidence intervals. The guarded positive400 union originally',
    'had an unresolved bookkeeping status:58 bounded and10 certified empty',
    'components. The independently verified empty components contribute',
    'nothing, so the union is bounded. A separate review records this logical',
    'correction without overwriting the frozen output. No unresolved set or',
    'bootstrap draw is discarded. Denominators and unbounded cases must be',
    'reported; successful widths alone must not hide lost boundedness.',
    'Reference acceptance: '+str(summary['reference_acceptance'])+'/12 per method.','',
    '5. Verification and scope',
    'Independent dense error-contrast checks for264 selected spline fits;',
    'replayed every spline selection, every original guarded selected fit,',
    'all generalized eigenvalue scores and coefficient/Loewner certificates.',
    'Representative covariance and raw bootstrap fields are independently',
    'reconstructed. All source/data hash links and prior manifests are checked.',
    'Selected-fit warnings: '+str(warnings),
    'The positive402 spline case hit the rho search endpoint in bootstrap',
    'draws46 and76, selecting p16. All draws and downstream calculations are',
    'retained, but coverage is unresolved and this case is excluded from valid',
    'width pairs while remaining in the12-case denominator. Conditional',
    'prediction can reward a near-random-walk noise model for omitted signal;',
    'good prediction therefore does not ensure accurate noise separation.',
    'Only4 outer fields per noise profile in the uncertainty comparison:',
    'even4/4 gives a wide binomial interval. This does not validate a95%',
    'or97.5% guarantee. The bootstrap was repeated after selection, but that',
    'alone cannot guarantee that it reproduces signal-model selection bias.',
    'No production method is replaced. No KdV repair or full15-system rerun',
    'is claimed here. Dyad remains provisional; constants and valid identities',
    'remain admissible. Next: test noise-identification criteria and stability',
    'across competing spline dimensions, rather than relying on conditional',
    'prediction alone or clipping rho. Calibrate uncertainty separately for',
    'the new estimator: compare uncorrected and basic centers on larger fresh',
    'cohorts, because inherited bias correction can worsen an accurate fit.',
    'Only then extend the comparison to the full default system set.','']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8');print('REPORT COMPLETE',json.dumps(dict(noise=bcov,weak=pcov,widths=comparisons,warnings=warnings)),flush=True)


if __name__=='__main__':main()
