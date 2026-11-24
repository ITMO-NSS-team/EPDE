from common import *

def pct(x):return f'{100*x:.2f}%'
def main():
    a=read(ROOT/'summary.json');s=read(ROOT/'studentized_summary.json');t=read(ROOT/'time_summary.json');p=read(ROOT/'time_pilot_summary.json');numeric=read(ROOT/'audit_state.json')
    lines=['Solver-free repairs: numerical fitting, signal/noise separation and uncertainty','===============================================================================','',
    'Decision: retain dyad provisionally for structural ranking. No dyad ablation or',
    'replacement is justified by this round. Repair noise estimation, reconstruction',
    'and uncertainty as distinct components. Do not promote the local-polynomial',
    'noise estimator globally. The temporal estimator with correlation-aware pilot',
    'is a tested candidate for further coverage validation, not a universal default.',
    'All work is in this research folder; no production source or prior completed',
    'study is changed. Constants and valid identities are retained. No candidate',
    'dynamical equation is integrated.', '',
    '1. Numerical convergence: repaired and independently checked',
    '----------------------------------------------------------',
    'Reevaluated84 previous AR2 variable fits, using Fisher coordinates and an exact',
    'stationary innovations likelihood with analytic sensitivities. Refined the11',
    'remaining unresolved fits. All84 meet the2e-5 projected-score criterion.',
    f"Centered finite differences on11 refinements: maximum error {numeric['max_finite_difference_error']:.3g}.",
    'This verifies local numerical stationarity, not global optimality or statistical',
    'identification. Relative to the transformed banded fits, sigma changes<0.03%;',
    'compound is essentially unchanged. Conditioning did not explain its main bias.', '',
    '2. Robust local polynomial fitting: fixes a bias, loses information',
    '----------------------------------------------------------------',
    'Initial cubic projection fails NS; its failed screen is preserved. Quintic',
    'projection improves compound markedly, including raw/noise-only paired screens.',
    'The fresh primary study covers15 iid systems plus4 systems under both positive',
    'and negative correlation:69 anchors42..44,138 paired procedure records, each',
    'with256 calibration/check/actual fields. All69 finish;2 contain42 noise-boundary',
    'warnings in AC calibration/check refits. All finite fits are retained.',
    'Fixed and refit branches share fields; refit reestimates moment correction on',
    'each draw. Its covariance normalization is still anchored in this first study.', '',
    'Primary operator coverage, target95% (local / broad):']
    for profile in ('iid','positive','negative'):
        for mode in ('fixed','refit'):
            values=[next(r['coverage'] for r in a['summary'] if (r['profile'],r['mode'],r['bank'])==(profile,mode,b)) for b in ('local','broad')];lines.append(f'  {profile:8s} {mode:5s}: {pct(values[0])} / {pct(values[1])}')
    lines += ['',
    'Compound refit broad coverage is98.05%; oracle actual-field correction changes',
    'it only slightly. Earlier AR2 broad coverage was3.65% on different anchors;',
    'that cross-study change is not a paired coverage estimate. The paired screen',
    'and oracle diagnostic support removal of the major correction bias.',
    'Pure-noise trials expose a separate flaw: short positive-correlation series',
    'lose identification after quintic projection. Correlation RMSE reaches0.33-0.34,',
    'and extreme sigma estimates exceed100x truth, despite numerical convergence.', '',
    '3. Studentization: normalization uncertainty matters, but is not sufficient',
    '-------------------------------------------------------------------------',
    'Recompute each field covariance metric at its fitted noise parameters, using32',
    'common Gaussian fields and validated local responses with direct fallbacks.',
    'The same69 anchors are exploratory follow-ups, not fresh confirmation.',
    'All per-draw covariance matrices/statistics are saved and independently audited.', '',
    'Correction-refit -> studentized operator coverage (local / broad):']
    for name,profile in [('ks','iid'),('ks','positive'),('ks','negative'),('ns','iid'),('pde_compound','iid'),('ac','positive')]:
        values=[]
        for mode in ('refit','studentized'):
            vv=[next(r['coverage'] for r in s['systems'] if (r['system'],r['profile'],r['mode'],r['bank'])==(name,profile,mode,b)) for b in ('local','broad')];values.append(' / '.join(pct(x) for x in vv))
        lines.append(f'  {name:12s} {profile:8s}: {values[0]} -> {values[1]}')
    lines += ['',
    'KS iid local coverage improves86.46% ->90.89%, still below95%. NS loses much',
    'of its previous conservatism. AC-positive collapses because its bootstrap noise',
    'fits drift toward high-amplitude, near-unit-root models, distorting per-draw',
    'normalization. Noise parameters and the bootstrap nuisance signal need repair;',
    'changing a cutoff or adding another instability penalty would hide this cause.',
    'Oracle correction leaves the principal KS/NS failures essentially unchanged,',
    'separating normalization/model mismatch from mean moment-correction bias.', '',
    'These are uniform operator bounds over the moment library. Failure of such a',
    'bound does not mean the reference equation itself is rejected. Reference-',
    'direction coverage and region geometry are recorded separately in the CSVs.', '',
    'In KS-positive specifically, local reference-direction noise coverage AND',
    'zero-residual acceptance are also0/768 under the separate local cutoff.',
    'Broad reference-direction coverage is99.61%. Joint observed-anchor regions',
    'accept84/84 references, with75 bounded outer regions and9 unbounded; this',
    'uses a different joint cutoff and is not evidence of valid repeated coverage.', '',
    '4. Temporal estimator: fresh parameter validation across the default set',
    '-----------------------------------------------------------------------',
    'Fit AR2 nuisance curvature plus AR1 observation noise on temporal differences.',
    'Estimate spatial correlations from signs of fourth temporal differences, which',
    'preserve separable spatial noise correlation without spatial frequency removal.',
    'One fixed rule uses the known time axis for every system. No truth-based axis',
    'selection. Synthetic, scale/trend invariance and likelihood-score audits pass.',
    'Fresh anchors45..47:69 cases,84 variables; compare both estimators on identical',
    'raw fields, then use isolated noise only for attribution. Separately assess192',
    'bootstrap fields in4 difficult cases and576 paired pure-noise realizations.',
    f"Fresh fit statuses: {t['fresh_statuses']}", '',
    'Fresh iid amplitude ratio ranges (all3 anchors and variables):']
    for name in ('ns','pde_compound','ks','ode','vdp'):
        entries=[]
        for method in ('local6','time'):
            r=next(r for r in t['systems'] if (r['system'],r['profile'],r['method'])==(name,'iid',method));entries.append(f"{method} {r['sigma_ratio_min']:.4f}..{r['sigma_ratio_max']:.4f}")
        lines.append('  '+name+': '+', '.join(entries))
    lines += ['', 'Positive-correlation pure-noise RMSE, time / local6 / previous AR2:']
    for n in (301,320,1001):
        entries=[next(r['rho_rmse'] for r in t['pure_noise'] if (r['method'],r['n'],r['rho'])==(method,n,.4)) for method in ('time','local6','previous_ar2')];lines.append(f'  n{n}: '+' / '.join(f'{x:.4f}' for x in entries))
    lines += ['',
    'These parameter checks do not establish confidence coverage. Raw signal and',
    'finite-sample estimation errors remain, including KS under positive correlation.', '',
    'The original pilot also breaks the temporal estimator on KS-positive bootstrap',
    'fields: sigma ratios1.267..5.031, with6/48 fits having |rho|>0.95, despite',
    'numerical status ok. Local6 on the same fields gives1.049..1.121. Thus retaining',
    'temporal information alone is not enough; the combined correlation-aware pilot',
    'repair is tested for coverage on AC below, not yet on KS.', '',
    '5. Correlation-aware pilot: a second identifiable source of bootstrap bias',
    '------------------------------------------------------------------------',
    'Ordinary GCV treats axis errors as independent and can retain correlated noise',
    'in the bootstrap signal. Reconstruct in estimated-noise-whitened coordinates',
    'using the same temporal spline/spatial DCT penalties and GCV criterion.',
    'The paired AC screen changes only this pilot, retaining identical bootstrap',
    'noise: mean fitted/generating sigma ratios1.239/1.241/1.296 become',
    '1.004/1.014/1.009. Direct normal-equation and invariance checks pass.',
    'This is exploratory attribution on anchors45..47.', '',
    'Fresh targeted AC-positive coverage: temporal fit + correlation-aware pilot,',
    'anchors48..50,256 calibration/check/actual draws per anchor. Method and',
    'studentization frozen before these results. Coverage (local / broad / joint):']
    for mode in ('fixed','refit','studentized'):
        vv=[next(r['coverage'] for r in p['summary'] if (r['mode'],r['bank'])==(mode,b)) for b in ('local','broad','both')];lines.append(f"  {mode:11s}: "+' / '.join(pct(x) for x in vv))
    lines += [f"Fit warnings: {len(p['warnings'])}. Per-anchor counts are in time_pilot_summary.json.",
    'This targeted test has3 independent fitted anchors in one PDE/noise regime.',
    'It cannot certify the combined method for the entire default system set.', '',
    '6. Practical conclusion and remaining work',
    '-----------------------------------------',
    'Numerical repair is verified. Compound correction bias is substantially reduced.',
    'Noise-parameter uncertainty is now propagated through correction and through',
    'normalization, but the69-system/profile study does not support universal95%',
    'coverage. The new temporal/pilot combination addresses specific measured causes;',
    'full default-set coverage validation remains before any production promotion.',
    'Use the specified noise family when the benchmark supplies it. Keep structural',
    'ranking, moment accuracy and coefficient-region precision separate. Accepted',
    'identities/constants remain valid; acceptance alone does not imply precise',
    'coefficients or uniquely identified dynamics.',
    'Next decisive experiment: frozen temporal estimator/correlation-aware pilot on',
    'new full-set coverage anchors, especially KS-positive and short ODEs. Assess',
    'joint reference-region geometry and bootstrap nuisance sensitivity alongside',
    'coverage. Do not use system-specific thresholds to force nominal percentages.', '',
    'Reproducibility and limits',
    '--------------------------',
    '256 Monte Carlo fields share one fitted anchor; do not treat them as256',
    'independent fitted datasets. Three anchors cannot establish unconditional',
    'coverage. Actual replicates retain the anchor pilot and weak design; pilot',
    'reconstruction is not repeated on every actual field. These are conditional',
    'operator-calibration tests, not full-pipeline confidence coverage. Propagating',
    'reconstruction uncertainty remains an additional requirement.',
    'Studentizer uses32 common fields and local approximation with direct',
    'fallbacks. Positive/negative profiles assume separable Gaussian AR1 noise.',
    'No conclusion extends automatically to nonseparable/heavy-tailed noise.',
    'Startup/reporting/import failures, source variants and frozen records remain',
    'archived. Exact stencil acceleration changes cutoffs by at most1.29e-11 relative',
    'to the initial implementation;120 ordinary and20 near-boundary covariance',
    'checks pass. The unused batch-moment helper is tested but not used in results.',
    'See method_notes.txt and time_method_notes.txt for derivations; summary.json,',
    'studentized_summary.json, time_summary.json and time_pilot_summary.json for',
    'complete tables. completion_audit.json and artifact_manifest.json record',
    'counts, checks and immutable source/input/result hashes.', '']
    (ROOT/'findings.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('WROTE',ROOT/'findings.txt',flush=True)
if __name__=='__main__':main()
