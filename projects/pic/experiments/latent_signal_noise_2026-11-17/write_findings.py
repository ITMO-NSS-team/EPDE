from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def read(name):return json.loads((ROOT/name).read_text())

def main():
    summary=read('summary.json');pairs=read('paired_summary.json');main_audit=read('audit.json');follow=read('audit_followups.json');attribution=read('audit_attribution.json')
    assert all(d['status']=='PASS' for d in (main_audit,follow,attribution,read('audit_profile.json'),read('audit_ks_attribution.json')))
    incomplete=summary['incomplete'];noise=read('noise_comparison.json');profile=read('profile_diagnostic.json');attrs=read('failure_attribution_summary.json')
    lines=['Latent signal/noise fit: solver-free EPDE study, 2026-11-17',
      '', 'Conclusion', '----------',
      'Retaining informative frequencies restores pure-noise correlation estimation,',
      'but the tested joint signal/noise model is not a reliable universal covariance',
      'estimator. Unknown signal curvature can still be allocated to noise incorrectly.',
      'Fresh tests expose both nonlinear correction bias and wrong calibration shape.',
      'Keep the known-iid estimator when that family is specified. Do not promote the',
      'latent model or replace dyad on this evidence. Parameter-uncertainty propagation',
      'is deferred: a confidence region around a misspecified covariance need not fix',
      'its systematic error. This round tests uncertainty, not a new dyad ablation.',
      '', 'Design and numerical checks', '---------------------------',
      'Two global working models: white signal curvature plus AR noise, then AR signal',
      'curvature plus separate AR noise. Exact finite-boundary contrast covariances,',
      'bandwidth two/three after AR filtering, profiled marginal noise amplitude.',
      'Multidimensional fields use axis composite likelihood, not independent-axis',
      'confidence inference. Constants/linear trends are nuisance signal components;',
      'explicit constants and valid identities remain in the weak candidate library.',
      'No candidate dynamics integration, PINNs, arbitrary ridge or per-system variant.',
      '',
      'White-curvature checks:40 dense cases. Correlated-curvature checks:96 dense',
      f"cases, maximum quadratic/logdet discrepancy {read('checks_correlated.json')['max_error']:.3g}.",
      'Synthetic correlated curvature with true sigma0.2/rho0.4 returns',
      'sigma0.203014/rho0.411879. Unit/trend invariance passes. Initial forward-gradient',
      'implementation failed that check; central differences repaired it without',
      'loosening the tolerance. Initial and final source versions are retained.',
      '', 'Screening, before the fresh test', '--------------------------------',
      'Each model screened69 old anchors33..35,84 raw-variable and84 noise-only fits.',
      'White curvature: iid raw amplitude median0.9553 times truth; three NS/u noise',
      'correlation boundaries. Correlated curvature: median0.9855; all84 raw and84',
      'noise-only fits status ok. This did not establish correctness. AC negative',
      'noise was fitted with positive correlation along one axis, error up to0.8513.',
      'A signal-phi edge is a reported nuisance-smoothness diagnostic, distinct from',
      'a noise-rho boundary. No per-system model choice was made.',
      '', 'Matched pure-noise trials', '-------------------------',
      '576 draws: n301/320/1001, rho0/+0.4/-0.4,64 draws per cell, sigma0.2.',
      'Same fields across all four methods; the zero-signal full AR fit is an oracle',
      'diagnostic, never the raw-field estimator. Incomplete fits: white curvature0,',
      'correlated curvature1 (optimizer failure, n1001/rho-0.4/draw57), spectral25.',
      'Spectral has23/192 positive-rho boundary fits. RMSE below includes every',
      'available estimate, including spectral boundary estimates. None of the',
      'positive-rho latent/oracle fits is missing.',
      '', 'Positive-rho correlation RMSE:', 'n       spectral       white curvature   correlated curvature  zero-signal oracle']
    for n in (301,320,1001):
        vals=[next(r['rho_rmse'] for r in noise['summary'] if r['n']==n and r['rho']==.4 and r['method']==m) for m in ('Spectral strips','White curvature','Correlated curvature','Known zero signal')]
        lines.append(f'{n:<7} '+''.join(f'{v:<17.5f}' for v in vals))
    lines += ['', 'Fresh coverage: frozen model, anchors36..38', '------------------------------------------',
      '138 planned paired records: all15 iid x3 anchors x2 models=90; core4',
      'AC/forcedKdV/KS/wave x3 anchors x2 correlation signs x2 models=48.',
      '69 distinct fresh anchors; each complete model record has256 calibration,',
      '256 check and256 independent evaluation fields. Same raw/evaluation fields',
      'between paired methods. Previous pilot, banks, library and correction/test',
      'code were retained. Fit/metric/cutoff/regions were frozen before clean truth',
      'or evaluation fields were inspected. Noise-parameter uncertainty held fixed.',
      f"Completed {138-main_audit['incomplete']}/138 records; incomplete records retained: {incomplete}.",
      'All workers exited0. The incomplete noise fit is not a successful coverage',
      'result and is not silently replaced by a control.',
      '', 'Nominal95% simultaneous operator coverage, matched complete pairs:',
      'profile      bank     pairs   latent(%)  specified-family control(%)']
    for r in pairs['summary']:
        lines.append(f"{r['profile']:<13}{r['bank']:<9}{r['complete_pairs']:<8}{100*r['candidate_coverage']:<11.2f}{100*r['control_coverage']:.2f}")
    lines += ['IID control estimates amplitude with rho fixed0; colored control estimates',
      'amplitude with rho fixed to the generating value. These are specified-family',
      'controls, not inferred correlation. Each mean weights fitted anchors equally.',
      'Three anchors per system do not establish unconditional95% validity; the256',
      'evaluation draws are not256 additional independent fitted anchors.',
      '', 'Failure attribution', '-------------------',
      'Oracle diagnostic changes only the actual-field Gaussian correction, keeping',
      'the pilot, fitted calibration, metric, cutoff and evaluation fields fixed.',
      'NS seed37 has no usable model, so attribution is explicitly SKIPPED.',
      'system         seed bank    original/256  oracle correction/256  bias-score/cutoff']
    for r in sorted((r for r in attrs['rows'] if 'bank' in r),key=lambda r:(r['system'],r['seed'],r['bank'])):
        lines.append(f"{r['system']:<15}{r['seed']:<5}{r['bank']:<8}{r['original_coverage_count']:<14}{r['oracle_correction_coverage_count']:<23}{r['analytic_bias_cutoff_ratio']:.5f}")
    lines += [
      'Compound broad coverage rises from12/256,25/256,21/256 to253/256,251/256,',
      '253/256. The remaining correction bias has score1.28..1.33 times the cutoff.',
      'Smaller correlation error than the earlier spectral fit did not remove',
      'sensitivity of nonlinear moments to covariance misspecification.',
      'NS local coverage stays near2..3% even with oracle correction. Mean bias is',
      'only about1% of its cutoff in score units. Empirical post-correction second',
      'moments exceed fitted calibration by factors1.83..1.91 (local) and1.99..2.03',
      '(broad) in the most affected coefficient directions. These are finite-sample',
      'diagnostic eigenvalue ratios, not confidence bounds. Covariance/calibration',
      'shape is the dominant NS issue. Both are different from scalar amplitude alone.',
      '', 'KS follow-up: variance error can make local coverage collapse',
      '-------------------------------------------------------------',
      'The completed primary test has zero KS local operator coverage on all nine',
      'anchors: three seeds in each of iid/positive/negative noise. The local test',
      'also rejects the exact reference on all nine anchors. A targeted seed36',
      'oracle-correction control for each profile still gives0/256 local coverage.',
      'Its analytic mean-bias score is only0.00014..0.00047 times the cutoff.',
      'Actual local second moments exceed calibration by factors1.26..1.30 (iid),',
      '1.35..1.58 (positive),1.19..1.21 (negative). The local operator statistic',
      'concentrates tightly, so this variance error is enough to destroy coverage:',
      'for iid, the actual score fifth percentile is1.324 while the cutoff is1.135.',
      'Broad coverage is less sensitive to the same error but remains below nominal.',
      'All three attribution records pass1536 independent score checks and12 native',
      'matrix replays. The diagnostic eigenvalues also pass a direct generalized',
      'eigenvalue calculation. This is calibration failure, not correction mean bias.',
      '', 'Objective diagnosis', '-------------------',
      'Nine old-anchor variable fields: AC negative seeds33..35, compound iid',
      'seeds33..35, NS iid seed33 all variables. Refit nuisance curvature with noise',
      'rho fixed to truth, then restart unrestricted fitting from that solution.',
      f"Best fixed-rho normalized objectives exceed the originals by {min(r['fixed_minus_original'] for r in profile['rows']):.6g}..{max(r['fixed_minus_original'] for r in profile['rows']):.6g}.",
      'None of those unrestricted restarts materially improves the original objective.',
      'This supports working-signal-model bias rather than a simple bad initialization;',
      'it does not prove global optimality. Composite objective differences are not',
      'interpreted as chi-square likelihood-ratio confidence statistics.',
      '', 'Coefficient compatibility and precision', '---------------------------------------']
    for r in summary['summary']:
        if r['profile']=='iid' and r['bank']=='both':lines.append(f"{r['model_kind']}: {r['exact_anchor_accepted']}/{r['reference_cases']} reference anchors accepted; combined geometry {r['geometry']}.")
    lines += ['On completed NS/compound cases the exact reference directions remain',
      'compatible despite poor full-library operator coverage. Compatibility does not',
      'establish correct uncertainty in every direction, narrow coefficients or unique',
      'structure. The previously observed same-target superset ambiguity remains.',
      '', 'Verification and artifacts', '--------------------------',
      f"Main audit PASS: {main_audit['totals']}.",
      f"Exploration/noise audit PASS: {follow['exploration_records']} records, {follow['likelihood_replays']} likelihood replays, {follow['field_replays']} field replays, {follow['hash_links']} linked hashes.",
      'Main case audits independently refit every raw noise model, replay moments,',
      'verify paired fields and use direct generalized eigenvalues for operator scores.',
      f"Attribution audit PASS: {attribution['operator_scores']} operator scores, {attribution['matrix_replays']} matrix replays, {attribution['skipped']} explicit skipped record.",
      'Profile audit PASS:9 fields/36 objective replays. Constant columns have exactly',
      'zero stochastic error throughout calibration. Source/input/result/array hashes,',
      'failed model attempts, figures and CSVs are retained. Completion audit and',
      'final artifact manifest accompany this report. No production files or prior',
      'completed experiment artifacts were edited; no commit was made.',
      '', 'Decision and next research step', '-------------------------------',
      'Keep dyad provisional for structure. Keep specified-iid noise estimation for',
      'that benchmark family. The latent covariance models remain research variants.',
      'The next model repair must check both correction bias and covariance shape,',
      'with a more adequate nuisance signal representation and observable adequacy',
      'diagnostics. Only then assess parameter-uncertainty propagation. Repeated',
      'measurements, if available, would allow signal-canceling noise estimates; the',
      'present single-field experiments do not assume they exist. No new instability',
      'penalty or arbitrary confidence inflation is justified by this round.']
    (ROOT/'findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='latent_signal_noise_2026-11-17/findings.txt';master=ROOT.parent/'solver_free_research_2026-11-16.txt';assert marker not in master.read_text(encoding='utf-8')
    log=['','2026-11-17: latent signal/noise follow-up',marker,
      'White-curvature and correlated-curvature solver-free contrast likelihoods;',
      '69 old-anchor cases per model, then138 fresh paired records on anchors36..38.',
      f"{138-main_audit['incomplete']} complete; {main_audit['incomplete']} incomplete noise fit retained. 576 paired pure-noise draws.",
      'Pure-noise correlation information largely restored, but raw-signal model',
      'misspecification remains. No per-system estimator selection or production edit.',
      'Matched operator coverage (latent/control):']
    for r in pairs['summary']:
        if r['bank']!='both':log.append(f"{r['profile']}/{r['bank']}: {100*r['candidate_coverage']:.2f}/{100*r['control_coverage']:.2f}% on{r['complete_pairs']} pairs.")
    log += ['Compound broad12/25/21 successes of256 become253/251/253 after oracle',
      'correction alone. NS local4/5 become6/7; calibration second moments are',
      'underestimated by about1.8..2.0 in affected directions; seed37 is incomplete.',
      'KS local coverage0 on all9 primary cases; oracle correction on seed36 in',
      'all3 regimes leaves0/256. Variance errors of19..58% exceed its concentrated',
      'local calibration distribution; mean correction bias is negligible.',
      'All main/diagnostic audits pass. Explicit constants/valid identities retained.',
      'Keep dyad and specified-iid estimation. Reject this latent fit as universal',
      'default; defer noise-parameter propagation until covariance adequacy is repaired.','']
    with master.open('a',encoding='utf-8') as f:f.write('\n'.join(log))
    print('Saved findings and appended master log.')

if __name__=='__main__':main()
