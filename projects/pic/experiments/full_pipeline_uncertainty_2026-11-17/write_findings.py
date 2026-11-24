from bridge import *
import csv

def main():
    summary=read(ROOT/'summary.json');factorial=read(ROOT/'factorial_analysis.json');effect=read(ROOT/'pilot_effect.json');acc=read(ROOT/'acceleration_checks.json');failure=read(ROOT/'failure_geometry.json');precision=read(ROOT/'precision_summary.json');panel=read(ROOT/'panel_summary.json');forced=read(ROOT/'forced_geometry_probe.json');refs=list(csv.DictReader((ROOT/'reference_directions.csv').open(encoding='utf-8')))
    lines=['KS reconstruction bias and full nuisance-procedure uncertainty','=============================================================','',
    'Scope: solver-free noise estimation, signal reconstruction, corrected weak',
    'moments and confidence-region calibration. The candidate library and weak-bank',
    'design remain fixed. This does not calibrate an adaptive structure search or',
    'ablate dyad. Constants and valid identities are retained. No candidate dynamics',
    'are integrated, and no production source is changed.', '',
    '1. Paired KS-positive diagnostic',
    '-------------------------------',
    'On three existing anchors45-47, reuse the same generating noise model and16',
    'noise increments per anchor. Change only ordinary versus correlation-aware',
    'GCV reconstruction; fit both noise estimators to identical fields within each',
    'pilot arm. These are procedure diagnostics, not coverage trials.', '',
    'Noise amplitude relative to its generating anchor estimate (min / median / max):']
    for r in factorial['summary']:
        values=' / '.join(f'{x:.5f}' for x in r['sigma_ratio_quantiles']);lines.append(f"  {r['method']:7s} {r['pilot']:10s}: {values}; near-unit fits {r['near_unit']}/{r['fields']}")
    lines += ['',
    'Changing the pilot removes most of the bootstrap noise inflation. The local',
    'estimator median changes1.07996 ->1.00034; temporal changes1.34852 ->1.03483.',
    'The six temporal near-unit-root fits disappear. The temporal estimator retains',
    'a small systematic discrepancy; accurate bootstrap parameters alone would',
    'still not establish calibrated regions. This identifies reconstruction bias',
    'as a major contributor to the previous KS-positive collapse.', '',
    '2. Fresh complete nuisance-procedure test',
    '----------------------------------------',
    '69 fresh observed datasets:15 iid systems plus4 core systems under positive',
    'and negative correlation, three independent noise seeds51-53 per case. Noise',
    'scale is the benchmark5% of clean signal SD. Each observed dataset gets its own',
    'temporal noise fit, correlation-aware pilot, covariance metric and cutoff.',
    'Fit noise and reconstruct all128 bootstrap fields. Covariance normalization',
    'uses16 fixed common Gaussian realizations and exact polynomial calculations.',
    'The paired modes differ only in whether normalization uses the anchor pilot',
    '(fixed_pilot) or each bootstrap field reconstructed anew (refit_pilot).',
    'Both modes center bootstrap errors on the generating anchor pilot. The123rd',
    'ordered statistic of128 sets the cutoff. All fits, cutoffs and regions are',
    'frozen before clean-field evaluation. Each observed dataset contributes ONE',
    'coverage indicator; bootstrap draws are never counted as independent trials.', '',
    f"Completion: {summary['datasets']}/{summary['expected']} datasets; {summary['result_statuses']}",
    f"Retained numerical warnings: {len(summary['warnings'])}.", '',
    'Observed datasets covered by operator bounds (local / broad / joint):']
    for profile in ('iid','positive','negative'):
        for mode in ('fixed_pilot','refit_pilot'):
            rr=[next(r for r in summary['summary'] if (r['profile'],r['mode'],r['bank'])==(profile,mode,b)) for b in ('local','broad','both')];lines.append(f"  {profile:8s} {mode:11s}: "+' / '.join(f"{r['covered']}/{r['datasets']}" for r in rr))
    lines += ['', 'KS specifically (local / broad / joint):']
    for profile in ('iid','positive','negative'):
        for mode in ('fixed_pilot','refit_pilot'):
            rr=[next(r for r in summary['systems'] if (r['system'],r['profile'],r['mode'],r['bank'])==('ks',profile,mode,b)) for b in ('local','broad','both')];lines.append(f"  {profile:8s} {mode:11s}: "+' / '.join(f"{r['covered']}/{r['datasets']}" for r in rr))
    lines += ['', 'Cases with at least one uncovered dataset under refitted reconstruction:']
    failures=[r for r in summary['systems'] if r['mode']=='refit_pilot' and r['covered']<r['datasets']]
    if not failures:lines.append('  None in this small outer-dataset screen.')
    for r in failures:lines.append(f"  {r['system']} / {r['profile']} / {r['bank']}: {r['covered']}/{r['datasets']}")
    lines += ['',
    '3. Effect of propagating reconstruction uncertainty',
    '--------------------------------------------------']
    for r in summary['discordance']:lines.append(f"  {r['bank']}: fixed-only coverage {r['fixed_only']}, refit-only {r['refit_only']}, both {r['both']}, neither {r['neither']}")
    for bank,qq in summary['cutoff_ratio_quantiles'].items():lines.append(f"  {bank} refit/fixed cutoff ratio min/10%/median/90%/max: "+', '.join(f'{x:.6g}' for x in qq))
    if all(r['fixed_only']==r['refit_only']==0 for r in summary['discordance']):
        lines += ['  Both normalization variants make identical coverage decisions on all',
                  '  observed datasets. This screen provides no coverage-based reason to pay',
                  '  for reconstruction inside every bootstrap normalization. It does not',
                  '  prove the variants equivalent on other data or at other noise levels.']
    lines += [f"  Joint reference geometries: {summary['geometry']}",
    '  Refitted/fixed coordinate-width ratio quantiles for pairs with bounded',
    '  outer regions under both methods: '+', '.join(f'{x:.6g}' for x in summary['width_ratio_quantiles']), '',
    'Reference-direction coverage and equation-zero acceptance are separate from',
    'uniform library-wide operator coverage. A bounded or accepted region alone',
    'does not prove calibration or precise coefficients. See reference_directions.csv',
    'and reference_geometry.csv; equations within one dataset are not independent',
    'coverage trials. The covariance changes themselves are in pilot_metric_effect.csv.', '',
    'Statistical noise-identification warning, separate from numerical convergence:',
    '  One of 10,752 bootstrap variable fits is extreme: ode/iid/seed52, draw112.',
    '  Marginal sigma is72.47 times its generating anchor value and rho=0.999916,',
    '  despite projected gradient1.60e-7. The correlation length is about37',
    '  record lengths, while temporal innovation amplitude changes only9.4%.',
    '  Curvature contrasts weakly constrain such long-range level variance.',
    '  The observed anchor itself estimated rho=0.516 under iid generating noise',
    '  and sigma about38.6% above its generating value. Optimizer convergence',
    '  therefore must not be treated as statistical noise identification.',
    '  The extreme draw ranks123rd locally, exactly selecting the cutoff.',
    '  A diagnostic leave-one-out calculation lowers local cutoffs by14.46%',
    '  (fixed pilot) and5.11% (refitted pilot); broad/joint cutoffs do not change.',
    '  The original draw is retained everywhere. These are sensitivity diagnostics,',
    '  not a license to delete finite fits or impose a truth-selected rho cap.',
    '  See noise_tail_diagnostic.json. Profile-based noise-identification checks',
    '  are a remaining solver-free task before production promotion.', '',
    'Post-calibration diagnostics for refitted-pilot bank failures:']
    missed=[r for r in failure['failures'] if r['mode']=='refit_pilot']
    if not missed:lines.append('  No individual-bank failures in this screen.')
    for r in missed:
        terms=', '.join(f"{t['term']} ({t['weight']:+.3g})" for t in r['leading_standardized_coordinates'])
        lines.append(f"  {r['system']}/{r['profile']}/seed{r['seed']}/{r['bank']}: observed/cutoff {r['observed_over_cutoff']:.6g}; bootstrap exceedances {r['bootstrap_exceedances']}/128.")
        lines.append('    Leading covariance-scaled coordinates: '+terms)
    badrefs=[r for r in failure['reference_failures'] if r['mode']=='refit_pilot']
    lines.append(f'  Reference-direction failures (noise or zero acceptance): {len(badrefs)}.')
    for r in badrefs:
        lines.append(f"    {r['system']}/{r['profile']}/seed{r['seed']}/{r['candidate']}/{r['bank']}: noise covered={r['noise_covered']}, zero accepted={r['zero_accepted']}; clean residual squared={r['clean_residual_squared']:.6g}, noise squared={r['noise_squared']:.6g}, radius squared={r['squared_radius']:.6g}.")
    lines += ['  Exceedance ranks are diagnostics, not certified p-values. Coordinates can',
              '  mix correlated library columns; they do not identify a unique causal term.', '',
              'Post hoc attribution using the known generating noise model:']
    correction=read(ROOT/'correction_diagnostic_summary.json')
    for ident in correction['diagnosed_records']:
        dd=read(ROOT/'correction_diagnostics'/(ident+'.json'));original=read(ROOT/'results'/(ident+'.json'))
        for r in dd['rows']:
            if all(original['coverage'][m][r['bank']] for m in original['coverage']):continue
            sc=r['scores'];lines.append(f"  {ident}/{r['bank']}: original score {sc['estimated_moments_estimated_covariance']:.6g}, oracle-correction-only score {sc['oracle_moments_estimated_covariance']:.6g}, oracle correction+covariance score {sc['oracle_moments_oracle_covariance']:.6g}; original refitted-pilot cutoff {r['original_cutoffs']['refit_pilot']:.6g}.")
    lines += ['  These diagnostic scores retain the original reconstructed pilot and do',
              '  not recalibrate any cutoff. They are not oracle coverage experiments.',
              '  Noise-parameter error alone does not explain every remaining miss.',
              '  Distinguishing tail realizations from systematic bootstrap error needs',
              '  further independent outer datasets; do not force every 95% bound to pass.', '',
    'Coefficient precision, beyond coverage and boundedness:',
    f"  {precision['zero_witnesses']}/{precision['nonzero_truth_coordinates']} nonzero reference coefficient coordinates in bounded outer regions have an explicit accepted zero-coefficient witness.",
    '  Each witness is checked against BOTH original joint quadratic inequalities.',
    '  An outer interval crossing zero alone is not counted as a witness. Failure',
    '  to find a witness among three proposals is not proof that zero is excluded.',
    '  Coordinates/equations within a dataset are not independent statistical trials.']
    for r in precision['summaries']:
        if r['zero_witnesses']:
            lines.append(f"  {r['system']}/{r['profile']}: {r['zero_witnesses']}/{r['nonzero_coefficients']} accepted zero witnesses; full width/absolute truth min/median/max: "+', '.join(f'{x:.6g}' for x in r['relative_full_width_quantiles']))
    if precision['recession_diagnostics']:
        lines += ['  Certified unbounded regions have a common coefficient direction whose',
                  '  data-residual curvature is smaller than its calibrated error allowance',
                  '  in both banks. Growing coefficients along that direction remains accepted.',
                  '  This is insufficient coefficient information at the tested noise/design;',
                  '  better calibration alone cannot turn it into precise identification.']
    lines += ['  Full coordinate bounds, explicit witnesses and recession diagnostics:',
              '  coefficient_precision.csv and precision_summary.json.', '',
              '4. Exploratory calibration over the fixed candidate panel',
              '---------------------------------------------------------',
              'This comparison was proposed after inspecting initial primary results.',
              'It reuses the archived draws and fits, restricting the operator maximum',
              'to the union of the existing candidate supports, with a maximum over all',
              'candidates and both banks. The same order-statistic cutoff rule applies.',
              'It does not account for adaptive searches outside that fixed panel.',
              'All gains/losses are exploratory and need independent outer validation.']
    for r in panel['summary']:
        lines.append(f"  {r['profile']}/{r['mode']}: panel-union coverage {r['covered']}/{r['datasets']}; original-library coverage {r['original_covered']}/{r['datasets']}; original-only {r['original_only']}, panel-only {r['panel_only']}.")
    lines.append(f"  Panel reference geometries: {panel['geometry']}")
    lines.append(f"  Geometry changes: {len(panel['geometry_changes'])}; reference-direction failures: {len(panel['reference_failures'])}.")
    for mode,qq in panel['cutoff_ratios'].items():lines.append(f'  {mode} panel/original cutoff min/median/max: '+', '.join(f'{x:.6g}' for x in qq))
    enclosures=read(ROOT/'panel_enclosures.json')
    lines += ['  A smaller panel cutoff can select a different enclosing ellipsoid.',
              '  In KdV the raw enclosure width increased by up to about 30 times,',
              '  despite tighter joint inequalities. This is an enclosure artifact.',
              '  Reevaluate the original positive combination at the new cutoff and',
              '  intersect its coordinate bounds with the new enclosure. Both contain',
              '  the unchanged joint region. All 300 component ellipsoids were checked.',
              '  These retained enclosures do not alter thresholds or coverage.']
    for mode,qq in enclosures['retained_width_quantiles'].items():lines.append(f'  {mode} retained/original width min/median/max: '+', '.join(f'{x:.6g}' for x in qq))
    lines += ['  All nine forced-KdV reference regions remain certified unbounded.',
              '  Along each archived recession direction, the cutoff would need a',
              '  reduction of about 59-83% merely to remove that one direction.',
              '  The panel restriction reduces these cutoffs only about 1-10%.',
              '  That is an explanatory calculation, not a license to shrink calibrated',
              '  bounds. Even removing one direction would not establish boundedness.',
              '  Clean normalized feature-Gram minimum eigenvalues are about 0.0156',
              '  locally and 0.00136 broadly: the current trajectory/weak design poorly',
              '  separates a combination of the reference terms. Test new informative',
              '  weak functions or additional trajectories with full recalibration.',
              '  See panel_summary.json and forced_geometry_probe.json.', '',
    '5. Numerical verification and runtime',
    '------------------------------------',
    'The direct implementation was interrupted before any case reached frozen',
    'calibration or clean evaluation. Its sources/partial logs are retained. It was',
    'replaced by equivalent algebra: temporal QR sufficient-statistic compression',
    'during profile optimization, and shared Wick noise features for both pilots.',
    'Final state-score certification and sigma profiling use original contrasts.',
    'The model, starts, bounds, convergence tolerances, streams, MC16, bootstrap128',
    'and cutoff rule remain unchanged. Tiny floating-point fit differences can',
    'occur; this is not a claim of bit-identical optimizer trajectories.',
    f"Covariance comparison: {acc['matrix_pairs']} pairs over15 systems, max relative error {acc['max_covariance_relative_error']:.3g}.",
    'Independent audits reconstruct scores via generalized eigenvalues, verify',
    'regions/certificates, replay exact stored inputs and noise streams, reestimate',
    'selected bootstrap fits, and compare metrics with individual-field evaluation.',
    'finish_audit.py checks the full queue after execution ends, preserving the',
    'live watcher summary as audit_live.json and auditing any unprocessed records.',
    'One Lorenz nonreference-region check failed because the historical auditor',
    'combined an eigensolve center/minimum with a separate LU inverse to construct',
    'boundary points. At condition number about 1.22e7, those ingredients can',
    'describe slightly different quadrics. An independent 80-digit calculation',
    'verified the stored endpoints to about 6.64e-10 on center/radius scale, using',
    'the existing condition-dependent forward-error criterion. No stored region',
    'or cutoff was changed; the failed record and diagnostic are preserved.',
    'The panel watcher also stopped before auditing two late-arriving records;',
    'finish_panel_audit.py retained its original report and checked both records.',
    'Cold-process pilot recomputation sometimes differs at relative7.43e-11.',
    'Exact original arrays were recovered for the diagnostic, and primary pilots',
    'are explicitly archived. Saved hashes remain exact; recomputation is checked',
    'numerically. Failed initial replay logs are preserved, not silently discarded.', '',
    'An optional native selected-inverse runtime probe is NOT used in this study.',
    'Four complete fits agreed closely and were faster, but ill-conditioned stress',
    'cases produced gradient discrepancies up to2.50e-4, above the convergence',
    'threshold. Strict numerical comparison therefore did not pass. This is not a',
    'validated drop-in replacement; native_checks.json and native_probe retain',
    'the characterization, strict checks and initial build/harness failures.', '',
    '6. Limits and decision',
    '----------------------',
    'Three observed datasets per case are a preliminary failure screen, not precise',
    '95% coverage certification. In particular3/3 does not establish95% reliability.',
    'The earlier percentages from many conditional actual draws cannot be pooled',
    'with these observed-dataset counts or presented as a paired percentage gain.',
    'The result applies to the existing trajectories, library/design,5% noise level',
    'and separable Gaussian AR1 family. Reconstruction and noise-model misspecification',
    'can remain outside that setting. MC16 and128 bootstrap draws also leave finite',
    'Monte Carlo uncertainty in each cutoff. No unconditional validity theorem is',
    'claimed. Independent outer replication is required before production promotion.',
    'Keep dyad provisional. Continue judging structure, weak-moment accuracy and',
    'coefficient-region precision separately. Do not remove valid identities to',
    'make a ranking appear unique, or tune per-system cutoffs to force95%.', '',
    'Artifacts: summary.json and system_coverage.csv contain all counts;',
    'full_pipeline_coverage.png/.pdf visualizes all23 cases; the paired diagnostic',
    'is ks_reconstruction_diagnostic.png/.pdf. method_notes.txt describes the full',
    'procedure. completion_audit.json and artifact_manifest.json record verification.', '']
    import re
    report='\n'.join(lines)
    report=re.sub(r'\b(anchors|and|changes|plus|seeds|benchmark|all|uses|The|of|over|relative|to|particular|establish|force|about|by|draw|is|gradient|only|ranks|not)(?=\d)',r'\1 ',report)
    report=report.replace('datasets:15','datasets: 15').replace('design,5%','design, 5%').replace('->1.','-> 1.')
    (ROOT/'findings.txt').write_text(report,encoding='utf-8');print('WROTE',ROOT/'findings.txt',flush=True)
if __name__=='__main__':main()
