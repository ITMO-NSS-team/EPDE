from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    pairs=read('paired_summary.json')['summary'];summary=read('summary.json')
    details=read('detailed_analysis.json');axes=read('axis_diagnostic.json')['rows']
    attribution=read('compound_attribution_summary.json')['rows'];audit=read('audit.json')
    systems=list(csv.DictReader((ROOT/'system_coverage.csv').open(encoding='utf-8')))
    table=['profile     bank    pairs   AR2(%)    specified-family control(%)']
    for r in pairs:
        table.append(f"{r['profile']:<11} {r['bank']:<7} {r['complete_pairs']:<7} {100*r['candidate_coverage']:<9.2f} {100*r['control_coverage']:.2f}")
    iid=['system             AR2 local(%)  AR2 broad(%)  control local/broad(%)']
    lookup={(r['system'],r['profile'],r['model_kind'],r['bank']):float(r['operator_coverage']) for r in systems}
    for name in sorted({r['system'] for r in systems if r['profile']=='iid'}):
        v=[100*lookup[name,'iid',k,b] for k,b in [('ar2','local'),('ar2','broad'),('iid','local'),('iid','broad')]]
        iid.append(f'{name:<18} {v[0]:<13.2f} {v[1]:<13.2f} {v[2]:.2f}/{v[3]:.2f}')
    pure=['n       AR1 curvature   AR2 finite difference   AR2 analytic score']
    for n in (301,320,1001):
        v=[next(r['rho_rmse'] for r in details['pure_noise'] if (r['method'],r['n'],r['rho'])==(m,n,.4)) for m in ('ar1','ar2_fd','ar2_score')]
        pure.append(f'{n:<7} {v[0]:<15.6f} {v[1]:<23.6f} {v[2]:.6f}')
    attr=['seed bank    original/256  oracle correction/256  bias-score/cutoff']
    for r in attribution:
        attr.append(f"{r['seed']:<4} {r['bank']:<7} {r['original_coverage_count']:<13} {r['oracle_correction_coverage_count']:<22} {r['analytic_bias_cutoff_ratio']:.5f}")
    axis=['system       variable axis status     sigma/true  shared sigma/true']
    for r in axes:
        axis.append(f"{r['system']:<12} {r['variable']:<8} {r['axis']:<4} {r['status']:<10} {r['sigma_ratio']:<11.6f} {r['global_sigma_ratio']:.6f}")
    geom=[]
    for kind in ('ar2','iid'):
        r=next(r for r in summary['summary'] if (r['profile'],r['model_kind'],r['bank'])==('iid',kind,'both'))
        geom.append(f"{kind}: {r['exact_anchor_accepted']}/{r['reference_cases']} exact anchors accepted; {r['geometry']}.")
    s=details['summary']
    text='''AR2 signal/noise study: solver-free EPDE, 2026-11-17

Decision
--------
The more flexible signal-curvature model substantially improves the previous
AR1-curvature results, especially KS and NS. It is still not a universal
replacement for specified-family noise estimation. Compound retains a large
nonlinear correction bias; KS local coverage remains below nominal; NS is now
conservative. Keep dyad provisional for structural ranking. This round does
not run a new dyad ablation and gives no evidence that replacing dyad fixes
these covariance errors. Explicit constants and valid identities are retained.

What changed
------------
The nuisance second-difference signal covariance is stationary AR2, while
observation noise remains AR1. Exact finite-boundary banded likelihoods retain
the informative contrasts. Noise amplitude is shared across axes of a variable;
curvature and noise correlations vary by axis. Axis likelihood sums are
composite likelihoods. No differential-equation solver or candidate integration
is used. method_notes.txt gives equations, boundary terms and numerical details.

There were two distinct changes: the AR2 statistical family, and a subsequent
numerical repair using an analytic likelihood gradient. The original central
finite-difference implementation and failed attempts remain archived. On the
23-case screen (28 raw variables), it selected a noise-boundary VDP fit despite
better unsuccessful starts. The analytic-score version returns usable fits for
all28 raw and28 noise-only screen variables. This does not establish optimality.

Numerical checks and remaining limitations
------------------------------------------
288 dense covariance checks,96 nested AR1 reductions,32 dense-gradient checks,
selected-inverse checks and synthetic unit/trend invariance all pass. The largest
gradient discrepancy is4.13e-7 or less. Gradient repair keeps the same bounds,
four prescribed starts and statistical family. Maximum iterations changes from
300 to500; no per-system choice is made from coverage or clean truth.
'''
    text+=f'''
Retrospective primary optimization diagnostics: {s['primary_variable_fits']} variable fits;
{s['primary_projected_gradient_above_1e_minus3']} have projected-gradient mapping above1e-3, and
{s['primary_selected_gap_above_1e_minus5']} selected fits have normalized objective above an unsuccessful start
by more than1e-5. Mapping quantiles(min,median,p90,p99,max):
{s['projected_gradient_quantiles']}.
The mapping depends on optimizer coordinates. 'status ok' means the frozen
acceptance rule returned a model, not a certified stationary or global optimum.
Every result remains in the primary evaluation; no post hoc replacement occurs.

Paired pure-noise comparison
---------------------------
576 shared fields: n301/320/1001, rho0/+0.4/-0.4,64 draws per cell, sigma0.2.
Both AR2 implementations have0/576 incomplete fits. The preceding AR1-curvature
model has1/576 optimizer failure. Positive-rho correlation RMSE:
'''+ '\n'.join(pure)+'''
Full nine-cell results, including amplitude medians, are in
pure_noise_comparison.csv. Exact fields are hash-matched to the prior experiment.
No oracle signal information is used in the raw estimator. Accurate pure-noise
point estimates do not establish covariance adequacy in the presence of signal.

Fresh primary test
------------------
138/138 records complete; all processes exit0 and independent case audits PASS.
All15 default systems with iid noise, plus AC/forcedKdV/KS/wave with rho+/-0.4.
Three fresh anchors per system/profile, seeds39..41:69 distinct fitted anchors,
paired estimator/control records. Every model has256 calibration,256 check and
256 evaluation fields. Added noise amplitude is5% of the clean-field SD.
Pilot, weak candidate libraries, local/broad banks, correction algebra and
confidence-region machinery are inherited unchanged. Fits, metrics, cutoffs
and regions are frozen before clean evaluation. Noise-parameter uncertainty
is held fixed; it has not been propagated in this study.

Nominal95% simultaneous operator coverage, matched pairs:
'''+ '\n'.join(table)+'''
The iid control fixes rho=0 and estimates amplitude. The colored control fixes
rho to its generating value and estimates amplitude. These are specified-family
controls, not inferred correlations. Means weight anchors equally. Three
anchors per system do not establish unconditional95% validity; the256 evaluation
fields share a fitted anchor. Reference-direction coverage is a different,
less demanding quantity than full-library simultaneous operator coverage.

IID breakdown across all15 systems:
'''+ '\n'.join(iid)+'''
KS local coverage is84.38% iid,87.76% positive and83.46% negative; broad coverage
is92.58%,94.92%,92.71%, respectively. The previous AR1-curvature study had zero
local coverage on all nine KS anchors. NS now has99.61% local and99.87% broad
coverage, compared with near2% local coverage for the preceding completed fits.
These cross-study comparisons use different anchors and are descriptive, not
a paired statistical effect estimate. The controls within this study are paired.
The broad iid mean hides compound's severe failure; high average/local coverage
does not validate every system or direction. NS's higher coverage is conservative,
not evidence of sharper coefficient estimates.

Compound: correction-bias attribution
-------------------------------------
Only the actual-field Gaussian correction is changed to true covariance.
Pilot, fitted calibration/metric, cutoff and evaluation draws remain fixed.
This is an oracle diagnostic, not a deployable repair.
'''+ '\n'.join(attr)+'''
Broad coverage rises from28/768=3.65% to758/768=98.70%. The analytic correction
bias alone is1.26..1.97 times the broad cutoff in operator-score units. Thus
mean correction error remains sufficient to explain the principal failure.

Axis diagnostic: where the shared noise fit is distorted
--------------------------------------------------------
Old iid anchor36, all axes of compound/KS/NS (13 fits), using the same AR2 family
and analytic score. Each axis has a separate amplitude for this diagnostic.
The fit is frozen before clean curvature and true-noise ratios are evaluated.
'''+ '\n'.join(axis)+'''
For compound, the temporal-axis amplitude is within1% of truth; the spatial
axis is about19.5% high, while the shared estimate is7.6% high. The top1% of
clean spatial curvature entries contain67.1% of its energy. Spatial curvature
energy is9.52% of expected injected-noise curvature energy, versus0.0080% in
time. These diagnostics support signal/noise misallocation by the stationary
curvature representation. They do not establish whether the localized roughness
is physical or numerical. A missing constant curvature mean explains only
about6.6% of spatial curvature energy; no quadratic-mean variant was tested.

Naively selecting an axis is not validated: NS/u axis1 reaches a noise boundary
with amplitude ratio837.6, while NS/v axis1 estimates ratio1.50. All these
failures remain visible. Neither clean curvature nor true-sigma ratios are used
to choose an axis in the primary estimator. A robust or locally varying curvature
model with observable adequacy diagnostics is a next hypothesis, not a result.

Coefficient compatibility
-------------------------
'''+ '\n'.join(geom)+'''
Exact references remain compatible even where simultaneous operator coverage
is poor. This does not establish unique support, narrow coefficients or valid
uncertainty in all library directions. Same-target superset ambiguity remains.

Verification and artifacts
--------------------------
'''+f"Main audit PASS: {audit['totals']}.\n"+'''
Every primary noise fit is independently rerun; paired fields, moment matrices,
operator scores, registered directions, regions and intersection certificates
are checked. Follow-up audit PASS:1264 likelihood,632 gradient,688 field replays,
962 hash links,46 screen records and576 paired pure-noise draws.
Axis audit PASS:13 fits (including the boundary),65 hash links.
Attribution audit PASS:3 records,1536 operator scores,12 matrix replays.
The analysis-only dictionary/list schema mismatch is repaired in a separate
analyze_final.py; the original hashed harness remains untouched. No estimator
output or primary statistic was changed by this reporting repair.
Sources, inputs, failed starts, protocols, frozen models, arrays, CSVs and plots
are retained. Completion audit and final artifact manifest accompany this report.
No production source file or prior completed experiment was edited; no commit.

Next research decision
----------------------
Keep specified-iid estimation when the noise family is supplied. AR2 is a useful
research improvement but not ready as a universal default. Before parameter
uncertainty propagation, address (1) numerical conditioning and convergence,
(2) localized/nonstationary signal curvature and observable axis disagreement,
and (3) residual KS local calibration. Evaluate bias correction and covariance
shape separately on fresh anchors. Preserve dyad, constants and valid identities;
an extra instability penalty cannot repair biased weak moments by itself.
'''
    (ROOT/'findings.txt').write_text(text,encoding='utf-8')
    names=['paired_summary.json','summary.json','detailed_analysis.json','axis_diagnostic.json','compound_attribution_summary.json','audit.json','system_coverage.csv','method_notes.txt','audit_followups.json','audit_axis.json','audit_attribution.json','checks.json','checks_score.json']
    provenance=dict(source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256={str(ROOT/n):sha(ROOT/n) for n in names},generated_sha256={str(ROOT/'findings.txt'):sha(ROOT/'findings.txt')})
    (ROOT/'report_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    master=ROOT.parent/'solver_free_research_2026-11-16.txt';marker='ar2_signal_noise_2026-11-17/findings.txt'
    if marker not in master.read_text(encoding='utf-8'):
        with master.open('a',encoding='utf-8') as f:
            f.write('\n\n2026-11-17: AR2 signal/noise follow-up\n'+marker+'\n')
            f.write('Stationary AR2 nuisance curvature plus AR1 observation noise. Exact banded\nlikelihood and analytic-score repair; initial finite-difference version retained.\n23 screen cases per version,576 paired pure-noise fields per version.\n138/138 fresh records complete (seeds39..41); all primary/diagnostic audits PASS.\n')
            for r in pairs:
                if r['bank']!='both':f.write(f"{r['profile']}/{r['bank']}: AR2/control {100*r['candidate_coverage']:.2f}/{100*r['control_coverage']:.2f}% on{r['complete_pairs']} pairs.\n")
            f.write('KS local83..88% (previous study0); NS99.6/99.9% local/broad, conservative.\nCompound broad3.65% ->98.70% with oracle actual-field correction only.\nCompound axis diagnostic: time sigma1.0066x truth, space1.1949x, shared1.0763x.\nTop1% spatial clean curvature holds67.1% energy; localized signal misallocation\nremains a plausible cause. Axis selection unvalidated: NS/u axis1 boundary.\n14/84 primary variable fits have projected-gradient mapping>1e-3;6 have a\nlower-objective unsuccessful start by>1e-5. Optimizer success is not certification.\nIID exact references60/60 accepted;57 bounded outer/3 unbounded.\nNo production change or dyad replacement. Keep specified-family control where\nknown. Next: conditioning/convergence, robust or nonstationary curvature,\nresidual KS local calibration; noise-parameter uncertainty remains unpropagated.\n')
    print('Saved findings, provenance and research log.')

if __name__=='__main__':main()
