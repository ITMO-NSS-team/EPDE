"""Verify completed artifacts and write the human-readable experiment report."""
import run_screen as R
import analyze_screen as A
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy


def main():
    root=R.ROOT
    initial=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'audited').glob('*_b32.json')]
    follow=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'followup128').glob('*_b128.json')]
    names=R.datasets.names('core')
    assert len(initial)==45 and len(follow)==15
    assert {(x['system'],x['seed']) for x in initial} == {(n,s) for n in names for s in range(3)}
    follow_names=['ac','kdv_cossin','pde_compound','pde_divide','ns']
    assert {(x['system'],x['seed']) for x in follow} == {(n,s) for n in follow_names for s in range(3)}
    original_hash=hashlib.sha256((root/'run_screen_initial.py').read_bytes()).hexdigest()
    current_hash=hashlib.sha256((root/'run_screen.py').read_bytes()).hexdigest()
    for x in initial:
        assert x['source_sha256']==original_hash
        raw_path=root/'results'/f"{x['system']}_seed{x['seed']}_b32.json"
        assert x['partition_audit']['input_sha256']==hashlib.sha256(raw_path.read_bytes()).hexdigest()
    for x in initial+follow:
        idx=[set(g['indices']) for g in x['groups']]
        assert not (idx[0]&idx[1] or idx[1]&idx[2] or idx[0]&idx[2])
        assert set.union(*idx)==set(range(x['shape'][0]))
        assert np.isfinite(x['instability']).all()
        for method,counts in x['aggregates'].items():
            for entry in counts.values():
                for metric in ('cross','ordinary','oracle','uncorrected_cross'):
                    assert np.isfinite(entry[metric]['scores']).all()
    for x in follow:
        assert x['source_sha256']==current_hash
        previous=next(y for y in initial if (y['system'],y['seed'])==(x['system'],x['seed']))
        np.testing.assert_allclose(x['instability'],previous['instability'],rtol=1e-12,atol=1e-12)
        assert x['data_sha256']==previous['data_sha256']
        assert [g['observed_sha256'] for g in x['groups']]==[g['observed_sha256'] for g in previous['groups']]
    base_rows=A.summarize(initial);follow_rows=A.summarize(follow)
    with (root/'followup128'/'per_equation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(follow_rows[0]));w.writeheader();w.writerows(follow_rows)
    final_cases={(x['system'],x['seed']):x for x in initial}
    final_cases.update({(x['system'],x['seed']):x for x in follow})
    combined=[]
    for x in final_cases.values():
        combined.extend(r for r in A.summarize([x]) if r['draws']==x['draws'])
    with (root/'latest_per_equation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    summary={}
    for label,rows in [('primary32',[r for r in base_rows if r['draws']==32]),('with_selected128',combined)]:
        summary[label]={method:{score:sum(bool(r[score+'_valid_front']) for r in rows if r['bandwidth']==method)
                                for score in ('cross','ordinary','oracle','uncorrected_cross')}
                        for method in ('gcv','loo')}
    R.dump(root/'final_summary.json',summary)
    table=[]
    for name in names:
        for slot in sorted({r['slot'] for r in combined if r['system']==name}):
            a=[r for r in base_rows if r['system']==name and r['slot']==slot and r['draws']==32 and r['bandwidth']=='loo']
            b=[r for r in combined if r['system']==name and r['slot']==slot and r['bandwidth']=='loo']
            counts=[sum(bool(r[s+'_valid_front']) for r in rows) for rows,s in
                    [(a,'ordinary'),(a,'cross'),(b,'ordinary'),(b,'cross')]]
            table.append(f"{name:18s} {slot:4d}       {counts[0]}/3          {counts[1]}/3          {counts[2]}/3          {counts[3]}/3")
    diagnostics=json.loads((root/'audited'/'fit_diagnostics.json').read_text())
    controls=[]
    for name in ['kdv_cossin','ks','pde_compound','pde_divide']:
        rr=[r for r in diagnostics if r['system']==name]
        controls.append(f"{name:18s} {np.mean([r['fits']['clean_split_ols']['clean_energy'] for r in rr]):.6g}"
                        f"    {np.mean([r['fits']['clsm']['clean_energy'] for r in rr]):.6g}")
    elapsed=sum(x['elapsed_seconds'] for x in initial)
    follow_elapsed=sum(x['elapsed_seconds'] for x in follow)
    text=f'''All 15 core systems: residual cross-product screen, 2026-11-16

Result: the improvement extends beyond Kepler, but is not universal. The
primary 32-draw run retains a catalog-valid equation on the Pareto front in
51/60 equation-seed cases, versus 46/60 for matched ordinary residual energy.
GCV and exact-LOO bandwidths give the same aggregate retention in this panel.
Higher-draw checks on five systems remove the marginal Navier-Stokes gain:
the combined result is 50/60 versus 46/60, with 128 draws on those five and
32 draws on the other ten. This is NOT an all-15 128-draw result.

All 45 system-seed cases completed: 15 saved default datasets, 3 independent
5%-noise seeds, 20 equation slots per seed. Each equation's two objectives
were evaluated separately. Catalog equivalents were credited; an unregistered
competitor is not automatically a false law. No equation solver was used.

Where it helps:
- pde_divide: 3/3 retained, versus 0/3 for ordinary energy, at both 32 and 128
  draws. The competing time-polynomial and time-harmonic fits have a smaller
  measurement-noise contribution. Cross-products remove enough of that
  advantage to preserve the governing structure.
- pde_compound: 1/3 retained, versus 0/3, at both draw budgets. The other seeds
  still favor models dropping a physical term. The clean-fit energies of the
  two dropped-term models in seed 0 are 0.304 and 0.977, versus 6.7e-8 for the
  full structure; these competitors are not exact clean weak identities.
- NS momentum slot 1, seed 1: the apparent 32-draw gain disappears at 64 and
  128 draws. Do not present this as established improvement.

Where it fails:
- Forced KdV (kdv_cossin) and KS: no catalog-valid candidate survives in any
  seed. Forced KdV's failures persist at 128 draws; KS was screened at 32.
- Allen-Cahn: one seed remains dominated. Increasing draws removes the
  missing-diffusion competitor but leaves an added cos(2t) competitor.
- Nonlinear diffusion (pde_compound): two of three seeds remain dominated.

Per-equation retention out of three seeds (LOO; GCV totals are identical):
System             Slot   Energy B32    Product B32    Energy latest Product latest
{chr(10).join(table)}

Latest uses B128 only for ac, kdv_cossin, pde_compound, pde_divide and ns.
Catalog-valid survival means at least one reference/registered alternative
lies on the finite candidate panel's nondominated front. It is not unique
identification, coefficient accuracy, or successful evolutionary recovery.

Why the hard PDEs still fail:
Controls fit the reference structure to clean split/reconstructed data, then
evaluate it on the clean field. The same control evaluates coefficients
fitted to noisy data by the tested CLSM method. The following are normalized
clean weak residual energies, averaged over the three rotations of seed 0:
System             Clean-split fit    Noisy CLSM fit
{chr(10).join(controls)}
The clean-split controls recover the hard PDEs accurately, whereas noisy
fits remain inaccurate even on clean fields. Thus time splitting alone is
not the main obstruction in these controls. High-order derivative and
nonlinear weak features remain sensitive to measurement noise; the current
bootstrap correction and fitting windows do not resolve their estimation.
These diagnostics support a fitting/measurement bottleneck as well as a
discrepancy bottleneck. They do not prove one unique source of the error.

Recommended next experiment: keep dyad fixed and use the same finite panel
to test a common solver-free weak coefficient estimator/window rule on KS,
forced KdV and pde_compound. Require the reference's clean-field residual
with noisy-fit coefficients to improve before judging discrepancy ranking.
Then compare both discrepancy scores in a fresh search across all 15.

Scope and reproducibility:
- Fixed diagnostic candidate panel, not a full EPDE evolutionary run. It
  includes missing/extra terms, registered alternatives, time/trig fits and
  alternate first-order targets. Trig competitors use the fixed 2t menu plus
  catalog forcing terms; this is not an exhaustive frequency search.
- PDE adaptation differs from the original Kepler protocol: pooled temporal
  splines, spatial DCT shrinkage, and a common spectral-resolution support
  rule. The instability axis is the full-record raw production dyad, not
  the earlier Kepler CLSM dyad. See protocol.txt for every choice.
- The initial harness inherited a width-6 dyad setting while the panel's
  widest candidate has 8 fitted features. analyze_screen.py recomputed ALL
  45 instability axes using the common width and refreshed every dominance
  result. Use audited/, not the initial results/, for B32 conclusions.
  Discrepancy values and coefficients were unaffected. run_screen.py now
  sets the common width before fitting. run_screen_initial.py preserves the
  exact initial experiment source. Smoke outputs are timing/debug only.
- All raw split index sets are disjoint and exhaustive. Reconstructions,
  bandwidths and bootstrap fields use only their own training/evaluation
  group. Nonlinear terms are rebuilt. Signed cross-products are not clipped.
- Numerical checks: spline max error 6.64e-14 versus SciPy at fixed lambda;
  2D zero-mode weak integral error 2.04e-14; exact LOO agrees with brute
  refits; all 15 catalog equations parse; final score arrays are finite.
- The 16/32 and 64/128 comparisons measure computational sensitivity, not
  confidence intervals. Three rotations reuse data and are not independent
  experiments. No full uncertainty interval over training, bandwidth and
  bias estimation was computed. Marginal nondominance is not strong evidence.
- One worker, one BLAS thread, below-normal priority. Primary calculation
  time summed over cases: {elapsed:.1f}s; follow-ups: {follow_elapsed:.1f}s.

Artifacts:
  audited/per_equation.csv             complete primary table
  audited/structural_survival.png      primary figure (also SVG)
  audited/fit_diagnostics.json         clean/noisy coefficient controls
  followup128/per_equation.csv         five-system sensitivity checks
  latest_per_equation.csv              mixed-budget table, draw count explicit
  final_summary.json                  machine-readable primary/latest counts
  manifest.json                       sources, environment and checks
'''
    (root/'findings.txt').write_text(text,encoding='utf-8')
    sources=[root/f for f in ['run_screen.py','run_screen_initial.py','analyze_screen.py','diagnose_fits.py','finalize.py','protocol.txt']]
    sources += [R.REPO/p for p in ['epde/preprocessing/weak_form.py','epde/operators/common/survival.py',
        'epde/interface/search_config.py','projects/pic/epde_bench/datasets.py','projects/pic/epde_bench/problem.py',
        'projects/pic/epde_bench/reconstruction.py','projects/pic/epde_bench/metrics.py']]
    R.dump(root/'manifest.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        checks=dict(primary_cases=45,followup_cases=15,source_hashes_match=True,split_indices_disjoint=True,
                    same_observed_records=True,finite_scores=True,same_dyad_axis_across_budgets=True),
        source_sha256={str(p.relative_to(R.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    print(json.dumps(summary,indent=2))
    print('Verified 45 primary + 15 follow-up cases; wrote findings and manifest.')


if __name__=='__main__':main()
