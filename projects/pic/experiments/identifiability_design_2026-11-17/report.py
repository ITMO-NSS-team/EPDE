from research_context import *
from collections import Counter
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    design=[read(p) for p in sorted((ROOT/'results').glob('*.json'))];fresh=[read(p) for p in sorted((ROOT/'repair_results').glob('*.json'))];reg=[read(p) for p in sorted((ROOT/'repair_regression').glob('*.json'))]
    assert len(design)==9 and len(fresh)==3 and len(reg)==45
    screen=read(ROOT/'design_screen.json');ch=read(ROOT/'repair_characterization.json');arch=read(ROOT/'archive_repair_summary.json');failure=read(ROOT/'fresh_failure.json')
    dc={b:sum(d['coverage'][b] for d in design) for b in design[0]['coverage']};geometry={m:dict(Counter(c['geometry'] for d in design for c in d['joint'][m]['certificates'] if c['label']=='reference')) for m in ('original','replacement')}
    ratios=[d['joint']['replacement']['cutoff']/d['joint']['original']['cutoff'] for d in design];rr=[r for d in reg for r in d['rows']];fr=[dict(seed=d['seed'],draw=j,variable=v,original=r['original'][v]['model'],repaired=r['model'][v],anchor=d['model'][v],info=i) for d in fresh for j,r in enumerate(d['refits']) for v,i in r['info'].items()]
    tails={kind:sum(r[kind][0]/r['anchor'][0]>2 or max(abs(x) for x in r[kind][1])>.95 for r in fr) for kind in ('original','repaired')}
    out=dict(status='COMPLETE',design_coverage=dc,design_geometry=geometry,design_joint_cutoff_ratio=dict(min=min(ratios),median=float(np.median(ratios)),max=max(ratios)),
         regression=dict(records=45,variable_fits=len(rr),statuses=dict(Counter(r['info']['status'] for r in rr)),max_sigma_relative_change=max(abs(r['model'][0]/r['old_model'][0]-1) for r in rr),max_rho_change=max(abs(r['model'][1][0]-r['old_model'][1][0]) for r in rr),max_likelihood_gain=max(r['info']['contrast_count']*r['info']['improvement'] for r in rr)),
         fresh_ode_coverage={b:sum(d['coverage'][b] for d in fresh) for b in fresh[0]['coverage']},fresh_ode_fit_statuses=dict(Counter(r['info']['status'] for r in fr)),fresh_ode_extreme_diagnostic=tails,
         fresh_ode_material_likelihood_improvements=sum(r['info']['contrast_count']*r['info']['improvement']>1e-3 for r in fr),fresh_ode_diagnostic=failure,archive=arch,selected_bank=screen['selected'],source_sha256=sources('report.py'))
    dump(ROOT/'summary.json',out)
    rows=[dict(profile=d['profile'],seed=d['seed'],original_covered=d['coverage']['original'],replacement_covered=d['coverage']['replacement'],original_cutoff=d['joint']['original']['cutoff'],replacement_cutoff=d['joint']['replacement']['cutoff'],original_geometry=next(c['geometry'] for c in d['joint']['original']['certificates'] if c['label']=='reference'),replacement_geometry=next(c['geometry'] for c in d['joint']['replacement']['certificates'] if c['label']=='reference')) for d in design]
    with (ROOT/'design_comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');key='ode_seed52_bootstrap112';points=sorted([read(p) for p in (ROOT/'noise_profiles').glob(key+'_rho*.json')],key=lambda r:r['rho']);changed=next(r for r in ch['rows'] if r['key']==key);n=changed['info']['contrast_count'];best=changed['info']['objective']
    ax=axs[0];ax.plot([np.arctanh(r['rho']) for r in points],[n*(r['objective']-best) for r in points],'o-',ms=4,label='fixed-rho nuisance fits')
    ax.scatter(np.arctanh(changed['old_model'][1][0]),n*changed['info']['improvement'],s=65,marker='x',color='red',label='previous selected fit');ax.scatter(np.arctanh(changed['model'][1][0]),0,s=60,marker='*',color='green',label='all-start exact refinement')
    ax.set(title='ODE: optimizer missed a better basin',xlabel='atanh(noise correlation)',ylabel='-2 log likelihood loss',ylim=(-1,35));ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axs[1];x=np.arange(9);ax.bar(x,ratios,color=['#4477aa']*3+['#228833']*3+['#cc6677']*3);ax.axhline(1,color='black',ls='--',lw=1);ax.set(xticks=x,xticklabels=[d['profile'][:3]+' '+str(d['seed']) for d in design],ylabel='Replacement / original joint cutoff',title='KdV: smaller cutoff, still unbounded');ax.tick_params(axis='x',rotation=45);ax.text(.5,.94,'9/9 covered; 9/9 unbounded in both procedures',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.savefig(ROOT/'research_results.png',dpi=180);fig.savefig(ROOT/'research_results.pdf');plt.close(fig)
    lines=['Solver-free optimizer and weak-design follow-up','==============================================','',
      '1. ODE tail diagnosis and repair',
      'The old ode iid seed52 bootstrap112 estimate (sigma10.3437, rho0.999916)',
      f"was a poor local optimum. Exact refinement of every original start gives sigma{changed['model'][0]:.8g}, rho{changed['model'][1][0]:.8g}; -2log likelihood improves by {n*changed['info']['improvement']:.6f}.",
      'Two band optimizations stopped early. Their initial values were worse than',
      'the near-unit basin, so the old routine discarded them before exact-state',
      'refinement. After refinement they both reach the better moderate-rho basin.',
      'The repair changes basin selection only: same model, bounds, contrasts and',
      'spatial estimator. No correlation cap, discarded draw or oracle start.',
      'All 84 fixed-rho profile fits passed the nuisance score check. Flat behavior',
      'near rho=1 does occur, but that tail is not the best fit in the failed draw.',
      'The grid supports are descriptive, not calibrated confidence intervals.',
      f"Default-set regression: 45 datasets / {len(rr)} variable fits; statuses {out['regression']['statuses']}.",
      f"Maximum relative sigma change {out['regression']['max_sigma_relative_change']:.6g}; maximum rho change {out['regression']['max_rho_change']:.6g}.",
      f"Complete paired archived128 refits: cutoff ratios {arch['cutoff_ratios']}; coverage {arch['coverage']}.",
      f"Materially improved archived draws (>1e-3 likelihood units): {arch['materially_changed_draws']}.",
      f"Fresh ODE seeds54..56: coverage {out['fresh_ode_coverage']} out of3;128 refits per seed.",
      f"Fresh refit statuses {out['fresh_ode_fit_statuses']}; extreme-fit diagnostic counts {tails} out of384.",
      'Coverage here concerns the full-library weak-error bound, not exact-equation',
      'recovery. All three ODE reference equations remain accepted by both banks.',
      'The local miss is seed55: score/cutoff1.11164, with3/128 bootstrap scores',
      'at least as large. The old fitter on the same fields also misses; its cutoff',
      'differs by only9.8e-8. Thus this miss was not introduced by the optimizer fix.',
      'Estimated sigma is98.28% of truth, but estimated rho=-0.12773 versus true0.',
      'Posthoc known-noise correction and normalization lower the local score',
      'to2.24856, below the unchanged2.68682 cutoff. This shows sensitivity to',
      'noise nuisance estimation; it is not an oracle-based repair or a calibrated',
      'coverage result. The finite sample is too small to diagnose systematic',
      'undercoverage versus an expected tail event.',
      'The old/new fit diagnostic uses identical fields generated by the repaired',
      'observed model. It is not a separate old-procedure coverage comparison.',
      'The observed archived ode seed52 rho0.516 estimate is unchanged. Sampling',
      'ambiguity or noise-model misspecification still needs study. Checking every',
      'original basin is a stronger safeguard, not a proof of a global optimum.','',
      '2. Weak-window replacement test',
      'A fixed12-bank menu was scored using only the archived iid seed51 pilot',
      'and estimated noise, with all stochastic candidate columns projected off',
      'the deterministic columns. Constants and valid identities were retained.',
      f"Selected bank: {screen['selected']} (full time, half-space windows, modulation k2).",
      'Reference labels, coefficients and clean moments were not used to select it.',
      'The log-det proxy score rose28.2338->28.6932. After freezing the choice,',
      'clean reference feature minimum correlation eigenvalue rose0.001362->0.002183,',
      'which still indicates severe near-collinearity. More high frequencies alone',
      'reduced the proxy score in the fixed menu.',
      f"Nine fresh datasets (iid,+.4,-.4; seeds54..56): coverage {dc}.",
      f"Joint cutoff ratio range {min(ratios):.4f}..{max(ratios):.4f}; median {np.median(ratios):.4f}.",
      f"Reference geometry: {geometry}.",
      'The cutoff is dimensionless and bank-dependent; its reduction does not',
      'establish a smaller coefficient region. Explicit common negative directions',
      'certify unboundedness under both procedures in all nine datasets.',
      'Therefore this bank replacement has not fixed the precision problem.','',
      '3. Decision and limitations',
      'Keep the all-start exact refinement as a research optimizer fix and validate',
      'its complete uncertainty procedure more widely before production promotion.',
      'Keep dyad provisional; these findings do not justify replacing it. Retain',
      'separate checks of weak accuracy, structure and coefficient precision.',
      'Next uncertainty work should test noise-parameter sensitivity on more',
      'independent observed datasets. Next weak-design work should target poorly identified coefficient',
      'directions rather than aggregate signal strength. Multiple informative',
      'trajectories or lower measurement noise may be needed if the same trajectory',
      'cannot constrain those directions at the chosen uncertainty level.',
      'Three fresh datasets per noise case are a development screen, not95% coverage',
      'certification. The library is fixed; adaptive EPDE-search coverage is untested.',
      'All work remains solver-free. Previous artifacts and production are unchanged.',
      'Numerical/replay audit details: audit_design.json, audit_profiles.json,',
      'audit_repair.json, audit_optimizer.json and completion_audit.json.','']
    import re
    report='\n'.join(lines);report=re.sub(r'\b(sigma|rho|out of|fixed|by|rose|to|seed|archived|datasets|draws|than|not|score/cutoff|with|only|is|true|unchanged)(?=\d)',r'\1 ',report);(ROOT/'findings.txt').write_text(report,encoding='utf-8')
    print('REPORT COMPLETE',out['fresh_ode_coverage'],tails,arch['cutoff_ratios'],flush=True)

if __name__=='__main__':main()
