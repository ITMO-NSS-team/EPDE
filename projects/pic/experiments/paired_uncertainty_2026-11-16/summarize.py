import run_paired as P
R,A,np=P.R,P.A,P.np
from pathlib import Path
import csv,json,hashlib


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    paths=sorted((P.ROOT/'results').glob('*_seed*.json'))
    data=[json.loads(p.read_text()) for p in paths]
    assert len(data)==45
    assert {(d['system'],d['seed']) for d in data}=={(n,s) for n in R.datasets.names('core') for s in range(3)}
    pairs=[];exact=[];sensitivity=[];equations=[];checks=[]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(P.__file__),Path(P.Q.__file__),Path(P.E.__file__)]}
    for d in data:
        assert d['source_sha256']==hashes
        path=P.Q.ROOT/'unit_results'/f"{d['system']}_seed{d['seed']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==d['baseline_sha256']
        old=json.loads(path.read_text());n=len(old['candidates']);scores=d['point_scores'][:n]
        assert np.isfinite(scores).all()
        z=np.load(P.ROOT/'results'/f"{d['system']}_seed{d['seed']}_samples.npz")
        for name in z.files:assert np.isfinite(z[name]).all()
        z.close()
        for c in old['candidates']:
            if c['label']!='reference':continue
            slot=c['slot'];front=A.frontier(scores,old['instability'],old['candidates'],slot)
            previous=A.frontier(old['aggregates'][f"full_continuous_clsm_{old['draws']}"]['cross']['scores'],
                                old['instability'],old['candidates'],slot)
            assert front==previous
            primary=[p for p in d['models']['128']['pairs'] if p['reference']==c['id']]
            dom=[p for p in primary if p['point_dominator']]
            equations.append(dict(system=d['system'],seed=d['seed'],slot=slot,valid_front='|'.join(front),
                primary_dominators=len(dom),primary_dominators_with_95_below_zero=sum(p['interval95'][1]<0 for p in dom),
                primary_dominators_better_on_clean=sum(p['oracle_gap']<0 for p in dom)))
        for p in d['models']['128']['pairs']:
            registered=next(c['registered'] for c in old['candidates'] if c['id']==p['candidate'])
            row=dict(system=d['system'],seed=d['seed'],candidate_registered=registered,**p)
            assert np.isfinite([p['gap'],p['oracle_gap'],p['paired_sd']]).all()
            for level in ('90','95'):
                row[f'lo{level}'],row[f'hi{level}']=row.pop('interval'+level)
            pairs.append(row)
        for e in d['models']['128']['exact']:
            row=dict(system=d['system'],seed=d['seed'],**e)
            row['lo95'],row['hi95']=row.pop('interval95');exact.append(row)
        if '64' in d['models']:
            half={(p['reference'],p['candidate']):p for p in d['models']['64']['pairs']}
            for p in d['models']['128']['pairs']:
                if not p['point_dominator'] or not p['reference'].endswith(':reference'):continue
                h=half[p['reference'],p['candidate']]
                sensitivity.append(dict(system=d['system'],seed=d['seed'],reference=p['reference'],candidate=p['candidate'],
                    sd64=h['paired_sd'],sd128=p['paired_sd'],sd_ratio=h['paired_sd']/p['paired_sd'] if p['paired_sd'] else None,
                    margin64=h['standardized_margin'],margin128=p['standardized_margin'],
                    interval64=str(h['interval95']),interval128=str(p['interval95'])))
        checks.append(d['frozen_score_max_scaled_error'])
    primary=[p for p in pairs if p['reference'].endswith(':reference')]
    dom=[p for p in primary if p['point_dominator']]
    wrong=[p for p in dom if p['oracle_gap']>0]
    eligible=[p for p in primary if p['dyad_eligible']]
    definite=[p for p in eligible if p['hi95']<0 or p['lo95']>0]
    summary=dict(cases=45,equation_seed_trials=len(equations),unchanged_valid_retention=sum(bool(e['valid_front']) for e in equations),
        primary_comparisons=len(primary),registered_comparisons=len(pairs),primary_point_dominators=len(dom),
        primary_point_dominators_95_below_zero=sum(p['hi95']<0 for p in dom),
        primary_point_dominators_worse_on_clean=len(wrong),
        unregistered_primary_point_dominators=sum(not p['candidate_registered'] for p in dom),
        unregistered_dominators_worse_on_clean=sum(not p['candidate_registered'] for p in wrong),
        wrong_dominators_95_below_zero=sum(p['hi95']<0 for p in wrong),
        dyad_eligible_primary_comparisons=len(eligible),
        eligible_intervals95_exclude_zero=len(definite),
        eligible_definite_sign_opposite_clean=sum((p['hi95']<0 and p['oracle_gap']>0) or
                                                 (p['lo95']>0 and p['oracle_gap']<0) for p in definite),
        primary_oracle_gap_coverage95=sum(p['oracle_in_95'] for p in primary),
        exact_controls=len(exact),exact_control_coverage95=sum(e['oracle_in_95'] for e in exact),
        max_frozen_score_scaled_error=max(checks),
        calculation_seconds=sum(d['elapsed_seconds'] for d in data))
    assert summary['unchanged_valid_retention']==53
    for name,rows in [('pairs.csv',pairs),('primary_dominators.csv',dom),('exact_controls.csv',exact),
                      ('equations.csv',equations),('covariance_sensitivity.csv',sensitivity)]:write_csv(P.ROOT/name,rows)
    R.dump(P.ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    highlighted=[p for p in dom if (p['system'],p['seed']) in P.FAILED]
    fig,ax=plt.subplots(figsize=(11,max(5,len(highlighted)*.33)))
    for i,p in enumerate(highlighted):
        sd=p['paired_sd'];factor=sd if sd else 1.
        color='#be3a30' if p['oracle_gap']>0 else '#237a53'
        ax.plot([p['lo95']/factor,p['hi95']/factor],[i,i],color=color,linewidth=1.5)
        ax.scatter(p['gap']/factor,i,color=color,s=22)
        ax.scatter(p['oracle_gap']/factor,i,color='black',marker='x',s=26)
    ax.set_yticks(range(len(highlighted)),[f"{p['system']} s{p['seed']} {p['candidate']}" for p in highlighted])
    ax.axvline(0,color='gray',linewidth=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Competitor minus reference discrepancy / paired standard deviation')
    ax.set_title('Previously failed trials: conditional 95% model intervals\nDots: noisy gaps; x: fixed-fit clean gaps; red: worse on clean data')
    fig.tight_layout();fig.savefig(P.ROOT/'paired_gaps.png',dpi=180);fig.savefig(P.ROOT/'paired_gaps.svg');plt.close(fig)
    R.dump(P.ROOT/'audit.json',dict(coverage_verified=True,source_hashes_verified=True,baseline_hashes_verified=True,
        frontiers_unchanged=True,finite_samples=True,max_frozen_score_scaled_error=max(checks),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2))
    print('Failed cases: candidate, gap/sd, conditional interval, clean gap')
    for p in highlighted:print(p['system'],p['seed'],p['candidate'],p['standardized_margin'],[p['lo95'],p['hi95']],p['oracle_gap'])


if __name__=='__main__':main()
