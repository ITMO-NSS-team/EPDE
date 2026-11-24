import raw_operator as O
R,A,np=O.R,O.A,O.np
import json,csv,hashlib


def main():
    paths=[O.ROOT/'results'/f'{n}_seed{s}.json' for n in ['wave','ks','kdv_cossin'] for s in range(3)]
    cases=[];options=[];allrows=[];summary={};controls=[];offsets=[];source={}
    sourcepaths=[O.ROOT/'run_validation.py',O.Path(O.__file__),O.Path(O.Q.__file__)]
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
    for path in paths:
        d=json.loads(path.read_text());assert d['source_sha256']==expected
        files=dict(joint=O.J.ROOT/'results'/path.name,local=O.J.L.ROOT/'results'/path.name,plan=O.F.ROOT/'fit_results'/path.name)
        assert d['input_sha256']=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
        source[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();cs=d['candidates']
        for kind,arms in d['aggregates'].items():
            for metric,entry in arms.items():
                retained=bool(A.frontier(entry['scores'],d['instability'],cs,0));summary.setdefault(kind,{}).setdefault(metric,0);summary[kind][metric]+=retained
                cases.append(dict(system=d['system'],seed=d['seed'],fit=kind,metric=metric,retained=retained,
                    unregistered_dominators=entry['dominance'][0]['unregistered']))
        with np.load(path.with_name(path.stem+'_samples.npz')) as z:noise=z['noise128']
        option_ids=[i for i,r in enumerate(d['rows']) if r['kind']=='option']
        rawbest=min(option_ids,key=lambda i:d['rows'][i]['raw_cross']);cleanbest=min(option_ids,key=lambda i:d['rows'][i]['clean_full'])
        smoothbest=min(option_ids,key=lambda i:d['rows'][i]['smooth_cross'])
        baseline=next(i for i,r in enumerate(d['rows']) if r['id']=='baseline/e0:reference')
        true=next(i for i,r in enumerate(d['rows']) if r['kind']=='exact')
        e=d['rows'][true];ref=d['rows'][baseline]
        controls.append(dict(system=d['system'],seed=d['seed'],raw_best=d['rows'][rawbest]['option'],
            smooth_best=d['rows'][smoothbest]['option'],clean_best=d['rows'][cleanbest]['option'],
            raw_ranking_matches_clean=rawbest==cleanbest,
            best_clean_vs_baseline_raw_gap=d['rows'][cleanbest]['raw_cross']-ref['raw_cross'],
            best_clean_vs_baseline_paired_sd=float(np.std(noise[:,cleanbest]-noise[:,baseline],ddof=1)),
            exact_raw_score=e['raw_cross'],exact_clean_full=e['clean_full'],exact_clean_subsample=e['clean_subsample_cross'],
            exact_raw_sd=e['sd128'],exact_raw_sd_half=e['sd64'],
            baseline_scaled_error=d['baseline_scaled_error'],
            rows_retained=[f['retained_rows'] for f in d['folds']],rows_original=[f['original_rows'] for f in d['folds']]))
        for r in d['rows']:
            row=dict(system=d['system'],seed=d['seed'],**r);allrows.append(row)
            if r['kind']=='option':options.append(row)
            if r['kind']=='baseline' and d['labels'].index([]) in r['xi']:
                offsets.append(dict(system=d['system'],seed=d['seed'],candidate=r['candidate'],
                    raw_effect=r['raw_cross']-r['raw_without_offset'],smooth_effect=r['smooth_cross']-r['smooth_without_offset'],
                    clean_effect=r['clean_full']-r['clean_without_offset'],mean_offset=r['offset']))
    for name,rows in [('raw_case_retention.csv',cases),('raw_controls.csv',controls),('raw_offset_effects.csv',offsets)]:
        with (O.ROOT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    R.dump(O.ROOT/'raw_options.json',options);R.dump(O.ROOT/'raw_summary.json',summary)
    audit=dict(completed_runs=9,systems=['wave','ks','kdv_cossin'],seeds=[0,1,2],equation_seed_cases=9,
        incomplete_system='ac: at least one fold in each seed has no common interior weak rows',
        source_and_input_hashes_verified=True,max_baseline_scaled_error=max(r['baseline_scaled_error'] for r in controls),
        raw_ranking_matches_clean=sum(r['raw_ranking_matches_clean'] for r in controls),
        result_sha256=source,controls=controls)
    R.dump(O.ROOT/'raw_audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,name in zip(axs,['wave','ks','kdv_cossin']):
        rr=[r for r in controls if r['system']==name];x=np.arange(3)
        ax.errorbar(x,[r['exact_raw_score'] for r in rr],yerr=[r['exact_raw_sd'] for r in rr],fmt='o',capsize=4,label='Raw score +/- conditional SD')
        ax.scatter(x,[r['exact_clean_full'] for r in rr],marker='x',color='black',label='Clean control')
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x);ax.set_xlabel('Noise seed');ax.set_title(name);ax.grid(alpha=.2)
    axs[0].set_ylabel('Score of exact coefficients');axs[0].legend(fontsize=7)
    fig.suptitle('Raw weak validation: exact coefficients can have very noisy scores')
    fig.tight_layout();fig.savefig(O.ROOT/'raw_exact_uncertainty.png',dpi=180);fig.savefig(O.ROOT/'raw_exact_uncertainty.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))
    for row in controls:print(row)


if __name__=='__main__':main()
