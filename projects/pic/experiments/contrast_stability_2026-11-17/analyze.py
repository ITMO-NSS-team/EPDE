import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def arms(d,dyad,qb,qr,complexity):
    return {'dyad':[d,dyad],'complexity':[d,complexity],'Q_bias':[d,qb],'Q_raw':[d,qr],
        'Q_bias_sum':[d,qb*np.asarray(complexity)],'Q_raw_sum':[d,qr*np.asarray(complexity)],
        'D_only':[d],'Q_bias_complexity':[d,qb,complexity],'dyad_complexity':[d,dyad,complexity]}


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==45
    fronts=[];calibration=[];candidates=[];changes=[];energies=[];infinite=[]
    for path in files:
        meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));cs=meta['candidates'];name,seed=meta['system'],meta['seed']
        frozen=json.loads((ROOT/'frozen'/path.name).read_text());complexity=[len(c['xi']) for c in cs]
        assert not meta['nonfinite'],('Explicit nonfinite case requires a separate incomplete-case report',name,seed,meta['nonfinite'])
        assert all(np.isfinite(a[k]).all() for k in a.files)
        slots=sorted({c['slot'] for c in cs});groups=frozen['groups']
        armset=arms(a['observed_discrepancy'][0],a['observed_dyad'],a['observed_q_bias'],a['observed_q_raw'],complexity)
        for j,c in enumerate(cs):
            indices=np.concatenate([np.arange(g['start'],g['stop']) for g in groups if g['candidate']==j])
            qboot=a['bootstrap_noise_q'][:,j];qactual=a['benchmark_noise_q'][:,j];radius=float(np.quantile(qboot,.95))
            row=dict(system=name,seed=seed,candidate=c['id'],slot=c['slot'],registered=c['registered'],reference=c['label']=='reference',
                features=len(c['xi']),contrasts=len(indices),dyad=float(a['observed_dyad'][j]),q_bias=float(a['observed_q_bias'][j]),
                q_raw=float(a['observed_q_raw'][j]),noise_q95=radius,
                contrast_noise_coverage_count=int(np.sum(qactual<=radius)),
                observed_null_tail=(1+int(np.sum(qboot>=a['observed_q_bias'][j])))/129,
                benchmark_stationarity_accept_count=int(np.sum(a['benchmark_q_bias'][:,j]<=radius)),
                observed_deficient_slabs=int(a['observed_deficient'][j]),clean_deficient_slabs=int(a['clean_deficient'][j]),
                anchor_clean_energy=float(a['anchor_energy'][j]),benchmark_median_energy=float(np.median(a['benchmark_energy'][:,j])))
            if len(indices):
                be=a['bootstrap_contrasts'][:,indices]-a['pilot_contrasts'][indices]
                ae=a['benchmark_contrasts'][:,indices]-a['clean_contrasts'][indices]
                q=np.quantile(be,[.025,.975],axis=0)
                row['coordinate_mean_coverage']=float(np.mean((ae>=q[0])&(ae<=q[1])))
                bs=np.std(be,axis=0,ddof=1);actuals=np.std(ae,axis=0,ddof=1);valid=actuals>0
                row['coordinate_median_sd_ratio']=float(np.median(bs[valid]/actuals[valid])) if valid.any() else None
                row['zero_variance_contrasts']=int(np.sum(a['variance'][indices]==0))
            candidates.append(row)
        for slot in slots:
            ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
            baseline=C.fronts(armset['dyad'],cs,slot);base_retained=any(cs[j]['registered'] for j in baseline)
            for arm,coords in armset.items():
                ids=C.fronts(coords,cs,slot);retained=any(cs[j]['registered'] for j in ids)
                row=dict(system=name,seed=seed,slot=slot,ensemble='anchor',mode='point',arm=arm,draws=1,
                    registered_count=int(retained),reference_count=int(ref in ids),mean_size=len(ids),
                    mean_unregistered=sum(not cs[j]['registered'] for j in ids),
                    mean_features=float(np.mean([len(cs[j]['xi']) for j in ids])),candidate_ids=json.dumps([cs[j]['id'] for j in ids]))
                fronts.append(row)
                if retained!=base_retained:changes.append(dict(system=name,seed=seed,slot=slot,arm=arm,change='gain' if retained else 'loss',
                    candidate_ids=[cs[j]['id'] for j in ids],baseline_ids=[cs[j]['id'] for j in baseline]))
            for ensemble in ('bootstrap','benchmark'):
                for mode,axis in [('fixed',0),('refit',1)]:
                    allfronts={arm:[] for arm in armset}
                    for b in range(128):
                        aa=arms(a[ensemble+'_discrepancy'][b,axis],a[ensemble+'_dyad'][b],a[ensemble+'_q_bias'][b],a[ensemble+'_q_raw'][b],complexity)
                        for arm,coords in aa.items():allfronts[arm].append(C.fronts(coords,cs,slot))
                    for arm,ff in allfronts.items():
                        fronts.append(dict(system=name,seed=seed,slot=slot,ensemble=ensemble,mode=mode,arm=arm,draws=128,
                            registered_count=sum(any(cs[j]['registered'] for j in ids) for ids in ff),reference_count=sum(ref in ids for ids in ff),
                            mean_size=float(np.mean([len(ids) for ids in ff])),
                            mean_unregistered=float(np.mean([sum(not cs[j]['registered'] for j in ids) for ids in ff])),
                            mean_features=float(np.mean([np.mean([len(cs[j]['xi']) for j in ids]) for ids in ff]))))
            energies.append(dict(system=name,seed=seed,slot=slot,anchor=float(a['anchor_energy'][ref]),
                benchmark_median=float(np.median(a['benchmark_energy'][:,ref])),benchmark_max=float(np.max(a['benchmark_energy'][:,ref]))))
    write('fronts.csv',fronts);write('candidates.csv',candidates);write('reference_energy.csv',energies);R.dump(ROOT/'changes.json',changes)
    fs=[]
    for ensemble,mode in [('anchor','point'),('bootstrap','fixed'),('bootstrap','refit'),('benchmark','fixed'),('benchmark','refit')]:
        for arm in armset:
            rr=[r for r in fronts if r['ensemble']==ensemble and r['mode']==mode and r['arm']==arm];assert len(rr)==60
            fs.append(dict(ensemble=ensemble,mode=mode,arm=arm,cases=len(rr),
                mean_registered_retention=float(np.mean([r['registered_count']/r['draws'] for r in rr])),
                registered_count=sum(r['registered_count'] for r in rr),mean_size=float(np.mean([r['mean_size'] for r in rr])),
                mean_unregistered=float(np.mean([r['mean_unregistered'] for r in rr])),mean_features=float(np.mean([r['mean_features'] for r in rr]))))
    for system in ['ALL',*sorted({r['system'] for r in candidates})]:
        for subset in ('all_with_contrasts','registered','reference'):
            # Keep the filter explicit; zero-feature equations have no coefficient contrast to calibrate.
            rr=[r for r in candidates if (system=='ALL' or r['system']==system) and r['contrasts']>0 and
                (subset=='all_with_contrasts' or (subset=='registered' and r['registered']) or (subset=='reference' and r['reference']))]
            calibration.append(dict(system=system,subset=subset,candidates=len(rr),
                mean_noise_ball_coverage=float(np.mean([r['contrast_noise_coverage_count']/128 for r in rr])),
                mean_coordinate_coverage=float(np.mean([r['coordinate_mean_coverage'] for r in rr])),
                median_coordinate_sd_ratio=float(np.median([r['coordinate_median_sd_ratio'] for r in rr if r['coordinate_median_sd_ratio'] is not None])),
                mean_benchmark_stationarity_acceptance=float(np.mean([r['benchmark_stationarity_accept_count']/128 for r in rr])),
                observed_any_deficient=sum(r['observed_deficient_slabs']>0 for r in rr)))
    write('front_summary.csv',fs);write('calibration_summary.csv',calibration)
    out=dict(status='COMPLETE',records=45,equation_seed_cases=60,calibration_measurements=45*256,evaluation_measurements=45*256,
        nonfinite=0,front_summary=fs,calibration=[r for r in calibration if r['system']=='ALL'],
        primary_anchor_changes=[r for r in changes if r['arm']=='Q_bias'])
    R.dump(ROOT/'summary.json',out)
    ff=[r for r in fs if r['ensemble']=='anchor'];xx=np.arange(len(ff));fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    bars=axs[0].bar(xx,[r['registered_count'] for r in ff],color='#087f8c');axs[0].bar_label(bars,padding=3);axs[0].set_ylim(0,65)
    axs[0].set_ylabel('Registered retained / 60');axs[1].bar(xx,[r['mean_size'] for r in ff],color='#64748b');axs[1].set_ylabel('Mean front size')
    for ax in axs:ax.set_xticks(xx,[r['arm'] for r in ff],rotation=40,ha='right');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh seeds21-23: identical fits/discrepancy, alternate structural axes')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
    labels=sorted({r['system'] for r in candidates});fig,ax=plt.subplots(figsize=(11,4.5))
    for subset,color in [('reference','#087f8c'),('all_with_contrasts','#b74929')]:
        rr=[next(r for r in calibration if r['system']==name and r['subset']==subset) for name in labels]
        ax.plot(labels,[r['mean_noise_ball_coverage'] for r in rr],'o-',label=subset,color=color)
    ax.axhline(.95,color='grey',ls='--');ax.set_ylim(0,1.02);ax.tick_params(axis='x',rotation=60);ax.legend()
    ax.set_ylabel('Coverage of bootstrap95% contrast-noise radius');ax.set_title('Calibration of coefficient-change noise, distinct from stationarity')
    fig.tight_layout();fig.savefig(ROOT/'calibration.png',dpi=180);fig.savefig(ROOT/'calibration.svg');plt.close(fig)
    print(json.dumps([r for r in fs if r['ensemble'] in ('anchor','benchmark')],indent=2));print(json.dumps(out['calibration'],indent=2))


if __name__=='__main__':main()
