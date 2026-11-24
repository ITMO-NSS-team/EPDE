import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    paths=[p for p in (ROOT/'iid').glob('*.json') if not p.name.endswith('_frozen.json')];assert len(paths)==90
    rows=[];amplitudes=[]
    for p in sorted(paths):
        d=json.loads(p.read_text());cohort='paired24_26' if d['seed']<=26 else 'confirmation27_29'
        rows.extend(dict(cohort=cohort,system=d['system'],seed=d['seed'],**r) for r in d['rows'])
        amplitudes.append(dict(cohort=cohort,system=d['system'],seed=d['seed'],anchor=d['amplitude'],actual_mean=d['actual_amplitude_mean'],actual_sd=d['actual_amplitude_sd']))
    write('iid_per_query.csv',rows);write('iid_amplitudes.csv',amplitudes);ss=[]
    for cohort in ('paired24_26','confirmation27_29'):
        for system in ['ALL',*sorted({r['system'] for r in rows})]:
            for subset in ('reference','all_registered'):
                for arm in ('local_fixed','local_studentized','broad_fixed','broad_studentized'):
                    rr=[r for r in rows if r['cohort']==cohort and (system=='ALL' or r['system']==system) and (subset=='all_registered' or r['label']=='reference') and r['arm']==arm]
                    row=dict(cohort=cohort,system=system,subset=subset,arm=arm,cases=len(rr),incomplete=sum(r['status']!='ok' for r in rr))
                    if row['incomplete']==0:
                        row.update(mean_noise_coverage=float(np.mean([r['noise_coverage_count']/256 for r in rr])),
                            mean_zero_acceptance=float(np.mean([r['zero_acceptance_count']/256 for r in rr])),
                            anchor_accept_count=sum(r['anchor_acceptance'] for r in rr),
                            mean_coordinate_coverage=float(np.mean([r['coordinate_coverage'] for r in rr])),
                            median_sd_ratio=float(np.median([r['median_sd_ratio'] for r in rr if r['median_sd_ratio'] is not None])),
                            median_mean_shift_score=float(np.median([r['mean_shift_score'] for r in rr])))
                    ss.append(row)
    write('iid_summary.csv',ss)
    pairs=[]
    for p in sorted((ROOT/'paired_contrast').glob('*.json')):pairs.extend(json.loads(p.read_text())['rows'])
    write('paired_contrast_rows.csv',pairs);ps=[]
    for name in ('ac','kdv_cossin','ks','wave'):
        rr=[r for r in pairs if r['system']==name and r['label']=='reference' and r['arm']=='corrected'];assert len(rr)==3
        ps.append(dict(system=name,anchors=3,direct_mean_coverage=float(np.mean([r['direct_noise_coverage_count']/256 for r in rr])),
            contrast_mean_coverage=float(np.mean([r['noise_coverage_count']/256 for r in rr])),nonfinite=sum(r['nonfinite'] for r in rr),
            actual_deficient_count=sum(r['actual_deficient_count'] for r in rr)))
    write('paired_contrast_summary.csv',ps)
    out=dict(status='COMPLETE',iid_records=90,paired_contrast_records=12,iid_nonfinite=sum(r['status']!='ok' for r in rows),
        summary=[r for r in ss if r['system']=='ALL'],paired_contrast=ps)
    R.dump(ROOT/'followup_summary.json',out)
    original=list(csv.DictReader((ROOT/'summary.csv').open()));names=sorted({r['system'] for r in rows});fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    for ax,bank in zip(axs,['local','broad']):
        before=[next(r for r in original if r['system']==name and r['subset']=='reference' and r['kind']=='exact' and r['arm']==bank+'_corrected') for name in names]
        ax.plot(names,[float(r['mean_noise_coverage']) for r in before],'o-',label='AR, fixed noise estimates (seeds24-26)',color='#a94b34')
        for cohort,label,color in [('paired24_26','iid, re-estimated amplitude (seeds24-26)','#087f8c'),('confirmation27_29','iid, re-estimated amplitude (fresh27-29)','#5153a6')]:
            rr=[next(r for r in ss if r['cohort']==cohort and r['system']==name and r['subset']=='reference' and r['arm']==bank+'_studentized') for name in names]
            ax.plot(names,[r.get('mean_noise_coverage',np.nan) for r in rr],'o-',label=label,color=color)
        ax.axhline(.95,color='grey',ls='--');ax.set_ylim(0,1.02);ax.tick_params(axis='x',rotation=65);ax.set_title(bank.title()+' native weak moments');ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Coverage of nominal95% noise region at exact coefficients');fig.tight_layout();fig.savefig(ROOT/'noise_model_comparison.png',dpi=180);fig.savefig(ROOT/'noise_model_comparison.svg');plt.close(fig)
    print(json.dumps([r for r in out['summary'] if r['subset']=='reference'],indent=2));print(json.dumps(ps,indent=2))

if __name__=='__main__':main()
