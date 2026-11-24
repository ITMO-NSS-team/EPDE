import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==45
    rows=[dict(system=d['system'],seed=d['seed'],**r) for p in files for d in [json.loads(p.read_text())] for r in d['rows']]
    write('per_query.csv',rows);summary=[]
    for system in ['ALL',*sorted({r['system'] for r in rows})]:
        for subset in ('reference','all_registered'):
            for kind in ('exact','raw_ols','corrected_ols'):
                for arm in ('local_raw','local_corrected','broad_raw','broad_corrected'):
                    rr=[r for r in rows if (system=='ALL' or r['system']==system) and (subset=='all_registered' or r['label']=='reference') and r['arm']==arm and r['kind']==kind]
                    good=[r for r in rr if r['status']=='ok']
                    row=dict(system=system,subset=subset,kind=kind,arm=arm,cases=len(rr),complete_cases=len(good),incomplete=len(rr)-len(good))
                    if len(good)==len(rr):
                        row.update(mean_noise_coverage=float(np.mean([r['noise_coverage_count']/256 for r in good])),
                            mean_zero_acceptance=float(np.mean([r['zero_acceptance_count']/256 for r in good])),
                            anchor_accept_count=sum(r['anchor_acceptance'] for r in good),
                            median_clean_score=float(np.median([r['clean_score'] for r in good])),
                            max_clean_score=float(np.max([r['clean_score'] for r in good])),
                            median_shift_score=float(np.median([r['mean_shift_score'] for r in good])),
                            mean_coordinate_coverage=float(np.mean([r['coordinate_coverage'] for r in good])),
                            median_sd_ratio=float(np.median([r['median_sd_ratio'] for r in good if r['median_sd_ratio'] is not None])))
                    summary.append(row)
    write('summary.csv',summary);write('incomplete.csv',[r for r in rows if r['status']!='ok'])
    out=dict(status='COMPLETE',records=45,reference_anchor_cases=60,registered_anchor_cases=93,
        calibration_fields=11520,independent_evaluation_fields=23040,
        incomplete_rows=sum(r['status']!='ok' for r in rows),summary=[r for r in summary if r['system']=='ALL'])
    R.dump(ROOT/'summary.json',out)
    names=sorted({r['system'] for r in rows});fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    for arm,color in [('local_corrected','#087f8c'),('broad_corrected','#b74929')]:
        rr=[next(r for r in summary if r['system']==name and r['subset']=='reference' and r['kind']=='exact' and r['arm']==arm) for name in names]
        for ax,key in zip(axs,['mean_noise_coverage','mean_zero_acceptance']):ax.plot(names,[r.get(key,np.nan) for r in rr],'o-',label=arm,color=color)
    for ax,title in zip(axs,['Noise coverage about the clean discrete moments','Acceptance of zero residual at exact coefficients']):
        ax.axhline(.95,color='grey',ls='--');ax.set_ylim(0,1.02);ax.tick_params(axis='x',rotation=65);ax.set_title(title,fontsize=10);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Fraction of independent measurements');fig.suptitle('Native raw weak moments: fresh anchors24/25/26, 256 actual draws each')
    fig.tight_layout();fig.savefig(ROOT/'calibration.png',dpi=180);fig.savefig(ROOT/'calibration.svg');plt.close(fig)
    print(json.dumps([r for r in out['summary'] if r['kind']=='exact'],indent=2))


if __name__=='__main__':main()
