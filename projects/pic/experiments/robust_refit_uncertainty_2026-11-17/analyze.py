from common import *
from collections import Counter,defaultdict
import csv
def write(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==69
    records=[read(p) for p in files];rows=[];params=[];geometry=[];warnings=[];initial_comparisons=[]
    for d in records:
        meta={k:d[k] for k in ('system','seed','profile')};p=R.datasets.load(d['system']);key=f"{d['system']}_{d['profile']}_seed{d['seed']}"
        prior=read(ROOT/'frozen'/(key+'.json'));frozen=read(ROOT/'frozen_fast'/(key+'.json'))
        # Exact same estimates, draws and parameter-refit records in slow/fast runs.
        assert prior['model']==frozen['model'] and prior['refits']==frozen['refits'] and prior['field_hashes']==frozen['field_hashes']
        diff=max(abs(prior['metrics'][mode][bank]['cutoff']/d['metrics'][mode][bank]['cutoff']-1) for mode in ('fixed','refit') for bank in ('local','broad'))
        initial_comparisons.append(dict(**meta,max_relative_cutoff_change=diff))
        for r in d['statistics']:rows.append(dict(**meta,mode=r['mode'],bank=r['bank'],coverage=r['operator_coverage_count']/256,anchor_references=len(r['references']),anchor_accepted=sum(x['anchor_accepted'] for x in r['references']),pointwise_coverage=float(np.mean([x['pointwise_coverage_count']/256 for x in r['references']]))))
        for mode,j in d['joint'].items():
            rows.append(dict(**meta,mode=mode,bank='both',coverage=j['operator_coverage_count']/256,anchor_references=None,anchor_accepted=None,pointwise_coverage=None))
            geometry.append(dict(**meta,mode=mode,geometry=dict(Counter(c['geometry'] for c in j['certificates'] if c['candidate'] in {x['id'] for x in frozen['candidates'] if x['label']=='reference'}))))
        for v,m in d['model'].items():params.append(dict(**meta,stream='anchor',variable=v,sigma_ratio=m[0]/(.05*np.std(p.data[v])),rho_min=min(m[1]),rho_max=max(m[1]),status=d['fit_info'][v]['status']))
        for stream,rr in d['refits'].items():
            for v,base in d['model'].items():
                sig=np.array([r['model'][v][0] for r in rr]);rho=np.array([r['model'][v][1] for r in rr]);params.append(dict(**meta,stream=stream,variable=v,sigma_ratio=float(np.median(sig)/(.05*np.std(p.data[v]))),sigma_min_relative_anchor=float(sig.min()/base[0]),sigma_max_relative_anchor=float(sig.max()/base[0]),rho_min=float(rho.min()),rho_max=float(rho.max()),rho_gt_095=int(np.sum(np.any(np.abs(rho)>.95,axis=1))),status='draw_distribution'))
        warnings.extend(dict(**meta,**w) for w in d['warnings'])
    summary=[];systems=[]
    for group in sorted({(r['profile'],r['mode'],r['bank']) for r in rows}):
        rr=[r for r in rows if (r['profile'],r['mode'],r['bank'])==group];summary.append(dict(profile=group[0],mode=group[1],bank=group[2],anchors=len(rr),coverage=float(np.mean([r['coverage'] for r in rr])),anchor_min=min(r['coverage'] for r in rr),anchor_max=max(r['coverage'] for r in rr)))
    for group in sorted({(r['system'],r['profile'],r['mode'],r['bank']) for r in rows}):
        rr=[r for r in rows if (r['system'],r['profile'],r['mode'],r['bank'])==group];systems.append(dict(system=group[0],profile=group[1],mode=group[2],bank=group[3],anchors=len(rr),coverage=float(np.mean([r['coverage'] for r in rr]))))
    pure=read(ROOT/'noise_trials.json');pure_summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in pure['rows'] if (r['n'],r['true_rho'])==(n,rho)];pure_summary.append(dict(n=n,rho=rho,draws=len(rr),rho_gt_095=sum(abs(r['model'][1][0])>.95 for r in rr),sigma_gt_twice_true=sum(r['model'][0]>.4 for r in rr),rho_rmse=float(np.sqrt(np.mean([(r['model'][1][0]-rho)**2 for r in rr])))))
    write('coverage.csv',rows);write('system_coverage.csv',systems);write('noise_parameters.csv',params);write('pure_noise_summary.csv',pure_summary)
    if warnings:write('fit_warnings.csv',warnings)
    dump(ROOT/'summary.json',dict(status='COMPLETE',anchors=69,paired_records=138,result_statuses=dict(Counter(d['status'] for d in records)),summary=summary,systems=systems,reference_geometry=geometry,warnings=warnings,pure_noise_summary=pure_summary,max_fast_slow_cutoff_relative_change=max(r['max_relative_cutoff_change'] for r in initial_comparisons),source_sha256=hashes('analyze.py'),input_sha256={str(p):sha(p) for p in files}))
    print(json.dumps(summary,indent=2));print('WARNINGS',len(warnings));plot(systems,summary)
def plot(systems,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=SYSTEMS;lookup={(r['system'],r['profile'],r['mode'],r['bank']):r['coverage'] for r in systems}
    data=np.array([[lookup[n,'iid',m,b] for m,b in [('fixed','local'),('refit','local'),('fixed','broad'),('refit','broad')]] for n in names])
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(data,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(15),names);ax.set_xticks(range(4),['Fixed/local','Refit/local','Fixed/broad','Refit/broad'])
    for i in range(15):
        for j in range(4):ax.text(j,i,f'{100*data[i,j]:.1f}',ha='center',va='center',color='black' if data[i,j]>.6 else 'white',fontsize=9)
    ax.set_title('Nominal 95% simultaneous coverage: iid, anchors 42-44');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'iid_coverage.png',dpi=180);fig.savefig(ROOT/'iid_coverage.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,profile in zip(axes,('iid','positive','negative')):
        for mode,x in [('fixed',-.18),('refit',.18)]:
            vv=[100*next(r['coverage'] for r in summary if (r['profile'],r['mode'],r['bank'])==(profile,mode,b)) for b in ('local','broad')];bars=ax.bar(np.arange(2)+x,vv,.34,label=mode);ax.bar_label(bars,fmt='%.1f')
        ax.set_xticks([0,1],['Local','Broad']);ax.set_ylim(0,108);ax.axhline(95,color='gray',ls='--');ax.set_title(profile);ax.set_ylabel('Coverage (%)')
    axes[0].legend();fig.suptitle('Noise estimation uncertainty: fixed versus full refit procedure');fig.tight_layout();fig.savefig(ROOT/'coverage_comparison.png',dpi=180);fig.savefig(ROOT/'coverage_comparison.pdf');plt.close(fig)
if __name__=='__main__':main()
