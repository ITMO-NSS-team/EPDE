from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parent
import stable_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S
from collections import Counter,defaultdict

def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==162
    rows=[];noise=[];references=[];incomplete=[]
    for path in files:
        d=json.loads(path.read_text());meta={k:d[k] for k in ('system','seed','profile','model_kind')};p=S.R.datasets.load(d['system'])
        for v,m in d['model'].items():
            if m is None:continue
            true=.05*float(np.std(p.data[v]));info=d['fit_info'].get(v,{});curv=info.get('curvature',{})
            for axis,rho in enumerate(m[1]):noise.append(dict(**meta,variable=v,axis=axis,sigma_ratio=m[0]/true,rho=rho,true_rho=S.PROFILES[d['profile']][1],status=info.get('status','control'),
                approximate_rho_sd=curv.get('approximate_rho_sd',[None]*len(m[1]))[axis] if curv.get('approximate_rho_sd') is not None else None))
        if d['status']!='COMPLETE':incomplete.append(dict(**meta,status=d['status']));continue
        for bank in ('local','broad'):
            rr=[r for r in d['statistics'] if r['bank']==bank and r['label']=='reference']
            if d['metrics'][bank]['status']!='ok':incomplete.append(dict(**meta,bank=bank,status=d['metrics'][bank]['status']));continue
            rows.append(dict(**meta,bank=bank,operator_coverage=rr[0]['operator_coverage_count']/256,
                directional_coverage=np.mean([r['pointwise_noise_coverage_count']/256 for r in rr]),
                directional_zero_acceptance=np.mean([r['pointwise_zero_acceptance_count']/256 for r in rr]),
                uniform_coverage=np.mean([r['uniform_noise_coverage_count']/256 for r in rr]),
                uniform_zero_acceptance=np.mean([r['uniform_zero_acceptance_count']/256 for r in rr])))
            for r in rr:
                region=next(c for c in d['regions'][bank] if c['candidate']==r['candidate'])
                references.append(dict(**meta,bank=bank,candidate=r['candidate'],geometry=region['status'],anchor_exact_accepted=r['anchor_exact_accepted']))
        if d['joint']:
            rows.append(dict(**meta,bank='both',operator_coverage=d['joint']['operator_coverage_count']/256,
                directional_coverage=None,directional_zero_acceptance=None,uniform_coverage=None,
                uniform_zero_acceptance=np.mean([r['actual_acceptance_count']/256 for r in d['joint']['reference_statistics']])))
            for r in d['joint']['reference_statistics']:
                cert=next(c for c in d['joint']['certificates'] if c['candidate']==r['candidate'])
                references.append(dict(**meta,bank='both',candidate=r['candidate'],geometry=cert['geometry'],anchor_exact_accepted=r['anchor_accepted']))
    summaries=[]
    for profile,kind in sorted({(r['profile'],r['model_kind']) for r in rows}):
        for bank in ('local','broad','both'):
            rr=[r for r in rows if (r['profile'],r['model_kind'],r['bank'])==(profile,kind,bank)]
            refs=[r for r in references if (r['profile'],r['model_kind'],r['bank'])==(profile,kind,bank)]
            out=dict(profile=profile,model_kind=kind,bank=bank,completed_records=len(rr),incomplete_records=sum(r['profile']==profile and r['model_kind']==kind for r in incomplete),
                reference_cases=len(refs),geometry=dict(Counter(r['geometry'] for r in refs)),exact_anchor_accepted=sum(r['anchor_exact_accepted'] for r in refs))
            for key in ('operator_coverage','directional_coverage','directional_zero_acceptance','uniform_coverage','uniform_zero_acceptance'):
                values=[r[key] for r in rr if r[key] is not None];out[key]=float(np.mean(values)) if values else None
            summaries.append(out)
    systems=[]
    for name,profile,kind,bank in sorted({(r['system'],r['profile'],r['model_kind'],r['bank']) for r in rows}):
        rr=[r for r in rows if (r['system'],r['profile'],r['model_kind'],r['bank'])==(name,profile,kind,bank)]
        out=dict(system=name,profile=profile,model_kind=kind,bank=bank,records=len(rr))
        for key in ('operator_coverage','directional_coverage','uniform_coverage'):
            values=[r[key] for r in rr if r[key] is not None];out[key]=float(np.mean(values)) if values else None
        systems.append(out)
    write('coverage.csv',rows);write('system_coverage.csv',systems);write('noise_estimates.csv',noise);write('reference_regions.csv',references)
    out=dict(status='COMPLETE',records=162,summary=summaries,incomplete=incomplete,source_sha256={str(Path(__file__)):S.E.sha(__file__)})
    S.R.dump(ROOT/'summary.json',out)
    print(json.dumps(summaries,indent=2));print('INCOMPLETE',incomplete)
    plot(summaries,systems)

def plot(summary,systems):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1))
    for ax,profile in zip(axes,('iid','positive','negative')):
        kinds=['iid','spectral'] if profile=='iid' else ['ar','spectral','known_rho']
        labels=['Known iid\nfamily','Joint\nspectral'] if profile=='iid' else ['Existing AR','Joint\nspectral','Known rho\ncontrol']
        for bank,offset,color in [('local',-.18,'#18709a'),('broad',.18,'#d18132')]:
            values=[100*next(r for r in summary if (r['profile'],r['model_kind'],r['bank'])==(profile,k,bank))['operator_coverage'] for k in kinds]
            bars=ax.bar(np.arange(len(kinds))+offset,values,.34,color=color,label=bank.title());ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.axhline(95,color='#444444',ls='--',lw=1);ax.set_ylim(0,107);ax.set_xticks(range(len(kinds)),labels);ax.set_ylabel('Simultaneous coverage (%)')
        ax.set_title('IID: all 15 systems' if profile=='iid' else 'Core 4: rho '+('+0.4' if profile=='positive' else '−0.4'))
    axes[0].legend(frameon=False,loc='lower left');fig.suptitle('Fresh solver-free noise-model test: seeds 33–35',fontsize=13);fig.tight_layout()
    fig.savefig(ROOT/'coverage_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'coverage_comparison.pdf',bbox_inches='tight');plt.close(fig)
    names=sorted({r['system'] for r in systems if r['profile']=='iid'})
    matrix=np.array([[next(r for r in systems if (r['system'],r['profile'],r['model_kind'],r['bank'])==(n,'iid',k,b))['operator_coverage'] for k,b in [('iid','local'),('spectral','local'),('iid','broad'),('spectral','broad')]] for n in names])
    fig,ax=plt.subplots(figsize=(7,6.5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(len(names)),names);ax.set_xticks(range(4),['IID/local','Spectral/local','IID/broad','Spectral/broad'])
    for i in range(len(names)):
        for j in range(4):ax.text(j,i,f'{100*matrix[i,j]:.1f}',ha='center',va='center',color='black' if matrix[i,j]>.6 else 'white',fontsize=9)
    ax.set_title('IID simultaneous coverage by system (%)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'iid_systems.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
