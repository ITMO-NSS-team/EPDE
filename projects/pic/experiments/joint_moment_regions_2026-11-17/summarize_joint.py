import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
from collections import Counter,defaultdict
import json,csv

def write(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==117
    group=defaultdict(list);noise=[];system_coverage=[];ks_witnesses=[];compound=[];rejected=[]
    for path in files:
        d=json.loads(path.read_text());f=json.loads((ROOT/'frozen'/path.name).read_text());j=json.loads((ROOT/'intersection'/path.name).read_text())
        meta={k:d[k] for k in ('system','seed','profile','model_kind')};group[(d['profile'],d['model_kind'])].append((d,f,j))
        p=R.datasets.load(d['system']);rho=S.PROFILES[d['profile']][1]
        for v,model in f['model'].items():
            truth=.05*float(np.std(p.data[v]))
            for axis,r in enumerate(model[1]):noise.append(dict(**meta,variable=v,axis=axis,sigma_ratio=model[0]/truth,rho=r,true_rho=rho,order=model[2][axis]))
        for bank in ('local','broad'):
            rows=[r for r in d['statistics'] if r['bank']==bank and r['label']=='reference']
            system_coverage.append(dict(**meta,bank=bank,operator_coverage=rows[0]['operator_coverage_count']/256,
                reference_pointwise_coverage=np.mean([r['pointwise_noise_coverage_count']/256 for r in rows]),
                reference_uniform_coverage=np.mean([r['uniform_noise_coverage_count']/256 for r in rows])))
        if d['profile']!='iid':continue
        for r in j['statistics']:
            if r['label']=='reference' and not r['joint_exact_anchor_accepted']:rejected.append(dict(**meta,**r))
        if d['system']=='pde_compound':
            c=next(r for r in j['certificates'] if r['label']=='reference');compound.append(dict(seed=d['seed'],lambda_local=c['lambda_local'],lower=c['outer_region']['lower'],upper=c['outer_region']['upper'],feasible='feasible_witness' in c))
        if d['system']!='ks':continue
        with np.load(path.with_suffix('.npz')) as a:
            c=next(c for c in f['candidates'] if c['label']=='reference');cut=j['cutoff'];aa=[a[b+'_observed_gram']-cut*a[b+'_sigma'] for b in ('local','broad')]
            for value in (0.,1.):
                xi=c['xi'];h=aa[1][np.ix_(xi,xi)];g=aa[1][xi,c['yi']];beta=np.zeros(3);beta[2]=value
                beta[:2]=np.linalg.solve(h[:2,:2],g[:2]-h[:2,2]*value);v=S.E.vector(c,beta,len(aa[0]));margins=[float(v@q@v) for q in aa]
                ks_witnesses.append(dict(system='ks',seed=d['seed'],coefficient=2,fixed_value=value,beta=beta.tolist(),margins=margins,accepted_both=max(margins)<=0))
    summaries=[]
    for (profile,kind),items in sorted(group.items()):
        refs=[c for d,f,j in items for c in j['certificates'] if c['label']=='reference']
        stats=[s for d,f,j in items for s in j['statistics'] if s['label']=='reference']
        allcert=[c for d,f,j in items for c in j['certificates']]
        summaries.append(dict(profile=profile,model_kind=kind,records=len(items),reference_cases=len(refs),
            operator_coverage=np.mean([j['operator_coverage_count']/256 for d,f,j in items]),
            exact_reference_anchor_accepted=sum(s['joint_exact_anchor_accepted'] for s in stats),
            actual_exact_reference_acceptance=sum(s['actual_exact_acceptance_count'] for s in stats)/(256*len(stats)),
            reference_geometry=dict(Counter(c['geometry'] for c in refs)),
            reference_feasible_witnesses=sum('feasible_witness' in c or c['geometry']=='point' for c in refs),
            all_candidate_geometry=dict(Counter(c['geometry'] for c in allcert)),
            all_candidate_feasible_witnesses=sum('feasible_witness' in c or c['geometry']=='point' for c in allcert),
            all_candidate_unresolved_nonemptiness=sum(c['geometry'] not in ('empty_certified','point') and 'feasible_witness' not in c for c in allcert)))
    # Candidate nonemptiness has NOT been solved by the outer-ellipsoid certificate.
    R.dump(ROOT/'intersection_summary.json',dict(status='COMPLETE',summary=summaries,compound=compound,rejected_iid_reference_anchors=rejected,
        source_sha256={str(Path(__file__)):S.E.sha(__file__)}))
    R.dump(ROOT/'ks_sign_witnesses.json',dict(method='Fix third coefficient at zero or +1; minimize broad-bank quadratic in other two coefficients, then directly test both original constraints.',rows=ks_witnesses,
        source_sha256={str(Path(__file__)):S.E.sha(__file__)},input_sha256={str(ROOT/'intersection'/f'ks_iid_iid_seed{s}.json'):S.E.sha(ROOT/'intersection'/f'ks_iid_iid_seed{s}.json') for s in (30,31,32)}))
    write('noise_diagnostics.csv',noise);write('coverage_by_record.csv',system_coverage)
    print(json.dumps(summaries,indent=2));print('COMPOUND',compound,'REJECTED',rejected)
    for key in sorted({(r['profile'],r['model_kind']) for r in noise}):
        rr=[r for r in noise if (r['profile'],r['model_kind'])==key]
        print('NOISE',key,'sigma_ratio min/median/max',np.quantile([r['sigma_ratio'] for r in rr],[0,.5,1]),'rho min/median/max',np.quantile([r['rho'] for r in rr],[0,.5,1]))
    plot(system_coverage,summaries)

def plot(rows,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),gridspec_kw={'width_ratios':[1,1,1.2]})
    for ax,profile in zip(axes[:2],('positive','negative')):
        for bank,offset,color in [('local',-.18,'#18709a'),('broad',.18,'#d18132')]:
            vals=[np.mean([r['operator_coverage'] for r in rows if r['profile']==profile and r['model_kind']==kind and r['bank']==bank])*100 for kind in ('iid','ar','known_rho')]
            bars=ax.bar(np.arange(3)+offset,vals,.34,color=color,label=bank.title())
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.axhline(95,color='#444444',ls='--',lw=1);ax.set_ylim(0,106);ax.set_xticks(range(3),['Assume iid','Estimate AR','Known rho\ncontrol']);ax.set_title('Noise correlation '+('+0.4' if profile=='positive' else '−0.4'));ax.set_ylabel('Simultaneous coverage (%)')
    axes[0].legend(loc='center left',frameon=False)
    ax=axes[2]
    with (ROOT/'iid_projections.csv').open() as f:pr=list(csv.DictReader(f))
    for i,seed in enumerate((30,31,32)):
        r=next(r for r in pr if r['system']=='ks' and int(r['seed'])==seed and r['bank']=='broad' and int(r['coefficient'])==2)
        ax.plot([float(r['lower']),float(r['upper'])],[i,i],color='#18709a',lw=3)
        ax.scatter([-1],[i],color='#242424',marker='|',s=180,zorder=3);ax.scatter([1],[i],facecolors='white',edgecolors='#c03c37',s=60,zorder=3)
    ax.axvline(0,color='#aaaaaa',lw=1);ax.set_yticks(range(3),['Seed 30','Seed 31','Seed 32']);ax.set_ylim(-.7,2.7);ax.set_xlabel('Coefficient of fourth spatial derivative');ax.set_title('KS: sign remains unresolved')
    ax.legend(handles=[Line2D([0],[0],color='#18709a',lw=3,label='Combined outer bounds'),Line2D([0],[0],color='#242424',marker='|',ls='',markersize=12,label='True coefficient −1'),Line2D([0],[0],color='#c03c37',marker='o',mfc='white',ls='',label='+1 satisfies both banks')],loc='upper left',bbox_to_anchor=(0,-.27),frameon=False)
    fig.suptitle('Solver-free joint weak-moment regions: noise calibration and coefficient precision',fontsize=13,y=1.01)
    fig.tight_layout();fig.savefig(ROOT/'joint_results.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'joint_results.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
