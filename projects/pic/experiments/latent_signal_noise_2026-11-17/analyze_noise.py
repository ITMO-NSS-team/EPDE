from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
import hashlib

def main():
    old=json.loads((ROOT/'noise_trials.json').read_text());new=json.loads((ROOT/'noise_trials_correlated.json').read_text());rows=[]
    for a,b in zip(old['rows'],new['rows']):
        models={'White curvature':a['latent_model'],'Correlated curvature':b['model'],'Spectral strips':a['spectral_model'],'Known zero signal':[a['oracle_sigma'],[a['oracle_rho']]]}
        statuses={'White curvature':a['latent_info']['status'],'Correlated curvature':b['info']['status'],'Spectral strips':a['spectral_status'],'Known zero signal':'ok'}
        for name,m in models.items():rows.append(dict(n=a['n'],rho=a['true_rho'],draw=a['draw'],method=name,status=statuses[name],sigma_ratio=m[0]/.2 if m else None,rho_error=m[1][0]-a['true_rho'] if m else None))
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            for name in models:
                rr=[r for r in rows if r['n']==n and r['rho']==rho and r['method']==name];good=[r for r in rr if r['rho_error'] is not None]
                summary.append(dict(n=n,rho=rho,method=name,draws=len(rr),incomplete=sum(r['status']!='ok' for r in rr),available=len(good),
                    rho_rmse=float(np.sqrt(np.mean([r['rho_error']**2 for r in good]))),sigma_median=float(np.median([r['sigma_ratio'] for r in good]))))
    with (ROOT/'pure_noise_comparison.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,rho in zip(axes,(0.,.4,-.4)):
        for name,marker in [('Spectral strips','x'),('White curvature','o'),('Correlated curvature','s'),('Known zero signal','^')]:
            rr=[r for r in summary if r['rho']==rho and r['method']==name];ax.plot([r['n'] for r in rr],[r['rho_rmse'] for r in rr],marker=marker,label=name)
        ax.set_title(f'True rho = {rho:+.1f}');ax.set_xlabel('Record length');ax.grid(alpha=.2)
    axes[0].set_ylabel('Correlation RMSE');axes[0].legend(fontsize=8);fig.suptitle('Pure noise: 64 paired draws per length/profile; available fits')
    fig.tight_layout();fig.savefig(ROOT/'pure_noise_comparison.png',dpi=180);fig.savefig(ROOT/'pure_noise_comparison.pdf');plt.close(fig)
    out=dict(status='COMPLETE',summary=summary,source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        input_sha256={str(ROOT/n):hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('noise_trials.json','noise_trials_correlated.json')})
    (ROOT/'noise_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Incomplete:',{name:sum(r['status']!='ok' for r in rows if r['method']==name) for name in models})

if __name__=='__main__':main()
