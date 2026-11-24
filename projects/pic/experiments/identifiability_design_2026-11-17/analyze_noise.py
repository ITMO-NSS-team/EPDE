from research_context import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    keys=read(ROOT/'noise_preparation.json')['objects'];summary=[];fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for key,ax in zip(keys,axs.flat):
        inp=read(ROOT/'noise_inputs'/(key+'.json'));rows=sorted([read(p) for p in (ROOT/'noise_profiles').glob(key+'_rho*.json')],key=lambda r:r['rho'])
        minimum=min([r['objective'] for r in rows]+[inp['fit_info']['objective']]);n=rows[0]['contrast_count']
        for r in rows:r['minus2loglik_drop']=n*(r['objective']-minimum)
        near=[r for r in rows if r['rho']>=.99];summary.append(dict(key=key,archived_model=inp['model'],grid_best_rho=min(rows,key=lambda r:r['objective'])['rho'],
            grid_improvement_over_unrestricted=n*(inp['fit_info']['objective']-minimum),
            support={str(t):dict(rhos=[r['rho'] for r in rows if r['minus2loglik_drop']<=t],sigmas=[r['sigma'] for r in rows if r['minus2loglik_drop']<=t]) for t in (2,6)},
            near_unit=[{k:r[k] for k in ('rho','sigma','innovation_sigma','minus2loglik_drop')} for r in near],
            maximum_projected_gradient=max(r['projected_gradient'] for r in rows),statuses=[r['status'] for r in rows]))
        x=[np.arctanh(r['rho']) for r in rows];ax.plot(x,[r['minus2loglik_drop'] for r in rows],'o-',ms=3,label='profile likelihood loss')
        ax.axhline(2,color='grey',ls=':',lw=1);ax.axhline(6,color='grey',ls='--',lw=1)
        ax.set(title=key.replace('_',' '),xlabel='atanh(rho)',ylabel='-2 log likelihood difference',ylim=(-.2,20));ax.grid(alpha=.2)
    fig.suptitle('Second-contrast noise fits: nuisance parameters reoptimized\nLines at 2 and 6 are descriptive, not confidence limits')
    fig.savefig(ROOT/'noise_profiles.png',dpi=180);fig.savefig(ROOT/'noise_profiles.pdf');plt.close(fig)
    dump(ROOT/'noise_summary.json',dict(status='COMPLETE',summary=summary,input_sha256={str(p):sha(p) for p in (ROOT/'noise_profiles').glob('*.json')},source_sha256=sources('analyze_noise.py')))
    for r in summary:print(r['key'],'gridbest',r['grid_best_rho'],'improvement',r['grid_improvement_over_unrestricted'],'support6',r['support']['6'],'nearunit',r['near_unit'][-1],flush=True)

if __name__=='__main__':main()
