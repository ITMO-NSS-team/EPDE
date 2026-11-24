from bridge import *
import csv

def main():
    rows=[];files=sorted((ROOT/'factorial_frozen').glob('*.json'));assert len(files)==3
    for path in files:
        f=read(path)
        for r in f['rows']:
            for v,(m,i) in r['fits'].items():rows.append(dict(seed=f['seed'],draw=r['draw'],pilot=r['pilot'],method=r['method'],variable=v,sigma_ratio=m[0]/f['model'][v][0],max_rho_error=float(np.max(np.abs(np.array(m[1])-f['model'][v][1]))),near_unit=max(abs(np.array(m[1])))>.95,status=i['status']))
    summary=[]
    for pilot in ('ordinary','correlated'):
        for method in ('time','local6'):
            rr=[r for r in rows if (r['pilot'],r['method'])==(pilot,method)];ss=np.array([r['sigma_ratio'] for r in rr]);rho=np.array([r['max_rho_error'] for r in rr]);summary.append(dict(pilot=pilot,method=method,fields=len(rr),sigma_ratio_quantiles=np.quantile(ss,[0,.5,1]),max_rho_error_quantiles=np.quantile(rho,[0,.5,1]),near_unit=sum(r['near_unit'] for r in rr),warnings=sum(r['status']!='ok' for r in rr)))
    with (ROOT/'factorial_parameters.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    dump(ROOT/'factorial_analysis.json',dict(status='COMPLETE',summary=summary,source_sha256=sources('analyze_factorial.py'),input_sha256={str(p):sha(p) for p in files}));print(json.dumps(summary,indent=2,default=lambda x:x.tolist()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));colors=['#0072B2','#D55E00','#009E73']
    for ax,method,title in zip(axes,('time','local6'),('Temporal estimator','Local polynomial estimator')):
        for j,pilot in enumerate(('ordinary','correlated')):
            for k,seed in enumerate((45,46,47)):
                rr=[r for r in rows if (r['method'],r['pilot'],r['seed'])==(method,pilot,seed)];x=j+(k-1)*.12+np.linspace(-.025,.025,len(rr));ax.scatter(x,[r['sigma_ratio'] for r in rr],s=15,color=colors[k],alpha=.7,label=f'Anchor {seed}' if j==0 else None)
        ax.axhline(1,color='black',ls='--',lw=1);ax.set_xticks([0,1],['Ordinary GCV','Correlation-aware GCV']);ax.set_title(title);ax.set_ylabel('Fitted / generating noise amplitude')
    axes[0].set_yscale('log');axes[0].set_yticks([1,2,5],['1','2','5']);axes[0].legend(fontsize=8);fig.suptitle('KS positive correlation: paired reconstruction diagnostic');fig.text(.5,.015,'Same noise increments in both pilot arms;16 draws per anchor. This is a parameter diagnostic, not coverage.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(ROOT/'ks_reconstruction_diagnostic.png',dpi=180);fig.savefig(ROOT/'ks_reconstruction_diagnostic.pdf');plt.close(fig)
if __name__=='__main__':main()
