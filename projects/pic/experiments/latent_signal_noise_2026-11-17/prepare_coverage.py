"""Create the isolated fresh-coverage harness; never modify prior experiments."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spectral_noise_2026-11-17'

def save(name,text):
    path=ROOT/name
    assert not path.exists(),path
    path.write_text(text,encoding='utf-8')

def main():
    text=(OLD/'coverage.py').read_text(encoding='utf-8')
    text=text.replace('import stable_fit as F','import correlated_fit as F')
    start=text.index('def curvature(');end=text.index('class Study:',start)
    text=text[:start]+text[end:]
    text=text.replace("'noise_fit.py','stable_fit.py'","'latent_fit.py','correlated_fit.py','correlated_curvature_protocol.txt'")
    text=text.replace("kind=='spectral'","kind=='curvature'").replace("F.estimate(y,'strips')","F.estimate(y)")
    text=text.replace("                if m is not None:info['curvature']=curvature(y,m)\n",'')
    text=text.replace('950101','960101').replace('950104','960104')
    save('coverage.py',text)
    for name in ('run_case.py','run_suite.py','audit.py','analyze.py'):
        text=(OLD/name).read_text(encoding='utf-8')
        text=text.replace('950102','960102').replace('950103','960103')
        text=text.replace('(33,34,35)','(36,37,38)').replace('==162','==138').replace('records=162','records=138')
        text=text.replace("('spectral','ar','known_rho')","('curvature','known_rho')").replace("('spectral','iid')","('curvature','iid')")
        text=text.replace("('checks.json','check_stable.json')","('checks.json','checks_correlated.json')")
        if name=='analyze.py':
            text=text.replace('import stable_fit as F','import correlated_fit as F')
            text=text[:text.index('def plot(')]+PLOT+'\n\nif __name__==\'__main__\':main()\n'
        save(name,text)
    paths=[OLD/n for n in ('coverage.py','run_case.py','run_suite.py','audit.py','analyze.py')]
    record={'prior_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'generated_sha256':{str(ROOT/p.name):hashlib.sha256((ROOT/p.name).read_bytes()).hexdigest() for p in paths}}
    save('harness_provenance.json',json.dumps(record,indent=2))

PLOT='''def plot(summary,systems):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,profile in zip(axes,('iid','positive','negative')):
        kinds=['iid','curvature'] if profile=='iid' else ['known_rho','curvature']
        labels=['Known iid family','Latent curvature'] if profile=='iid' else ['Known rho control','Latent curvature']
        for bank,offset,color in [('local',-.18,'#18709a'),('broad',.18,'#d18132')]:
            values=[100*next(r for r in summary if (r['profile'],r['model_kind'],r['bank'])==(profile,k,bank))['operator_coverage'] for k in kinds]
            bars=ax.bar(np.arange(len(kinds))+offset,values,.34,color=color,label=bank.title());ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.axhline(95,color='#444444',ls='--',lw=1);ax.set_ylim(0,107);ax.set_xticks(range(len(kinds)),labels,fontsize=9);ax.set_ylabel('Simultaneous coverage (%)')
        ax.set_title('IID: default 15 systems' if profile=='iid' else 'Core 4: rho '+('+0.4' if profile=='positive' else '-0.4'))
    axes[0].legend(frameon=False,loc='lower left');fig.suptitle('Latent signal/noise: fresh anchors 36-38; completed fits only',fontsize=12);fig.tight_layout()
    fig.savefig(ROOT/'coverage_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'coverage_comparison.pdf',bbox_inches='tight');plt.close(fig)
    names=sorted({r['system'] for r in systems if r['profile']=='iid'})
    lookup={(r['system'],r['profile'],r['model_kind'],r['bank']):r['operator_coverage'] for r in systems}
    matrix=np.array([[lookup.get((n,'iid',k,b),np.nan) for k,b in [('iid','local'),('curvature','local'),('iid','broad'),('curvature','broad')]] for n in names])
    fig,ax=plt.subplots(figsize=(7,6.5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(len(names)),names);ax.set_xticks(range(4),['IID/local','Latent/local','IID/broad','Latent/broad'])
    for i in range(len(names)):
        for j in range(4):ax.text(j,i,f'{100*matrix[i,j]:.1f}' if np.isfinite(matrix[i,j]) else 'incomplete',ha='center',va='center',color='black' if matrix[i,j]>.6 else 'white',fontsize=9)
    ax.set_title('IID simultaneous coverage by system (%)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'iid_systems.png',dpi=180,bbox_inches='tight');plt.close(fig)
'''

if __name__=='__main__':main()
