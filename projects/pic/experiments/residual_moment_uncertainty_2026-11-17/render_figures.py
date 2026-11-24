"""Readable final scientific figures from audited numerical tables."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    original=list(csv.DictReader((ROOT/'summary.csv').open()));rows=list(csv.DictReader((ROOT/'iid_summary.csv').open()))
    names=sorted({r['system'] for r in rows if r['system']!='ALL'});fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    for ax,bank in zip(axs,['local','broad']):
        before=[next(r for r in original if r['system']==name and r['subset']=='reference' and r['kind']=='exact' and r['arm']==bank+'_corrected') for name in names]
        ax.plot(names,[float(r['mean_noise_coverage']) for r in before],'o-',label='AR, fixed estimates (seeds24-26)',color='#a94b34')
        for cohort,label,color in [('paired24_26','iid, re-estimated amplitude (seeds24-26)','#087f8c'),('confirmation27_29','iid, re-estimated amplitude (fresh27-29)','#5153a6')]:
            rr=[next(r for r in rows if r['cohort']==cohort and r['system']==name and r['subset']=='reference' and r['arm']==bank+'_studentized') for name in names]
            ax.plot(names,[float(r['mean_noise_coverage']) for r in rr],'o-',label=label,color=color)
        ax.axhline(.95,color='grey',ls='--');ax.set_ylim(0,1.02);ax.tick_params(axis='x',rotation=65);ax.set_title(bank.title()+' native weak moments');ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('95% noise-region coverage');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(ROOT/('noise_model_comparison.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    inv=json.loads((ROOT/'wave_inversion.json').read_text());fig,axs=plt.subplots(1,2,figsize=(10,4.5),sharex=True)
    for ax,bank in zip(axs,['local','broad']):
        rr=[r for r in inv['rows'] if r['bank']==bank]
        for r in rr:
            for lo,hi in r['intervals']:
                assert lo is not None and hi is not None
                ax.plot([lo,hi],[r['seed'],r['seed']],lw=3,color='#087f8c')
            ax.plot(r['ols'],r['seed'],'x',color='#b74929')
        ax.axvline(.04,color='grey',ls='--');ax.set_yticks([r['seed'] for r in rr]);ax.set_title(bank.title()+' weak moments');ax.set_xlabel('Coefficient of u_xx');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Anchor seed');fig.suptitle('Wave: inverted95% residual regions; x = ordinary fit (common horizontal scale)')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(ROOT/('wave_intervals.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='PASS',source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256={str(ROOT/p):sha(ROOT/p) for p in ('summary.csv','iid_summary.csv','wave_inversion.json')})
    (ROOT/'figure_sources.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
