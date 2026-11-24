from pathlib import Path
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,profile in zip(axes,('iid','positive','negative')):
        p=ROOT/'results'/f'ks_{profile}_curvature_seed36.json';d=json.loads(p.read_text());cut=d['metrics']['local']['cutoff']
        with np.load(p.with_suffix('.npz')) as z:check=z['local_check_operator_scores'];actual=z['local_actual_operator_scores']
        for data,label,color in [(check,'Fitted calibration','#18709a'),(actual,'Actual noise','#d18132')]:
            xx=np.sort(data);ax.step(xx,np.arange(1,len(xx)+1)/len(xx),where='post',label=label,color=color)
        ax.axvline(cut,color='black',ls='--',lw=1,label='Nominal 95% cutoff');ax.axhline(.95,color='#aaaaaa',ls=':',lw=.8)
        ax.set_title(profile.title());ax.set_xlabel('Local operator score');ax.set_ylim(0,1.03);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Empirical cumulative probability');axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('KS: modest variance bias, complete loss of local coverage (seed 36)')
    fig.tight_layout();fig.savefig(ROOT/'ks_calibration_shift.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'ks_calibration_shift.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
