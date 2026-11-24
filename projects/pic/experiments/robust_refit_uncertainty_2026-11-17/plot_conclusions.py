from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    s=read(ROOT/'studentized_summary.json');p=read(ROOT/'time_pilot_summary.json');fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.4,1]})
    ax=axes[0];profiles=['iid','positive','negative'];settings=[('refit','local','Correction refit / local'),('studentized','local','Studentized / local'),('refit','broad','Correction refit / broad'),('studentized','broad','Studentized / broad')]
    for j,(m,b,label) in enumerate(settings):
        values=[100*next(r['coverage'] for r in s['summary'] if (r['profile'],r['mode'],r['bank'])==(profile,m,b)) for profile in profiles];bars=ax.bar(np.arange(3)+(j-1.5)*.2,values,.19,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=7)
    ax.set_xticks(range(3),['IID: 15 systems','Positive: 4 systems','Negative: 4 systems']);ax.set_title('Original pilot + local polynomial estimator');ax.legend(fontsize=7,loc='lower left');ax.set_ylabel('Conditional operator coverage (%)')
    ax=axes[1]
    for j,b in enumerate(('local','broad')):
        values=[100*next(r['coverage'] for r in p['summary'] if (r['mode'],r['bank'])==(mode,b)) for mode in ('fixed','refit','studentized')];bars=ax.bar(np.arange(3)+(j-.5)*.3,values,.28,label=b.title());ax.bar_label(bars,fmt='%.1f',fontsize=8)
    ax.set_xticks(range(3),['Fixed','Correction refit','Studentized']);ax.set_title('Fresh AC-positive: temporal fit + correlated pilot');ax.legend(fontsize=8,loc='lower left')
    for ax in axes:ax.axhline(95,color='black',ls='--',lw=1);ax.set_ylim(0,106)
    fig.suptitle('Uncertainty repairs: successes and remaining failures',fontsize=13);fig.text(.5,.015,'Left: anchors42-44. Right: new anchors48-50. Three anchors/case,256 draws/anchor; not paired across studies.\nPilot/design remain fixed within each anchor; this is not unconditional full-pipeline confidence coverage.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(ROOT/'repair_overview.png',dpi=180);fig.savefig(ROOT/'repair_overview.pdf');plt.close(fig)
if __name__=='__main__':main()
