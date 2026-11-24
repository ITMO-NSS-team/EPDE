"""Compare complete-system recovery of the confirmed validation designs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
NAMES={'ac':'Allen–Cahn','burgers':'Viscous Burgers','burgers_inviscid':'Inviscid Burgers',
       'duffing':'Duffing','kdv':'KdV','kdv_cossin':'Forced KdV','ks':'Kuramoto–Sivashinsky',
       'lorenz':'Lorenz','lv':'Lotka–Volterra','ns':'Navier–Stokes','ode':'Forced oscillator',
       'pde_compound':'Nonlinear diffusion','pde_divide':'PDE with reciprocal coefficient',
       'vdp':'Van der Pol','wave':'Wave'}


def plot():
    with (ROOT/'summary_confirmation.csv').open() as stream:
        all_rows=[r for r in csv.DictReader(stream) if r['group'] in NAMES]
    rows=[r for r in all_rows if r['design']=='alternate' and r['rule']=='held']
    slabs={r['group']:int(r['six_test']) for r in all_rows if r['rule']=='slab_cross_scale'}
    assert len(rows)==15
    assert len(slabs)==15
    rows.sort(key=lambda r:(int(r['six_base'])-slabs[r['group']],-slabs[r['group']],NAMES[r['group']]))
    fig,ax=plt.subplots(figsize=(8,6.4),layout='constrained')
    for y,r in enumerate(rows):
        a,b=int(r['six_base']),int(r['six_test'])
        c=slabs[r['group']]
        ax.plot([min(a,b,c),max(a,b,c)],[y,y],lw=1.5,color='#bcc4c8',zorder=1)
        ax.scatter(a,y-.18,s=26,color='#264f67',zorder=2,
                   label='Contiguous halves' if y==0 else None)
        ax.scatter(b,y,s=22,color='#bd572c',marker='s',zorder=3,label='Alternating samples' if y==0 else None)
        ax.scatter(c,y+.18,s=36,color='#296e53',marker='^',zorder=3,label='Raw-grid slabs' if y==0 else None)
    ax.set_yticks(range(len(rows)),[NAMES[r['group']] for r in rows])
    ax.set(xticks=range(13),xlim=(-.45,12.45),ylim=(14.6,-.7),xlabel='Correct system in six returned candidates (out of 12 runs)')
    ax.set_title('Equation recovery across 15 datasets',loc='left',pad=14)
    ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.07),ncol=3,frameon=False,borderaxespad=0)
    fig.savefig(ROOT/'recovery_by_dataset.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    plot()
