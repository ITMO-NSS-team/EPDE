"""Plot paired 0.4 s field-increment scores in the five omitted recordings."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swarm_term_fields import DEST


def main():
    density = pd.read_csv(DEST/'selected.csv').query("target == 'density'")
    polarization = pd.read_csv(DEST/'combined'/'selected.csv')
    tags = sorted(density.held.unique())
    y = np.arange(len(tags))
    fig, axes = plt.subplots(1,2,figsize=(10,4.15),sharey=True)
    colors = ('#7a838b','#00756d','#a84508')
    for ax, title, df, names, labels in [
        (axes[0],'Density',density,('previous benchmark','baseline plus BlockedPair','native front refit'),
         ('Previous law','Add pair blocking','EPDE front, refitted')),
        (axes[1],'Polarization',polarization,('previous benchmark','baseline plus SlowTurn','native front refit'),
         ('Previous law','Add earlier 12 s spin','EPDE front, refitted'))]:
        values = [df[df.model==n].set_index('held').loc[tags,'r2'].to_numpy() for n in names]
        for idx in y:
            v = [scores[idx] for scores in values]
            ax.plot([min(v),max(v)],[idx,idx],color='#abb6bb',lw=1.4,zorder=1)
        for values_i,label,c,marker in zip(values,labels,colors,('o','s','D')):
            ax.scatter(values_i,y,label=label,color=c,marker=marker,s=38,edgecolor='white',linewidth=.55,zorder=2)
        ax.set_title(title,loc='left',weight='bold',fontsize=13)
        ax.set_xlabel('Held-out R² for the change over 0.4 s')
        ax.grid(axis='x',color='#e0e5e6')
        ax.spines[['left','right','top']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.legend(loc='upper left',bbox_to_anchor=(-.025,-.21),frameon=False,fontsize=8)
    axes[0].set_yticks(y,[t.replace('_',' · ').replace('PWM','PWM ').replace('D','D ').replace('N','N ') for t in tags])
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=.17,left=.19,right=.98,top=.91,bottom=.33)
    fig.savefig(DEST/'field_transfer.png',dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


if __name__=='__main__':
    main()
