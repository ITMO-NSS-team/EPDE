"""Within-pair matched third-neighbor comparisons and plots for the three population sweeps."""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from new_pair_interactions import DEST


def matched(physical=False):
    folder = DEST/'physical' if physical else DEST
    d = pd.read_csv(folder/('events.csv.gz' if physical else 'encounters.csv.gz'))
    d = d[(d.gate==1.1)&(d.incoming<0)].copy()
    cols = ['incoming','side','prior','r']
    tolerances = np.array([.30,.45,.35,.02])
    records = []
    for (exp,size,i,j),part in d.groupby(['experiment','N','id_i','id_j']):
        a,b = part[part.third],part[~part.third]
        if a.empty or b.empty:
            continue
        delta = (a[cols].to_numpy()[:,None,:]-b[cols].to_numpy()[None,:,:])/tolerances
        good = (abs(delta)<=1).all(2)
        cost = np.sum(delta**2,axis=2)
        cost[~good] = 1+len(cols)*min(len(a),len(b))
        ia,ib = linear_sum_assignment(cost)
        for x,y in zip(ia,ib):
            if good[x,y]:
                v,w = a.iloc[x],b.iloc[y]
                records.append(dict(experiment=exp,N=size,id_i=i,id_j=j,third_frame=v.frame,
                    isolated_frame=w.frame,third_reversed=int(v.outward>0),isolated_reversed=int(w.outward>0),
                    third_out=v.outward,isolated_out=w.outward,third_in=v.incoming,isolated_in=w.incoming,
                    third_side=v.side,isolated_side=w.side))
    m = pd.DataFrame(records)
    m.to_csv(folder/'matched_third.csv.gz',index=False)
    m['delta_reversal'] = m.third_reversed-m.isolated_reversed
    m['delta_out'] = m.third_out-m.isolated_out
    m['delta_in'] = m.third_in-m.isolated_in
    m['delta_change'] = m.delta_out-m.delta_in
    rng = np.random.default_rng(3407)
    summaries,draws = [],[]
    for exp,s in m.groupby('experiment'):
        blocks = s.groupby('N').agg(k=('N','size'),reversal=('delta_reversal','sum'),change=('delta_change','sum')).to_numpy()
        counts,rev,change = blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].sum(axis=1).T
        draws.append(np.stack([rev/counts,change/counts],axis=-1))
        summaries.append(dict(experiment=exp,matches=len(s),stages=s.N.nunique(),third_reversed=s.third_reversed.mean(),
            isolated_reversed=s.isolated_reversed.mean(),delta_reversal=s.delta_reversal.mean(),
            delta_out=s.delta_out.mean(),delta_change=s.delta_change.mean(),delta_in=s.delta_in.mean()))
    sums = pd.DataFrame(summaries)
    row = sums.select_dtypes('number').mean().to_dict()
    row.update(experiment='equal experiment mean',matches=len(m),stages=sums.stages.sum())
    ci = np.quantile(np.mean(draws,axis=0),[.025,.975],axis=0)
    row.update(reversal_ci_low=ci[0,0],reversal_ci_high=ci[1,0],change_ci_low=ci[0,1],change_ci_high=ci[1,1])
    sums = pd.concat([sums,pd.DataFrame([row])],ignore_index=True)
    sums.to_csv(folder/'matched_summary.csv',index=False)
    print(sums.round(4).to_string(index=False))


def plot(physical=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = DEST/'physical' if physical else DEST
    d = pd.read_csv(folder/('encounters.csv' if physical else 'event_summary.csv'))
    if physical:
        d = d.rename(columns={'opening':'reversed'})
    d = d[(d.group=='all')&(d.experiment!='all')]
    scores = pd.read_csv(folder/'with_drive'/'heldout.csv')
    scores = scores[(scores.target=='outward') & (scores.group=='all')]
    fig,ax = plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    colors = ['#006d77','#be592c','#5151a5']
    for color,(exp,s) in zip(colors,d.groupby('experiment')):
        lab = exp.replace('_',' ')
        abscissa = s.entry_mm if physical else s.gate
        ax[0].plot(abscissa,100*s.reversed,'o-',color=color,label=lab)
        m = scores[scores.held==exp].set_index('model')
        ax[1].plot([0,1,2,3],m.loc[['free motion only','motion history','history and clearance','native EPDE front'],'r2'],
                   'o-',color=color)
    ranges = pd.read_csv(folder/('encounters.csv' if physical else 'separation.csv'))
    pooled = ranges[(ranges.group=='all')&(ranges.experiment=='all')]
    x = pooled.entry_mm if physical else pooled.gate
    increased = pooled.increased_distance if physical else pooled.range_increased
    ax[0].plot(x,100*increased,'s--',color='#28313b',label='Distance increased, pooled')
    ax[0].set(xticks=[48,50,52,54] if physical else [1,1.05,1.1,1.15],
              xlabel='Estimated entry separation (mm)' if physical else 'Entry separation / reference spacing',
              ylabel='Approaches (%)',title='Outward motion along the entry axis',ylim=(0,100))
    if physical:
        ax[0].axvline(48.7,color='#69737a',linewidth=.8,zorder=0)
    ax[0].legend(frameon=False,fontsize=8)
    ax[1].set(xticks=[0,1,2,3],xticklabels=['Free\nmotion','+ motion\nhistory','+ pair\ngeometry','EPDE\nfront'],
              ylabel='Held-out R²',title='Normal-velocity forecast')
    for a in ax:
        a.grid(axis='y',alpha=.2)
        a.spines[['right','top']].set_visible(False)
    fig.savefig(folder/'pair_response.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plot',action='store_true')
    p.add_argument('--physical',action='store_true')
    args = p.parse_args()
    if args.plot:
        plot(args.physical)
    else:
        matched(args.physical)
