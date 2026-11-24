"""Transfer gate-only models to all measurable crossings in the held-out recording."""
import json

import numpy as np
import pandas as pd

from video_gate_joint import read_events
from video_gate_response import BASE, CAP, CUTOFF, DEST, FPS, GATE, SIGMA, episodes
from epde.operators.common.multisample import joint_subset_table
from joint_rebound import errors


def crossings(path):
    tag=path.stem
    arena=np.array(json.loads((BASE.parent/tag/'training.json').read_text())['arenas'][tag[-2:]])
    with np.load(path) as src:
        pos,flags=src['raw_pos'],src['quality_flags']
    bad=(flags!=0)|(np.linalg.norm(pos-arena[:2],axis=2)>arena[2]-.25*SIGMA)
    rows=[]
    for i,j in zip(*np.triu_indices(pos.shape[1],1)):
        rel=pos[:,j]-pos[:,i]
        dist=np.linalg.norm(rel,axis=1)
        t,_=episodes(dist,GATE,CUTOFF)
        t=t[(t>=5)&(t+2<len(pos))]
        frames=t[:,None]+np.arange(-5,3)
        t=t[~(bad[frames,i]|bad[frames,j]).any(1)]
        direction=rel[t]/dist[t,None]
        tangent=np.column_stack((-direction[:,1],direction[:,0]))
        inc=(rel[t-1]-rel[t-3])*FPS/2
        closing=-(inc*direction).sum(1)
        d=pd.DataFrame(dict(recording=tag,closing=closing,abs_side=abs((inc*tangent).sum(1)),
                            gate_gap=dist[t]-CAP))
        for step in (1,2):
            d[f'change_{40*step}']=((rel[t+step]-rel[t])*direction).sum(1)*FPS/step+closing
        rows.append(d[closing>0])
    return pd.concat(rows,ignore_index=True)


def joint_transfer(reviewed,population):
    scores=[]
    for ms in (40,80):
        for scope,test_set in [('reviewed clips',reviewed),('all track crossings',population)]:
            observations=[]
            for held,test in test_set.groupby('recording'):
                train=reviewed[reviewed.recording!=held]
                def design(d):
                    hinge=(d.closing-d.gate_gap.clip(lower=0)/(ms/1000)).clip(lower=0)
                    return np.column_stack([d.closing,d.abs_side,hinge])
                x=design(test)
                for target in ('velocity change','outgoing velocity'):
                    def outcome(d):
                        y=d[f'change_{ms}']
                        return y if target=='velocity change' else y-d.closing
                    groups=[(name,design(g),outcome(g),np.ones(len(g))) for name,g in train.groupby('recording')]
                    _,_,_,coefs,ranks=joint_subset_table(groups,subsets=[8,11,15],discrepancy='wape',weighting='samples')
                    for model,coef,rank in zip(('constant','motion','motion + clearance'),coefs,ranks):
                        if not rank:
                            raise ValueError('Gate-only training design is rank deficient')
                        y=outcome(test).to_numpy()
                        pred=x@coef[:-1]+coef[-1]
                        observations.append(pd.DataFrame(dict(held=held,model=model,target=target,y=y,pred=pred)))
            for (model,target),g in pd.concat(observations).groupby(['model','target']):
                for label,sub in [('balanced',g),*g.groupby('held')]:
                    weights=1/sub.groupby('held')['y'].transform('size')
                    rmse,r2=errors(sub.y,sub.pred,weights)
                    scores.append(dict(horizon_ms=ms,scope=scope,target=target,model=model,held=label,
                                       n=len(sub),rmse_px_s=rmse,r2=r2))
    return pd.DataFrame(scores)


if __name__=='__main__':
    population=pd.concat([crossings(path) for path in sorted(BASE.glob('N*_PWM*_D*.npz'))])
    joint_transfer(read_events(),population).to_csv(DEST/'gate_only_transfer.csv',index=False)
