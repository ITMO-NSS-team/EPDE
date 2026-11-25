"""Replay selected equations independently and check geometric and cohort invariants."""
import json
import numpy as np
import pandas as pd

from shared_interaction_data import DEST, DRIVE, MS, extract
from shared_interaction_fit import library


def geometry_checks():
    hz=25
    t=np.arange(110)/hz
    dist=1.5-.13*t
    r=dist*np.exp(1j*.22*t)
    pos=np.stack([-.5*r,.5*r,.5*r+1.17j],axis=1)
    phi=t[:,None]*np.array([2.,-1.,3.])
    def run(p,ang):
        _,e,_=extract('synthetic',hz,20,np.arange(3),p,ang,np.zeros(p.shape,bool),100,'native')
        return pd.concat(e).query("kind=='near' and i==0 and j==1").reset_index(drop=True)
    d=run(pos,phi)
    assert len(d)==1
    for name,a,sign in [('reflection',run(pos.conj(),-phi),-1),
                        ('exchange',run(pos[:,[1,0,2]],phi[:,[1,0,2]]),1),
                        ('rotation',run(2-3j+pos*np.exp(.71j),phi+.71),1)]:
        for axis,factor in [('n',1),('t',sign)]:
            for ms in MS:
                np.testing.assert_allclose(a[f'y{ms}_{axis}'],factor*d[f'y{ms}_{axis}'],atol=1e-12)
                f=d['phase_'+axis].to_numpy()
                x=library(d,ms/1000,axis,f)
                actual=library(a,ms/1000,axis,a['phase_'+axis].to_numpy())
                np.testing.assert_allclose(actual,factor*x,atol=1e-11,err_msg=name)
    np.testing.assert_allclose(d.spin_sum, .5)


def validate_study(directory,fronts):
    count=0
    for folder in sorted(directory.glob('*')):
        obj=pd.read_csv(folder/'search_objectives.csv')
        for (ax,h,timing),g in obj.groupby(['axis','horizon','timing']):
            assert set(g['mask'])==set(range(1,64))
            expected=set()
            good=g[g.full_rank & np.isfinite(g.wape)&np.isfinite(g.dyad)]
            for row in good.itertuples():
                better=(good.wape<=row.wape)&(good.dyad<=row.dyad)&((good.wape<row.wape)|(good.dyad<row.dyad))
                if not better.any():
                    expected.add(row.mask)
            assert set(g.loc[g.on_front,'mask'])==expected
            found=fronts[(fronts.fold==g.fold.iloc[0])&(fronts['mode']==g['mode'].iloc[0])&
                         (fronts.axis==ax)&(fronts.timing==timing)]
            if h:
                found=found[found.horizon==h]
            assert set(found['mask'])==expected
        file=folder/'predictions.parquet'
        if not file.exists():
            continue
        out=pd.read_parquet(file).query("kind=='near'")
        fits=pd.read_csv(folder/'free_fits.csv').set_index('horizon')
        fold=out.fold.iloc[0]; mode=out['mode'].iloc[0]
        for tag,p in out.groupby('recording'):
            raw=pd.read_parquet(DEST/'data'/f'{tag}_pairs.parquet')
            raw=raw[(raw['mode']==mode)&(raw.kind=='near')].set_index(['stage','frame','i','j'])
            for (axis,ms),g in p.groupby(['axis','horizon']):
                assert tag not in fits.loc[ms,'training'].split(',')
                keys=g.set_index(['stage','frame','i','j']).index
                assert len(keys)==len(raw) and keys.is_unique
                d=raw.loc[keys]
                free=d[[name+'_'+axis for name in DRIVE]].to_numpy()@np.array(json.loads(fits.loc[ms,'coeff']))
                np.testing.assert_allclose(free,g.free,atol=1e-11)
                np.testing.assert_allclose(g.y,d[f'y{ms}_{axis}'],atol=1e-11)
                for timing,col in [('separate','separate'),('log','log_time')]:
                    row=fronts[(fronts.fold==fold)&(fronts['mode']==mode)&(fronts.held==tag)&(fronts.axis==axis)&
                               (fronts.horizon==ms)&(fronts.timing==timing)&fronts.selected].iloc[0]
                    b=np.array(json.loads(row.coeff))
                    if timing=='log':
                        b=b[:6]+b[6:]*np.log(ms/80)
                    predicted=free+library(d,ms/1000,axis,free)@b
                    np.testing.assert_allclose(predicted,g[col],atol=1e-11)
                    np.testing.assert_allclose(48.7*np.sqrt(((g[col]-g.y)**2).mean()),row.rmse_mm_s,atol=1e-9)
                    count+=len(g)
    for _,g in fronts.groupby(['mode','fold','held','horizon','axis','timing']):
        assert g.selected.sum()==1
    return count


def main():
    geometry_checks()
    count=validate_study(DEST/'folds',pd.read_csv(DEST/'all_fronts.csv'))
    count+=validate_study(DEST/'drive_sensitivity',pd.read_csv(DEST/'fixed_drive_fronts.csv'))
    print('Geometry, holdout membership, exact fronts and',count,'forecast values verified.')


if __name__=='__main__':
    main()
