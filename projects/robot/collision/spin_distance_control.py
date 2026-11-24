"""Within-pair near versus separated comparison of spin-aligned tangential motion."""
import numpy as np
import pandas as pd

from physics import CAP, CUTOFF, DT, LAG, OUT, input_at, robot_parameters, rotate, wall_vectors
from projected_data import fold_calibrations, projected_tracks


def compare(raw, clean, calibration):
    rows = np.arange(LAG,len(raw.pos)-LAG,LAG)
    lookup,common,arenas = calibration
    pars = robot_parameters(clean,lookup)
    p,e,velocity,omega = input_at(clean,rows)
    base = (pars[:,0,None]*e + pars[:,1,None]*rotate(e) + common[0]*velocity
            + common[1]*wall_vectors(p,arenas[int(clean.tag[-2:])]))
    y = (raw.pos[rows+LAG]-raw.pos[rows])/DT
    left,right = np.triu_indices(len(raw.ids),1)
    delta = raw.pos[rows][:,right]-raw.pos[rows][:,left]
    dist = np.linalg.norm(delta,axis=2)
    near = (dist>0)&(dist<CUTOFF)
    far = (dist>=1.3*CAP)&(dist<1.6*CAP)
    f,k = np.nonzero((near|far)&(omega[:,left]*omega[:,right]>0))
    i,j = left[k],right[k]
    spin = np.sign(omega[f,i])
    tangent = rotate(delta[f,k]/dist[f,k,None])
    def aligned(v):
        return -spin*np.sum((v[f,j]-v[f,i])*tangent,axis=1)
    measured = aligned(y)
    table = pd.DataFrame(dict(pair=k,sign=spin,near=near[f,k],vt=measured,residual=measured-aligned(base)))
    strata = table.groupby(['pair','sign','near']).agg(n=('vt','size'),vt=('vt','mean'),
                                                       residual=('residual','mean')).unstack('near')
    # Use strata observed at both distances and give each retained pair/sign equal weight.
    results = []
    for cut in (1,5,10):
        use = strata[(strata['n'][True]>=cut)&(strata['n'][False]>=cut)]
        assert len(use)
        row = dict(recording=raw.tag,min_origins_per_band=cut,pairs=use.index.get_level_values('pair').nunique(),
                   pair_sign_strata=len(use),near_origins=int(use['n'][True].sum()),far_origins=int(use['n'][False].sum()))
        for k in ('vt','residual'):
            row['near_'+k] = use[k][True].mean()
            row['far_'+k] = use[k][False].mean()
            row['difference_'+k] = (use[k][True]-use[k][False]).mean()
        results.append(row)
    return results


if __name__=='__main__':
    raw,clean = projected_tracks()
    params = fold_calibrations(clean)
    table = pd.DataFrame([r for tag in sorted(raw) for r in compare(raw[tag],clean[tag],params[tag])])
    table.to_csv(OUT/'preprocessed'/'spin_distance_control.csv',index=False)
    print(table[table.min_origins_per_band==5].round(2).to_string(index=False))
    print(table.groupby('min_origins_per_band')[['difference_vt','difference_residual']].mean().round(2))
