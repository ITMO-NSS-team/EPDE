"""Frame-resolved encounter paths and signed third-neighbor geometry at the gate."""
import numpy as np
import pandas as pd

from shared_interaction_data import DEST, MS, stages

OUT = DEST / 'direction_and_paths'
KEY = ['recording','stage','frame','i','j']
KNOTS = np.array([.95,1.05,1.15,1.30,1.50])
BASIS = [f'radial_{k}' for k in range(len(KNOTS)-1)]
PATHS = ('near, small turn','near, large turn','early clearance','later clearance','reapproach')


def classes(radii,angles,hz,clearance=1.30,angle_deg=10.):
    clear = radii>=clearance
    released = clear.any(1)
    first = clear.argmax(1)
    reentry = ((np.arange(radii.shape[1])>first[:,None]) & (radii<1.10)).any(1) & released
    swing = np.rad2deg(np.max(abs(angles-angles[:,[0]]),axis=1))
    return np.select([reentry,released & (first<=round(.080*hz)),released,swing>=angle_deg],
                     [PATHS[4],PATHS[2],PATHS[3],PATHS[1]],default=PATHS[0])


def third_geometry(p,cols,i,j,axis):
    """Each other robot counts once even when it is close to both focal robots."""
    delta_i=p[np.arange(len(p)),i,None]-p
    delta_j=p[np.arange(len(p)),j,None]-p
    inactive=(cols==i[:,None])|(cols==j[:,None])
    di,dj=np.where(inactive,np.inf,abs(delta_i)),np.where(inactive,np.inf,abs(delta_j))
    def projection(delta,dist):
        return np.divide(delta*axis.conj()[:,None],dist,out=np.zeros_like(delta),
                         where=(dist>0)&np.isfinite(dist))
    ui,uj=projection(delta_i,di),projection(delta_j,dj)
    context=((np.maximum(1.30-dj,0)*uj)-(np.maximum(1.30-di,0)*ui)).sum(1)
    hat=[]
    for k in range(len(BASIS)):
        heights=np.eye(len(KNOTS))[k]
        hat.append((np.interp(dj,KNOTS,heights)*uj.imag-np.interp(di,KNOTS,heights)*ui.imag).sum(1))
    return context, np.minimum(di,dj)<1.30, np.stack(hat,axis=1)


def event_details(g,hz,ids,pos,phi,bad,radius):
    t=g.frame.to_numpy()
    i,j=np.searchsorted(ids,g.i),np.searchsorted(ids,g.j)
    if not (np.array_equal(ids[i],g.i) and np.array_equal(ids[j],g.j)):
        raise ValueError('Unmatched focal IDs')
    dt=np.arange(round(.40*hz)+1)
    frames=t[:,None]+dt
    rel=pos[frames,j[:,None]]-pos[frames,i[:,None]]
    radii=abs(rel);angles=np.unwrap(np.angle(rel),axis=1)
    n=rel[:,0]/radii[:,0]
    np.testing.assert_allclose(radii[:,0],g.r,atol=1e-12)
    for ms in MS:
        uv=(rel[:,round(hz*ms/1000)]-rel[:,0])*n.conj()/(ms/1000)
        np.testing.assert_allclose(uv.real,g[f'y{ms}_n'],atol=1e-12)
        np.testing.assert_allclose(uv.imag,g[f'y{ms}_t'],atol=1e-12)
    ctx,close,basis=third_geometry(pos[t],np.arange(len(ids))[None,:],i,j,n)
    np.testing.assert_allclose(ctx,g.third_n+1j*g.third_t,atol=1e-12)
    w=np.sign(phi[t-round(.04*hz)]-phi[t-round(.20*hz)])
    # Only the sign pairing is used, so spinning with either angular convention is supported.
    wi,wj=w[np.arange(len(t)),i],w[np.arange(len(t)),j]
    out=g[KEY].copy()
    out['spin_class']=np.minimum(wi,wj)*3+np.maximum(wi,wj)
    out['crowd']=close.sum(1)
    out['third_flagged']=(bad[t]&close).any(1)
    at_gate=pos[t]
    other_dist=np.minimum(abs(at_gate-pos[t,i,None]),abs(at_gate-pos[t,j,None]))
    other_dist[np.arange(len(t)),i]=np.inf
    other_dist[np.arange(len(t)),j]=np.inf
    out['nearest_third']=other_dist.min(axis=1)
    out['kernel_flagged']=(bad[t] & (other_dist<1.50)).any(1)
    out['wall_gap']=radius-np.maximum(abs(pos[t,i]),abs(pos[t,j]))
    out['abs_third']=abs(ctx.imag)
    out[BASIS]=basis
    out['path']=classes(radii,angles,hz)
    for threshold,angle in [(1.2,10),(1.4,10),(1.3,5),(1.3,15)]:
        out[f'path_{threshold:.2f}_{angle}']=classes(radii,angles,hz,threshold,angle)
    out['turn_deg']=np.rad2deg(np.max(abs(angles-angles[:,[0]]),axis=1))
    out['gap_speed_400']=(radii[:,-1]-radii[:,0])/.4
    # A disjoint pre-gate velocity, held out of matching, probes motion already underway.
    for field,start,stop in [('lead_t',.64,.44),('earlier_t',.88,.68)]:
        older=t-round(start*hz)
        end=(t-round(stop*hz)).clip(min=0)
        history=(t[:,None]-np.arange(round(stop*hz),round(start*hz)+1)).clip(min=0)
        valid=(older>=0) & ~(bad[history,i[:,None]]|bad[history,j[:,None]]).any(1)
        begin=older.clip(min=0)
        lead=((pos[end,j]-pos[end,i])-(pos[begin,j]-pos[begin,i]))*n.conj()/.20
        out[field]=np.where(valid,lead.imag,np.nan)
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    archive={}
    for tag,hz,_,ids,pos,phi,bad,rad in stages():
        if tag not in archive:
            archive[tag]=pd.read_parquet(DEST/'data'/f'{tag}_pairs.parquet').query("mode=='native' and kind=='near'")
        g=archive[tag]
        g=g[g.stage==len(ids)]
        for start in range(0,len(g),1024):
            rows.append(event_details(g.iloc[start:start+1024],hz,ids,pos,phi,bad,rad))
    result=pd.concat(rows,ignore_index=True)
    assert len(result)==sum(len(g) for g in archive.values()) and not result.duplicated(KEY).any()
    result.to_parquet(OUT/'event_details.parquet',index=False)
    print('Verified paths, all three velocity endpoints and third-neighbor geometry for',len(result),'gates.')


if __name__=='__main__':
    main()
