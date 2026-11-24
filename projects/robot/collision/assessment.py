"""Freeze each model on the four training recordings, then score and simulate the omitted setup."""
import json
from itertools import combinations

import numpy as np

from discovery import mse, solve, weak_average, wf, write_json
from physics import (CAP, CUTOFF, DT, FPS, LAG, NAMES, OUT, SEEDS, SIGMA, design, geometry, heading,
                     input_at, motion_inputs, partition, recordings, robot_parameters)


def select(held):
    dest = OUT/held
    with np.load(dest/'prepared.npz') as z:
        strong, single, valid = (z[name].mean(0) for name in ('strong_gram','single_gram','valid_gram'))
        models = {}
        def model(g,names):
            beta = solve(g,names)
            return dict(coef={name:float(beta[NAMES.index(name)]) for name in names},
                        validation_rmse=np.sqrt(mse(valid,beta)), train_noise_sd=np.sqrt(mse(strong,beta)))
        models['self'] = model(strong, [])
        models['local_pair'] = model(single, ['push','front'])
        models['constant_drive'] = model(strong, ['push','drive','drive_in'])
        models['banded_drive'] = model(strong, [f'band{i}' for i in range(5)]+['drive','drive_in'])
        for form in ('strong','weak'):
            all_supports = set()
            for seed in SEEDS:
                archive = json.loads((dest/f'{form}_{seed}.json').read_text())
                if (archive['form'],archive['seed'],archive['epochs'])!=(form,seed,50):
                    raise ValueError(f'Wrong search receipt for {held}/{form}/{seed}')
                sx,sy=z[form+'_X'],z[form+'_y'].astype(float)
                for row in archive['native_front']:
                    coef=np.array([row['terms'].get(name,0.) for name in NAMES])
                    residual=sx@coef+row['intercept']-sy
                    np.testing.assert_allclose(np.abs(residual).sum()/np.abs(sy).sum(),row['objectives'][0],
                                               rtol=1e-6,err_msg='Native front failed physical-column replay')
                all_supports.update(tuple(sorted(row['terms'])) for row in archive['evaluated']+archive['native_front'])
            gram = z[form+'_gram'].mean(0)
            candidates, exact, identifiable = [], None, 0
            def order(item):
                return item['validation_rmse'],len(item['coef']),tuple(item['coef'])
            for size in range(7):
                for names in combinations(sorted(NAMES),size):
                    if names:
                        ix = [NAMES.index(n) for n in names]
                        sub = gram[np.ix_(ix,ix)]
                        scale = np.sqrt(np.maximum(1e-20,sub.diagonal()))
                        if np.linalg.matrix_rank(sub/scale[:,None]/scale, tol=1e-8)<len(ix):
                            continue
                    identifiable+=1
                    fitted=model(gram,names)
                    if exact is None or order(fitted)<order(exact):
                        exact=fitted
                    if names in all_supports:
                        candidates.append(fitted)
            if not candidates:
                raise ValueError(f'No eligible {form} candidates')
            candidates.sort(key=order)
            models[form] = candidates[0]
            write_json(dest/f'{form}_refits.json',candidates)
            write_json(dest/f'{form}_coverage.json',dict(discovered=len(candidates),identifiable=identifiable,
                       validation_best_over_subsets=exact,archived_validation_best=candidates[0]))
    winner = min(models,key=lambda n:(models[n]['validation_rmse'],len(models[n]['coef']),n))
    return models,winner


def array(model):
    return np.array([model['coef'].get(n,0.) for n in NAMES])


def velocity_scores(track,rows,y,pred,contacts):
    err = pred-y
    out = dict(r2=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)), rmse=float(np.sqrt(np.mean(err**2))))
    out['contact_classes'] = {}
    for k in range(5):
        m = np.minimum(contacts,4)==k
        out['contact_classes'][str(k)] = dict(n=int(m.sum()),rmse=float(np.sqrt(np.mean(err[m]**2))) if m.any() else None)
    a,b = np.triu_indices(len(track.ids),1)
    proj = []
    counts=np.zeros(5,int)
    obs_sum,pred_sum=np.zeros(5),np.zeros(5)
    for start in range(0,len(rows),48):
        sl=slice(start,start+48)
        pos=track.pos[rows[sl]]
        delta=pos[:,b]-pos[:,a]
        dist=np.linalg.norm(delta,axis=-1)
        t,k=np.nonzero((dist<CUTOFF)&(dist>0))
        if not len(k):
            continue
        u=delta[t,k]/dist[t,k,None]
        er=err[sl][t,b[k]]-err[sl][t,a[k]]
        normal=(er*u).sum(1)
        tangent=u[:,0]*er[:,1]-u[:,1]*er[:,0]
        proj.append(np.column_stack((normal,tangent)))
        band=np.digitize(dist[t,k]/SIGMA,[.93,.97,1.,1.05])
        obs=np.sum((y[sl][t,b[k]]-y[sl][t,a[k]])*u,axis=1)
        counts+=np.bincount(band,minlength=5)
        obs_sum+=np.bincount(band,weights=obs,minlength=5)
        pred_sum+=np.bincount(band,weights=obs+normal,minlength=5)
    e=np.concatenate(proj)
    out['pair_mae']=np.mean(np.abs(e),0).tolist()
    out['pair_rmse']=np.sqrt(np.mean(e**2,0)).tolist()
    out['bond_profile']=dict(count=counts.tolist(),
                             observed=[float(v/c) if c else None for v,c in zip(obs_sum,counts)],
                             predicted=[float(v/c) if c else None for v,c in zip(pred_sum,counts)])
    return out


def packing(pos):
    nearest,counts=[],[]
    for i in range(0,len(pos),48):
        d,_=geometry(pos[i:i+48])
        nearest.extend(d.min(-1).ravel())
        counts.extend(np.minimum((d<CUTOFF).sum(-1),4).ravel())
    v=np.array(nearest)
    hist=np.histogram(v/SIGMA,[0,.85,.9,.95,1,1.05,1.1,1.25,np.inf])[0]
    return dict(gap_hist=(hist/hist.sum()).tolist(), contacts=(np.bincount(counts,minlength=5)/len(v)).tolist(),
                gap_p001_px=float(np.percentile(v,.1)), overlap_fraction=float(np.mean(v<CAP-1.)),
                strict_overlap_fraction=float(np.mean(v<CAP)))


def simulate(track,params,common,arena,model,seed,core):
    from swarm_sim import project
    rng=np.random.default_rng(seed)
    p0,p=track.pos[[0,LAG]].copy()
    prev=(p-p0)/DT
    phi=track.phi[LAG].copy()
    omega=(phi-track.phi[0])/DT
    container=dict(xc=arena[0],yc=arena[1],r_max=arena[2])
    out=[p]
    for _ in range(2,len(track.pos[::LAG])):
        X,base,_=motion_inputs(p,heading(phi),prev,omega,params,common,arena)
        v=base+X@array(model)+model['train_noise_sd']*rng.standard_normal(p.shape)
        new=project(p+DT*v,container,core=core)
        prev=(new-p)/DT
        omega=params[:,2]+params[:,3]*rng.standard_normal(len(phi))/DT
        phi=phi+DT*omega
        p=new
        out.append(p)
    return packing(np.array(out))


def memory_audit(train,test,meta,lookup,common,arenas,weak_model):
    # Compare on the same interior; each history feature ends at or before its prediction time.
    def memory(tr):
        rows=np.arange(24*LAG,len(tr.pos)-LAG)
        p,e,prev,w=input_at(tr,rows)
        from physics import rotate,wall_vectors
        params=robot_parameters(tr,lookup)
        prop=params[:,0,None]*e+params[:,1,None]*rotate(e)
        base=prop+common[1]*wall_vectors(p,arenas[int(tr.tag[-2:])])
        mem=[prev]+[(tr.pos[rows-a*LAG]-tr.pos[rows-b*LAG])/((b-a)*DT)
                    for a,b in [(1,2),(2,4),(4,9),(9,24)]]
        return rows,np.stack(mem,axis=-1),base,(tr.pos[rows+LAG]-p)/DT
    grams={form:[] for form in ('strong','weak')}
    for tr in train:
        rows,x,b,y=memory(tr)
        m,s,q=meta['windows'][tr.tag]
        val,der=wf.window_kernels([m],[1/FPS],[q],[1])[0]
        for form,X,Y in [('strong',x[::LAG],(y-b)[::LAG]),
                         ('weak',weak_average(x,val,s),-weak_average(tr.pos[rows],der,s)-weak_average(b,val,s))]:
            a=np.column_stack((X.reshape(-1,5),Y.ravel()))
            grams[form].append((a.T@a)/len(a))
    rows,x,b,y=memory(test)
    rows,x,b,y=rows[::LAG],x[::LAG],b[::LAG],y[::LAG]
    audit={}
    for form,gs in grams.items():
        g=np.mean(gs,axis=0)
        coef=np.linalg.lstsq(g[:-1,:-1],g[:-1,-1],rcond=None)[0]
        audit[form]=dict(coef=coef.tolist(),rmse=float(np.sqrt(np.mean((y-b-x@coef)**2))))
    rr,X,base,target,_=design(test,lookup,common,arenas,LAG)
    idx=np.searchsorted(rr,rows)
    audit['weak_contact_rmse']=float(np.sqrt(np.mean((target[idx]-base[idx]-X[idx]@array(weak_model))**2)))
    return audit


def propulsion_audit(train,test,models,lookup,common,arenas,meta):
    """Sensitivity of frozen supports to unobserved individual propulsion in heavily crowded runs."""
    free_rows={}
    for tr in train:
        pos=tr.pos[LAG:-LAG:LAG]
        free=(geometry(pos)[0]>=CUTOFF).all(-1).sum(0)
        for rid,count in zip(tr.ids,free):
            key=(tr.pwm,int(rid))
            free_rows[key]=free_rows.get(key,0)+int(count)
    missing=[key for key,n in free_rows.items() if n==0]
    audit=dict(unobserved=missing,heldout_affected=[int(i) for i in test.ids if (test.pwm,int(i)) in missing])
    if not missing:
        audit['scores']={k:m['heldout']['r2'] for k,m in models.items()}
        return audit
    alt={k:v.copy() for k,v in lookup.items()}
    for pwm in {k[0] for k in missing}:
        prior=np.mean([alt[k][:2] for k,count in free_rows.items() if k[0]==pwm and count],axis=0)
        for key in [k for k in missing if k[0]==pwm]+[(pwm,-1)]:
            alt[key][:2]=prior
    from discovery import gram,weak_design
    gs={k:[] for k in ('strong','weak','single')}
    for tr in train:
        rows,X,b,y,n=design(tr,alt,common,arenas,1)
        wx,wy=weak_design(tr.pos[rows],X,b,meta['windows'][tr.tag])
        gs['weak'].append(gram(wx,wy))
        xs,ys=X[::LAG],(y-b)[::LAG]
        gs['strong'].append(gram(xs,ys))
        m=n[::LAG]==1
        gs['single'].append(gram(xs[m],ys[m]))
    _,x,b,y,_=design(test,alt,common,arenas,LAG)
    scores={}
    for name,model in models.items():
        g=gs['single' if name=='local_pair' else 'weak' if name=='weak' else 'strong']
        coef=solve(np.mean(g,axis=0),list(model['coef']))
        scores[name]=float(1-np.sum((y-b-x@coef)**2)/np.sum((y-y.mean())**2))
    audit['scores']=scores
    return audit


def assess(held):
    dest=OUT/held
    models,winner=select(held)
    meta=json.loads((dest/'training.json').read_text())
    train,_,test=partition(recordings(),held)
    lookup={tuple(map(int,k.split('/'))):np.array(v) for k,v in meta['lookup'].items()}
    arenas={int(k):np.array(v) for k,v in meta['arenas'].items()}
    common=np.array(meta['common'])
    rows,X,base,y,n=design(test,lookup,common,arenas,LAG)
    for model in models.values():
        model['heldout']=velocity_scores(test,rows,y,base+X@array(model),n)
    selected_search=min(('strong','weak'),key=lambda k:models[k]['validation_rmse'])
    best_baseline=min(('self','local_pair','constant_drive','banded_drive'),key=lambda k:models[k]['validation_rmse'])
    actual=packing(test.pos[LAG::LAG])
    rep={}
    sim_params=robot_parameters(test,lookup)
    speeds=[abs(v[2]) for (pwm,rid),v in lookup.items() if pwm==test.pwm and rid!=-1]
    for j,rid in enumerate(test.ids):
        if (test.pwm,int(rid)) not in lookup:
            # The recording starts with ten frames of observed history. Preserve an unseen
            # robot's initial chirality when borrowing the motor setting's typical spin magnitude.
            sign=np.sign(test.phi[LAG,j]-test.phi[0,j])
            sim_params[j,2]=sign*np.mean(speeds)
    for key in (selected_search,best_baseline):
        rep[key]={}
        for name,core in [('cap',CAP),('unprojected',0.)]:
            runs=[simulate(test,sim_params,common,arenas[int(held[-2:])],models[key],seed,core)
                  for seed in range(4)]
            for r in runs:
                for stat in ('gap_hist','contacts'):
                    r[stat+'_tv']=float(.5*np.sum(np.abs(np.array(r[stat])-actual[stat])))
            rep[key][name]=runs
    audit=memory_audit(train,test,meta,lookup,common,arenas,models['weak'])
    prior_audit=propulsion_audit(train,test,models,lookup,common,arenas,meta)
    write_json(dest/'assessment.json',dict(held=held,selected=winner,selected_search=selected_search,
               best_baseline=best_baseline,models=models,observed=actual,simulations=rep,memory=audit,
               propulsion_sensitivity=prior_audit))
