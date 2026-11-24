"""Finite-panel diagnostic of the user's first-order ODE_simple_discovery."""
import raw_operator as O
R,Q,J,A,np=O.R,O.Q,O.J,O.A,O.np
import json,csv,hashlib


def panel():
    atom=lambda name,**params:(name,tuple(sorted({'power':1.,**params}.items())))
    u=(atom('u'),);d=(atom('du/dx0'),)
    u2=(atom('u',power=2.),);u3=(atom('u',power=3.),);d2=(atom('du/dx0',power=2.),)
    sn=(atom('sin',freq=1.,dim=0.),);co=(atom('cos',freq=1.,dim=0.),)
    us=R.canon(u+sn);dc=R.canon(d+co);uc=R.canon(u+co);ds=R.canon(d+sn)
    spec=[('energy',d2,[u2,()],[-1.,2.69]),('energy_without_constant',d2,[u2],None),
          ('phase_sin',dc,[us,()],[-1.,1.]),('phase_sin_without_constant',dc,[us],None),
          ('phase_cos',ds,[uc,()],[1.,-1.3]),('phase_cos_without_constant',ds,[uc],None),
          ('explicit_harmonic',d,[sn,co],[-1.3,1.]),('explicit_harmonic_plus_constant',d,[sn,co,()],[-1.3,1.,0.]),
          ('explicit_u_cos',d,[u,co],[-1.3,2.69]),('autonomous_linear',d,[u,()],None),
          ('autonomous_cubic',d,[u,u2,u3,()],None)]
    labels=sorted({z for _,y,x,_ in spec for z in [y,*x]});lookup={x:i for i,x in enumerate(labels)}
    cs=[dict(id=name,target=y,terms=x,xi=[lookup[v] for v in x],yi=lookup[y],exact=truth,
             interpretation='trajectory_identity' if truth is not None else 'misspecified_control') for name,y,x,truth in spec]
    return cs,labels


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    # The source defines these samples analytically. No equation is integrated.
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t);du=np.cos(t)-1.3*np.sin(t)
    from types import SimpleNamespace
    p=SimpleNamespace(shape=(200,),data={'u':u},grids=[t],named_arrays={})
    cs,labels=panel();terms,opaque=R.symbols(p,labels);ci=labels.index(())
    identities=dict(energy=float(np.max(np.abs(u*u+du*du-2.69))),
        phase_sin=float(np.max(np.abs(u*np.sin(t)+du*np.cos(t)-1))),
        phase_cos=float(np.max(np.abs(u*np.cos(t)-du*np.sin(t)-1.3))))
    assert max(identities.values())<3e-15
    rows=[];folds=[];dyads=[];point_rows=[]
    for noise,seed in [(0,0),*( (5,s) for s in range(3) )]:
        sig=noise*.01*np.std(u);rng=np.random.default_rng(seed);raw={'u':u+sig*rng.standard_normal(len(u))}
        freq,_=R.bandwidth(raw,[t]);plan=R.local_plan([t],terms,freq['loo'],5);A.common_width(5)
        db=R.LocalBank([t],terms,opaque,plan);dm=db.matrix(raw)
        field=R.WF.Fields(raw,[t],opaque);point=np.column_stack([field.point(term.data,term.coords,term.opaque) for term in terms])[10:-10]
        for c in cs:
            score=R.SV.dyad_scores(dm[:,c['xi']],dm[:,c['yi']],np.ones(len(dm)),plan['shape'],fit_intercept=False)
            pos=c['xi'].index(ci) if ci in c['xi'] else None
            dyads.append(dict(noise=noise,seed=seed,candidate=c['id'],dyad=float(np.sum(score)),
                constant_dyad=float(score[pos]) if pos is not None else 0.,per_term=score))
            coef,_=R.signed_fit(point.T@point,point,c['xi'],c['yi']);res=point[:,c['yi']]-point[:,c['xi']]@coef
            point_rows.append(dict(noise=noise,seed=seed,candidate=c['id'],coefficients=coef,
                residual_energy=float(res@res/(point[:,c['yi']]@point[:,c['yi']]))))
        indices=[np.arange(h,len(t),3) for h in range(3)];groups=[];mappings=[]
        for h,idx in enumerate(indices):
            observed={'u':raw['u'][idx]};pilot,_=R.reconstruct(observed,[t],idx);fields=[pilot]
            rr=np.random.default_rng(np.random.SeedSequence([421997,seed,h]));mappings.append(O.interpolation(t,idx))
            for b in range(128):
                draw={'u':pilot['u'][idx]+sig*rr.standard_normal(len(idx))}
                fields.append(R.reconstruct(draw,[t],idx)[0])
            groups.append(fields)
        for g in range(3):
            freq,_=R.bandwidth({'u':raw['u'][indices[g]]},[t[indices[g]]]);plan=R.local_plan([t],terms,freq['loo'],5)
            bank=Q.Bank([t],terms,opaque,plan=plan);fits=J.E.local_fit(bank,groups[g],slice(None),[dict(c) for c in cs])
            active=np.flatnonzero(np.any(np.stack(bank.kernels[0])!=0,axis=(0,1)))
            reach=max(R.WF.leftover_orders(m,1)[0] for m in bank.monos)*R.WF.FD_REACH
            assert active.min()-reach>=max(idx[0] for idx in indices)
            assert active.max()+reach<=min(idx[-1] for idx in indices)
            smooth=[];direct=[]
            for h,idx in enumerate(indices):
                mm=np.stack([bank.matrix(v) for v in groups[h]]);smooth.append(2*mm[0]-mm[1:].mean(axis=0))
                cov=O.Covariance([t],mappings[h],{'u':(sig,[0.],[])})
                direct.append(O.matrix(bank,O.map_values(mappings[h],{'u':raw['u'][idx]}),cov))
            clean=bank.matrix({'u':u});left,right=[h for h in range(3) if h!=g]
            for method in ('continuous_ols','continuous_clsm_128'):
                for i,c in enumerate(cs):
                    coef,diag=fits[method][i];v=np.zeros(len(labels));v[c['yi']]=1.;v[c['xi']]-=coef
                    den=float(clean[:,c['yi']]@clean[:,c['yi']]);rr=clean@v
                    row=dict(noise=noise,seed=seed,training=g,candidate=c['id'],method=method,coefficients=coef,
                        intercept=float(coef[c['xi'].index(ci)]) if ci in c['xi'] else 0.,
                        clean_energy=float(rr@rr/den),smooth_cross=float((smooth[left]@v)@(smooth[right]@v)/den),
                        raw_cross=float((direct[left]@v)@(direct[right]@v)/den),fit=diag)
                    if c['exact'] is not None:
                        exact=v.copy();exact[c['xi']]=-np.array(c['exact']);er=clean@exact;row['exact_clean_energy']=float(er@er/den)
                    rows.append(row)
            folds.append(dict(noise=noise,seed=seed,training=g,plan=plan,quadrature=bank.metadata))
        print('ode_simple',noise,seed,'complete',flush=True)
    summary=[]
    for noise,seed in [(0,0),*((5,s) for s in range(3))]:
        for c in cs:
            for method in ('continuous_ols','continuous_clsm_128'):
                rr=[r for r in rows if r['noise']==noise and r['seed']==seed and r['candidate']==c['id'] and r['method']==method]
                summary.append(dict(noise=noise,seed=seed,candidate=c['id'],method=method,
                    mean_coefficients=np.mean([r['coefficients'] for r in rr],axis=0),
                    clean_energy=float(np.mean([r['clean_energy'] for r in rr])),
                    intercept_mean=float(np.mean([r['intercept'] for r in rr])),
                    intercept_std=float(np.std([r['intercept'] for r in rr])),
                    raw_cross=float(np.mean([r['raw_cross'] for r in rr])),smooth_cross=float(np.mean([r['smooth_cross'] for r in rr]))))
    source=R.REPO/'projects/pic/data/ode/ode.py'
    R.dump(O.ROOT/'ode_simple_constants.json',dict(dataset='ODE_simple_discovery',source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),experiment_sha256=hashlib.sha256(O.Path(__file__).read_bytes()).hexdigest(),
        exact_identity_errors=identities,candidates=cs,labels=labels,folds=folds,rows=rows,summary=summary,dyad=dyads,source_fd_ols=point_rows,
        noise_model='known iid Gaussian variance in a controlled additive-noise test; not estimated-noise validation',
        search_scope='first derivatives only; no second-derivative candidate. Decorated targets needed for energy/phase invariants. No evolutionary search.'))
    for filename,data in [('ode_simple_summary.csv',summary),('ode_simple_dyad.csv',dyads)]:
        with (O.ROOT/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print('identity checks',identities)
    for r in summary:
        if r['noise']==0 and r['method']=='continuous_clsm_128':print(r)


if __name__=='__main__':main()
