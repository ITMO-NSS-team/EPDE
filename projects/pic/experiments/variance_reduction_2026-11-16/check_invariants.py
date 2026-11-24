"""Check that new weighted broad moments preserve ODE_simple's constants."""
import moments as M
O,R,Q,np=M.O,M.R,M.Q,M.np
import ode_simple_constants as D
from types import SimpleNamespace
import hashlib


def main():
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t);sig=.05*np.std(u)
    rng=np.random.default_rng(0);raw=u+sig*rng.standard_normal(len(u))
    p=SimpleNamespace(shape=(200,),data={'u':u},grids=[t],named_arrays={})
    cs,labels=D.panel();terms,opaque=R.symbols(p,labels);ci=labels.index(())
    idx=np.arange(0,200,3);segment=slice(2,198);ax=[t[segment]];opq={k:v[segment] for k,v in opaque.items()}
    pilot,_=R.reconstruct({'u':raw[idx]},[t],idx);tm=O.interpolation(t,idx)[segment]
    cov=O.Covariance(ax,tm,{'u':(sig,[0.],[])})
    _,selection=R.bandwidth({'u':raw[idx]},[t[idx]]);kmax=selection[0]['loo']['max_mode'];rows=[]
    for extra in (0,1):
        bank,meta=M.B.identify_bank(ax,terms,opq,{'u':pilot['u'][segment]},cs,kmax,extra)
        assert bank is not None;basis=M.RealBasis(meta['k'],1)
        pp=basis.convert(bank.matrix({'u':pilot['u'][segment]}));nr=np.random.default_rng(59137);draws=[]
        for _ in range(128):
            fields={'u':pilot['u'][idx]+sig*nr.standard_normal(len(idx))}
            draws.append(O.matrix(bank,O.map_values(tm,fields),cov))
        w,wm=M.whitening(basis.convert(np.stack(draws)),pp);assert w is not None
        # Inspect clean values only after the noisy-data weighting is frozen.
        clean=basis.convert(bank.matrix({'u':u[segment]}))
        for method,weight in [('plain',None),('white128',w)]:
            mat=M.transform(clean,weight)
            for c in cs:
                if c['id'] not in ('phase_sin','phase_cos','energy'):continue
                v=np.zeros(len(labels));v[c['yi']]=1.;v[c['xi']]-=c['exact'];without=v.copy();without[ci]=0.
                den=float(mat[:,c['yi']]@mat[:,c['yi']]);r=mat@v;rr=mat@without
                row=dict(q=extra,method=method,identity=c['id'],constant=c['exact'][c['xi'].index(ci)],
                    clean_energy=float(r@r/den),constant_deleted_energy=float(rr@rr/den),
                    constant_column_norm=float(np.linalg.norm(mat[:,ci])),k=meta['k'])
                assert row['clean_energy']<1e-7 and row['constant_deleted_energy']>row['clean_energy']
                rows.append(row)
    R.dump(M.ROOT/'invariant_checks.json',dict(rows=rows,phase_derivative_power=1,energy_derivative_power=2,
        maximum_derivative_order=1,all_constants_preserved=True,
        source_sha256=hashlib.sha256(O.Path(__file__).read_bytes()).hexdigest()))
    for row in rows:print(row)


if __name__=='__main__':main()
