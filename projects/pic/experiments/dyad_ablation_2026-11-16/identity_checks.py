import criteria as K
R,np,ROOT=K.R,K.np,K.ROOT
import ode_simple_constants as D
from types import SimpleNamespace


def main():
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t);du=np.cos(t)-1.3*np.sin(t)
    assert np.max(np.abs(u*u+du*du-2.69))<3e-15
    assert np.max(np.abs(u*np.sin(t)+du*np.cos(t)-1))<3e-15
    assert np.max(np.abs(u*np.cos(t)-du*np.sin(t)-1.3))<3e-15
    p=SimpleNamespace(shape=u.shape,data={'u':u},grids=[t],named_arrays={});cs,labels=D.panel();terms,opaque=R.symbols(p,labels)
    width=max(len(c['xi']) for c in cs);K.common_width(width);rows=[]
    for noise,seed in [(0,0),(5,0),(5,1),(5,2)]:
        raw={'u':u+.01*noise*np.std(u)*np.random.default_rng(seed).standard_normal(len(t))}
        frequencies,_=R.bandwidth(raw,[t]);plan=R.local_plan([t],terms,frequencies['loo'],width)
        matrix=R.LocalBank([t],terms,opaque,plan).matrix(raw);scores,_=K.evaluate(matrix,cs,plan['shape'])
        for j,c in enumerate(cs):rows.append(dict(noise=noise,seed=seed,candidate=c['id'],constant_present=labels.index(()) in c['xi'],
                                                  **{k:v[j] for k,v in scores.items()}))
    for r in rows:
        if r['candidate'] in ('energy','phase_sin','phase_cos'):assert r['constant_present']
    R.dump(ROOT/'identity_checks.json',dict(analytic_identities_exact=True,no_constant_removed=True,rows=rows,
        source_sha256={str(p):K.S.sha(p) for p in (K.__file__,__file__,D.__file__)}))
    print('ODE_simple phase/energy identities and explicit constants preserved.')


if __name__=='__main__':main()
