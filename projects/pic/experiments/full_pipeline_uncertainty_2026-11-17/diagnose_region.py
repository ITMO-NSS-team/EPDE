from bridge import *
import audit_full_fast as A
import mpmath as mp


def main():
    path=ROOT/'results/lorenz_iid_seed52.json';d=read(path);rows=[];mp.mp.dps=80
    with np.load(path.with_suffix('.npz')) as z:
        for mode in ('fixed_pilot','refit_pilot'):
            for bank in ('local','broad'):
                g=S.gram(z[bank+'_observed']);s=z[bank+'_anchor_covariance'];cut=d['cuts'][mode][bank];a=g-cut*s
                for c,r in zip(d['candidates'],d['regions'][mode][bank]):
                    try:A.Q.region_check(r,a,g,s,c,cut)
                    except AssertionError:
                        xi,yi=c['xi'],c['yi'];scale=np.array(r['scale']);h=a[np.ix_(xi,xi)]/scale[:,None]/scale[None,:];inverse=np.linalg.inv(h);center=np.array(r['center']);minimum=r['minimum'];condition=float(np.linalg.cond(h));endpoints=[]
                        hh=mp.matrix([[mp.mpf(float(a[i,j]))/mp.mpf(float(scale[k]))/mp.mpf(float(scale[l])) for l,j in enumerate(xi)] for k,i in enumerate(xi)])
                        bb=mp.matrix([mp.mpf(float(a[i,yi]))/mp.mpf(float(scale[k])) for k,i in enumerate(xi)]);hinv=hh**-1;cc=hinv*bb;mm=mp.mpf(float(a[yi,yi]))-(bb.T*cc)[0]
                        for j in range(len(xi)):
                            shift=np.sqrt(-minimum/inverse[j,j])*inverse[:,j]/scale
                            for sign,key in ((-1,'lower'),(1,'upper')):
                                beta=center+sign*shift;v=S.E.vector(c,beta,len(a));res=float(v@a@v);natural=float(abs(v)@(abs(g)+cut*abs(s))@abs(v));tol=2e-7*(abs(v@g@v)+abs(cut*v@s@v)+1e-300)+64*np.finfo(float).eps*natural
                                exact=(cc[j]+sign*mp.sqrt(-mm*hinv[j,j]))/mp.mpf(float(scale[j]))
                                exactcenter=cc[j]/mp.mpf(float(scale[j]));exactradius=mp.sqrt(-mm*hinv[j,j])/mp.mpf(float(scale[j]))
                                endpoints.append(dict(coordinate=j,side=key,residual=res,old_tolerance=tol,residual_over_tolerance=abs(res)/tol,
                                                      stored=r[key][j],exact=float(exact),relative_endpoint_error=float(abs(mp.mpf(float(r[key][j]))-exact)/(abs(exactcenter)+abs(exactradius)))))
                        rows.append(dict(mode=mode,bank=bank,candidate=c['id'],label=c['label'],condition=condition,
                                         stored_minimum=minimum,exact_minimum=float(mm),endpoints=endpoints))
    dump(ROOT/'region_diagnostic.json',dict(status='COMPLETE_DIAGNOSTIC',rows=rows,
         source_sha256=sources('diagnose_region.py'),input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))}))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
