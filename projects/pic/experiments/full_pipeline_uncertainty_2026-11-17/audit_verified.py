"""Retain ordinary replay checks; independently verify ill-conditioned ellipsoids.

The historical audit mixes a production eigensolve center/minimum with an LU
inverse to construct boundary points. Under high conditioning those ingredients
can describe slightly different quadrics. Verify the original stored matrix at
80 digits when that check fails; preserve the old failed record separately.
"""
from bridge import *
import audit_full_fast as Legacy
import mpmath as mp

original=Legacy.Q.region_check
proofs=[]


def verified(reg,a,g,s,c,cut):
    try:
        return original(reg,a,g,s,c,cut)
    except AssertionError:
        if reg['status']!='ellipsoid':raise
    mp.mp.dps=80
    xi,yi=c['xi'],c['yi'];scale=np.array(reg['scale']);p=len(xi)
    h=a[np.ix_(xi,xi)]/scale[:,None]/scale[None,:];h=(h+h.T)/2
    condition=float(np.linalg.cond(h));bound=64*np.finfo(float).eps*condition+3e-10
    assert bound<.02
    hh=mp.matrix([[(mp.mpf(float(a[i,j]))+mp.mpf(float(a[j,i])))/2/mp.mpf(float(scale[k]))/mp.mpf(float(scale[l])) for l,j in enumerate(xi)] for k,i in enumerate(xi)])
    bb=mp.matrix([(mp.mpf(float(a[i,yi]))+mp.mpf(float(a[yi,i])))/2/mp.mpf(float(scale[k])) for k,i in enumerate(xi)])
    mp.cholesky(hh)  # Independent positive-definiteness check.
    inv=hh**-1;center=inv*bb;yy=mp.mpf(float(a[yi,yi]));minimum=yy-(bb.T*center)[0]
    assert minimum<0
    minimum_error=abs(mp.mpf(float(reg['minimum']))-minimum)
    assert minimum_error<=mp.mpf(bound)*(abs(yy)+abs((bb.T*center)[0]))
    errors=[];residuals=[]
    for j in range(p):
        sc=mp.mpf(float(scale[j]));cn=center[j]/sc;radius=mp.sqrt(-minimum*inv[j,j])/sc;natural=abs(cn)+radius
        assert abs(mp.mpf(float(reg['center'][j]))-cn)<=mp.mpf(bound)*natural
        for sign,key in ((-1,'lower'),(1,'upper')):
            exact=cn+sign*radius;error=abs(mp.mpf(float(reg[key][j]))-exact)/natural;assert error<=bound
            beta=center+sign*mp.sqrt(-minimum/inv[j,j])*inv[:,j]
            residual=(beta.T*hh*beta)[0]-2*(bb.T*beta)[0]+yy
            norm=abs((beta.T*hh*beta)[0])+2*abs((bb.T*beta)[0])+abs(yy)+1
            assert abs(residual)<=mp.mpf('1e-60')*norm
            errors.append(float(error));residuals.append(float(abs(residual)/norm))
    np.testing.assert_allclose(reg['witness'],reg['center'],rtol=3e-8,atol=3e-10)
    v=S.E.vector(c,reg['witness'],len(a));assert v@a@v<=1e-8*(abs(v@g@v)+abs(cut*v@s@v)+1e-300)
    proofs.append(dict(candidate=c['id'],label=c['label'],cutoff=cut,condition=condition,
                       allowed_forward_error=bound,max_endpoint_relative_error=max(errors),
                       max_high_precision_boundary_residual=max(residuals),exact_minimum=float(minimum)))


Legacy.Q.region_check=verified


def safe(path):
    proofs.clear();result=Legacy.safe(path);result['high_precision_region_checks']=list(proofs)
    dump(Path(path).parent.parent/'audit_records'/Path(path).name,result)
    return result
