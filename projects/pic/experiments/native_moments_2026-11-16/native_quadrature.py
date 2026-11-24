"""Physical weak windows on native grids, including partial boundary cells."""
import os, sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'eiv_fitting_2026-11-16'))
import run_eiv as V
E,M,R,Q,O,F,np=V.E,V.M,V.R,V.Q,V.O,V.F,V.np
from scipy.special import beta
from math import comb, factorial


def one_window(axis, cut, center, radius, q, frequency, order, degree, count):
    lo,hi=center-radius,center+radius
    tol=1e-11*max(1.,abs(lo),abs(hi))
    assert lo>=axis[cut]-tol and hi<=axis[len(axis)-1-cut]+tol
    cells=np.flatnonzero((axis[:-1]<hi-tol*.01)&(axis[1:]>lo+tol*.01))
    left=np.maximum(axis[cells],lo);right=np.minimum(axis[cells+1],hi)
    assert np.all(right>left)
    starts=np.clip(cells-degree//2,cut,len(axis)-cut-degree-1)
    nodes,weights=np.polynomial.legendre.leggauss(count)
    xx=(left+right)[:,None]/2+(right-left)[:,None]*nodes[None,:]/2
    indices=starts[:,None]+np.arange(degree+1)[None,:]
    interpolation=np.ones(xx.shape+(degree+1,))
    knots=axis[indices]
    for j in range(degree+1):
        for k in range(degree+1):
            if j!=k: interpolation[:,:,j]*=(xx-knots[:,k,None])/(knots[:,j,None]-knots[:,k,None])
    s=(xx-center)/radius;poly=np.polynomial.polynomial.polypow([1.,0.,-1.],q)
    derivatives=[np.polynomial.polynomial.polyval(s,np.polynomial.polynomial.polyder(poly,g))/radius**g
                 for g in range(order+1)]
    phase=np.exp(1j*frequency[:,None,None]*(xx-center)[None,:,:]);mass=radius*beta(.5,q+1)
    out=[]
    for g in range(order+1):
        kernel=phase*sum(comb(g,j)*(1j*frequency[:,None,None])**(g-j)*derivatives[j][None,:,:] for j in range(g+1))
        kernel*=((right-left)[:,None]*weights[None,:]/(2*mass))[None,:,:]
        values=np.einsum('fck,ckj->fcj',kernel,interpolation,optimize=True).reshape(len(frequency),-1)
        matrix=np.zeros((len(frequency),len(axis)),complex)
        for i in range(len(frequency)): np.add.at(matrix[i],indices.ravel(),values[i])
        out.append(matrix)
    return out


def axis_weights(axis,cut,centers,radius,q,frequency,order,degree):
    degree=min(degree,len(axis)-2*cut-1)
    if degree<order: raise ValueError('Insufficient native nodes for derivative quadrature')
    count=q+(degree+2)//2;all_weights=[];errors=[]
    for center in centers:
        before=one_window(axis,cut,center,radius,q,frequency,order,degree,count)
        error=0.
        if np.any(frequency):
            for extra in (2,4,8,16,32):
                after=one_window(axis,cut,center,radius,q,frequency,order,degree,count+extra)
                error=max(float(np.max(np.sum(np.abs(a-b),axis=1)))/max(float(np.max(np.sum(np.abs(a),axis=1))),np.finfo(float).tiny)
                          for a,b in zip(after,before))
                before=after
                if error<=1e-11: break
            else: raise ValueError('Native modulated quadrature did not converge')
        all_weights.append(before);errors.append(error)
    result=[np.concatenate([a[g] for a in all_weights],axis=0) for g in range(order+1)]
    if not np.any(frequency):result=[a.real for a in result]
    return result,dict(degree=degree,refinement_error=max(errors),rows=len(result[0]))


class Bank:
    matrix=Q.Bank.matrix

    def __init__(self,axes,terms,opaque,spec):
        self.axes,self.terms,self.opaque=axes,terms,opaque
        self.monos=[t.monomials(psi_cap=spec['q']) for t in terms]
        cut=np.max([R.WF.leftover_orders(m,len(axes)) for m in self.monos],axis=0).astype(int)*R.WF.FD_REACH
        cut=np.maximum(cut,np.asarray(spec.get('interpolation_cut',cut)))
        self.kernels=[];checks=[]
        for i,axis in enumerate(axes):
            order=max(key[2][i] for mono in self.monos for key in mono)
            weights,check=axis_weights(axis,int(cut[i]),np.asarray(spec['centers'][i]),spec['radii'][i],spec['q'][i],
                                      np.asarray(spec['frequencies'][i]),order,3 if i==0 else 5)
            self.kernels.append(weights);checks.append(check)
        self.shape=tuple(len(c)*len(f) for c,f in zip(spec['centers'],spec['frequencies']))
        self.metadata=dict(spec=spec,cut=cut,shape=self.shape,checks=checks)


def spec_from_plan(axes,plan,k=0):
    signed=np.r_[np.arange(k+1),-np.arange(1,k+1)]
    radii=[float(m*(a[1]-a[0])) for m,a in zip(plan['ms'],axes)]
    centers=[(a[0]+(c+m+np.arange(n)*s)*(a[1]-a[0])).tolist()
             for a,c,m,n,s in zip(axes,plan['cut'],plan['ms'],plan['shape'],plan['ss'])]
    return dict(q=list(plan['q']),radii=radii,centers=centers,interpolation_cut=list(plan['cut']),
                frequencies=[(signed*np.pi/r).tolist() for r in radii])


def common_spec(group_axes,terms,q,radii=None,k=0):
    dim=len(group_axes[0]);monos=[t.monomials(psi_cap=q) for t in terms]
    cut=np.max([R.WF.leftover_orders(m,dim) for m in monos],axis=0).astype(int)*R.WF.FD_REACH
    result=dict(q=list(q),centers=[],radii=[],frequencies=[])
    signed=np.r_[np.arange(k+1),-np.arange(1,k+1)]
    for i,c in enumerate(cut):
        lo=max(ax[i][c] for ax in group_axes);hi=min(ax[i][-1-c] for ax in group_axes)
        radius=(hi-lo)/2 if radii is None else min(radii[i],(hi-lo)/2)
        assert radius>0
        count=1 if radii is None else int(np.floor((hi-lo-2*radius)/radius+1e-9))+1
        centers=lo+radius+np.arange(count)*radius
        result['centers'].append(centers.tolist());result['radii'].append(float(radius))
        result['frequencies'].append((signed*np.pi/radius).tolist())
    return result


def checks():
    t=np.linspace(-2,2,43);maximum=0.;aligned=0.
    for q,order,degree in [(3,2,3),(5,4,5)]:
        for freq in (np.array([0.]),np.array([0.,2.,-2.,6.,-6.])):
            center=t[21];radius=18*(t[1]-t[0])
            new,_=axis_weights(t,0,[center],radius,q,freq,order,degree)
            old,_=Q.axis_weights(t,0,[21],18,q,freq,order,degree)
            for a,b in zip(new,old):
                np.testing.assert_allclose(a,b,atol=5e-11,rtol=5e-10)
                aligned=max(aligned,float(np.max(np.abs(a-b))))
            weights,_=axis_weights(t,0,[.073],1.72,q,freq,order,degree)
            for power in range(degree+1):
                for g in range(order+1):
                    left=weights[g]@t**power
                    right=(-1)**g*(weights[0]@(factorial(power)/factorial(power-g)*t**(power-g))) if power>=g else np.zeros_like(left)
                    err=float(np.max(np.abs(left-right)))/max(1.,float(np.max(np.abs(right))))
                    maximum=max(maximum,err)
                    np.testing.assert_allclose(left,right,atol=2e-9,rtol=2e-9)
    # Identical physical polynomial moment on shifted, differently sampled grids.
    a=np.linspace(0,1,35);b=np.linspace(.01,.99,27)
    wa,_=axis_weights(a,0,[.5],.4,3,np.array([0.]),2,3)
    wb,_=axis_weights(b,0,[.5],.4,3,np.array([0.]),2,3)
    for power in range(4):
        for g in range(3):np.testing.assert_allclose(wa[g]@a**power,wb[g]@b**power,atol=2e-12)
    np.testing.assert_allclose(wa[0]@np.ones(len(a)),1.,atol=2e-14)
    rejected=False
    try:axis_weights(a,2,[.1],.2,3,np.array([0.]),2,3)
    except AssertionError: rejected=True
    assert rejected
    R.dump(ROOT/'checks.json',dict(aligned_weights_max_error=aligned,off_grid_polynomial_ibp_error=maximum,
           shifted_grid_polynomial_agreement=True,constant_preserved=True,unsafe_support_rejected=True))


if __name__=='__main__': checks()
