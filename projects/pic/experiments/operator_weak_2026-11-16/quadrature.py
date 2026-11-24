"""Continuous weak integrals of locally interpolated monomial fields.

Integrate the compact polynomial test analytically differentiated, against a
piecewise local polynomial interpolant. Gauss quadrature is exact for DC
tests and is checked by order refinement for modulated tests. No PDE solves.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'weak_fitting_2026-11-16'),str(ROOT.parent/'core15_residual_products_2026-11-16')]
import run_fitting as F
R,A,np=F.R,F.A,F.np
from scipy.special import beta
from math import comb, factorial
import time


def _one_window(axis,cut,center,m,q,freq,gmax,degree,ng):
    h=float(axis[1]-axis[0]);length=m*h
    cells=np.arange(center-m,center+m)
    starts=np.clip(cells-degree//2,cut,len(axis)-cut-degree-1)
    nodes,weights=np.polynomial.legendre.leggauss(ng)
    fraction=(nodes+1)/2
    theta=cells[:,None]-starts[:,None]+fraction[None,:]
    lagrange=np.ones(theta.shape+(degree+1,))
    for j in range(degree+1):
        for k in range(degree+1):
            if j!=k:lagrange[:,:,j]*=(theta-k)/(j-k)
    s=(cells[:,None]-center+fraction[None,:])/m
    phi=np.polynomial.polynomial.polypow([1.,0.,-1.],q)
    derivs=[np.polynomial.polynomial.polyval(s,np.polynomial.polynomial.polyder(phi,g))/length**g
            for g in range(gmax+1)]
    phase=np.exp(1j*freq[:,None,None]*length*s[None,:,:])
    mass=length*beta(.5,q+1)
    indices=(starts[:,None]+np.arange(degree+1)[None,:]).reshape(-1)
    result=[]
    for g in range(gmax+1):
        kernel=phase*sum(comb(g,j)*(1j*freq[:,None,None])**(g-j)*derivs[j][None,:,:]
                         for j in range(g+1))
        kernel*=h*weights[None,None,:]/(2*mass)
        contribution=np.einsum('fck,ckj->fcj',kernel,lagrange,optimize=True).reshape(len(freq),-1)
        out=np.zeros((len(freq),len(axis)),complex)
        for i in range(len(freq)):np.add.at(out[i],indices,contribution[i])
        result.append(out)
    return result


def axis_weights(axis,cut,centers,m,q,freq,gmax,degree):
    degree=min(degree,len(axis)-2*cut-1)
    if degree<gmax:raise ValueError('Interpolation cannot represent the required derivative order')
    ng=q+(degree+2)//2
    weights=[];largest=0.;orders=[]
    for center in centers:
        previous=_one_window(axis,cut,int(center),m,q,freq,gmax,degree,ng)
        if np.any(freq):
            for extra in (2,4,8,16,32):
                current=_one_window(axis,cut,int(center),m,q,freq,gmax,degree,ng+extra)
                error=max(float(np.max(np.sum(np.abs(a-b),axis=1)))/
                          max(float(np.max(np.sum(np.abs(b),axis=1))),np.finfo(float).tiny)
                          for a,b in zip(previous,current))
                if error<=1e-11:break # numerical quadrature tolerance, not a statistical threshold
                previous=current
            else:raise RuntimeError(f'Quadrature refinement did not converge: {error}')
            largest=max(largest,error);orders.append(ng+extra);previous=current
        else:orders.append(ng)
        weights.append(previous)
    result=[np.concatenate([v[g] for v in weights],axis=0) for g in range(gmax+1)]
    if not np.any(freq):result=[v.real for v in result]
    return result,dict(degree=degree,gauss_orders=orders,refinement_relative_error=largest)


class Bank:
    def __init__(self,axes,terms,opaque,plan=None,frequencies=None):
        self.axes,self.terms,self.opaque=axes,terms,opaque
        if plan is None:
            q,self.monos,cut=R.common_geometry(axes,terms)
            ms=tuple((len(a)-2*c-1)//2 for a,c in zip(axes,cut))
            centers=[np.array([c+m]) for c,m in zip(cut,ms)]
        else:
            q,cut,ms=plan['q'],plan['cut'],plan['ms']
            centers=[c+m+np.arange(n)*s for c,m,n,s in zip(cut,ms,plan['shape'],plan['ss'])]
            self.monos=[t.monomials(psi_cap=q) for t in terms]
        frequencies=frequencies or [np.array([0.]) for _ in axes]
        self.kernels=[];checks=[]
        for axis,(a,c,ct,m,qs,f) in enumerate(zip(axes,cut,centers,ms,q,frequencies)):
            order=max(key[2][axis] for mono in self.monos for key in mono)
            weights,check=axis_weights(a,c,ct,m,qs,np.asarray(f),order,3 if axis==0 else 5)
            self.kernels.append(weights);checks.append(check)
        self.shape=tuple(len(c)*len(f) for c,f in zip(centers,frequencies))
        self.metadata=dict(shape=self.shape,checks=checks,ms=ms,q=q,cut=cut,
                           rows=int(np.prod(self.shape)))
    def matrix(self,values):
        fields=R.WF.Fields(values,self.axes,self.opaque)
        cache={};integrals={};columns=[]
        for term,monos in zip(self.terms,self.monos):
            column=0.
            for (data,coords,orders),factor in monos.items():
                key=(data,coords,term.opaque)
                if key not in cache:cache[key]=fields.point(data,coords,term.opaque)
                ikey=(key,orders)
                if ikey not in integrals:
                    arr=cache[key]
                    for axis in sorted(range(len(self.axes)),key=lambda a:self.shape[a]/arr.shape[a]):
                        arr=np.moveaxis(np.tensordot(self.kernels[axis][orders[axis]],arr,axes=(1,axis)),0,axis)
                    integrals[ikey]=arr
                column=column+factor*integrals[ikey]
            columns.append(np.asarray(column).reshape(-1))
        matrix=np.column_stack(columns)
        assert np.isfinite(matrix).all()
        return matrix


def mapped_plan(old,group,plan,axes,opaque):
    idx=np.array(old['groups'][group]['indices'])
    segment=slice(idx[0],idx[-1]+1)
    axes=[axes[0][segment],*axes[1:]]
    opq={k:v[segment] for k,v in opaque.items()}
    fine={k:((v[0]*3,)+tuple(v[1:]) if k in ('ms','ss','cut') else tuple(v)) for k,v in plan.items()}
    return segment,axes,opq,fine


def checks():
    t=np.linspace(-2,2,43);maximum=0.
    for q,gmax,degree in [(3,2,3),(5,4,5)]:
        for f in [np.array([0.]),np.array([0.,2.,-2.,6.,-6.])]:
            ks,meta=axis_weights(t,0,[21],18,q,f,gmax,degree)
            for p in range(degree+1):
                for g in range(gmax+1):
                    left=ks[g]@t**p
                    right=(-1)**g*(ks[0]@(factorial(p)/factorial(p-g)*t**(p-g))) if p>=g else np.zeros_like(left)
                    error=float(np.max(np.abs(left-right)))/max(1.,float(np.max(np.abs(right))))
                    maximum=max(maximum,error)
                    np.testing.assert_allclose(left,right,atol=2e-9,rtol=2e-9)
    R.dump(ROOT/'quadrature_checks.json',dict(max_polynomial_IBP_error=maximum))
    print('Polynomial and modulated IBP check:',maximum,flush=True)


if __name__=='__main__':checks()
