from common import np,R
from scipy.linalg import cholesky,solve_triangular,eigh
from scipy.fft import dct

def axis_basis(axis,rho,time_axis):
    n=len(axis)
    if time_axis:
        d,u,_=R.spline_basis(tuple(axis),tuple(range(n)));null=2
    else:
        d=(np.arange(n)/n)**4;u=dct(np.eye(n),axis=0,norm='ortho').T;null=1
    if rho==0:return d,u,u.T
    k=(u*d)@u.T;lag=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);l=cholesky(rho**lag,lower=True,check_finite=False);b=l.T@k@l;d,u=eigh((b+b.T)/2,check_finite=False);d[:null]=0.
    assert np.all(d[null:]>0)
    forward=solve_triangular(l.T,u,lower=False,check_finite=False).T
    return d,l@u,forward

def reconstruct(raw,axes,models):
    result={};info={}
    for v,y in raw.items():
        signal=np.asarray(y).copy();entries=[]
        for axis,x in enumerate(axes):
            d,back,forward=axis_basis(x,models[v][1][axis],axis==0);flat=np.moveaxis(signal,axis,0).reshape(len(x),-1);z=forward@flat;f,entry=R.pooled_gcv_filter(np.mean(z*z,axis=1),d);smooth=back@(f[:,None]*z);signal=np.moveaxis(smooth.reshape(np.moveaxis(signal,axis,0).shape),0,axis);entries.append(entry)
        result[v]=signal;info[v]=entries
    return result,info
