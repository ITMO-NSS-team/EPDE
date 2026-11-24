import fit as Base
np=Base.np
from scipy.linalg import cholesky_banded,cho_solve_banded

def selected_inverse(chol):
    width,n=chol.shape;width-=1;inv=np.zeros_like(chol)
    for i in range(n-1,-1,-1):
        k=min(width,n-1-i)
        if k:
            idx=np.arange(1,k+1);block=inv[np.abs(idx[:,None]-idx[None,:]),i+np.minimum(idx[:,None],idx[None,:])]
            weights=chol[1:k+1,i]/chol[0,i];off=-block@weights;inv[1:k+1,i]=off;inv[0,i]=1/chol[0,i]**2-weights@off
        else:inv[0,i]=1/chol[0,i]**2
    return inv

def band_inner(a,b):
    return float(np.sum(a[0]*b[0])+2*np.sum(a[1:]*b[1:]))

def profile_score(cc,rhos,kappas,p1s,p2s,covariance_band):
    dim=len(cc.fields);qsum=0.;ldsum=0.;dq=np.zeros(4*dim);dld=np.zeros(4*dim)
    for axis,(d,r,k,p1,p2) in enumerate(zip(cc.fields,rhos,kappas,p1s,p2s)):
        band=covariance_band(len(d),r,k,p1,p2);chol=cholesky_banded(band,lower=True,check_finite=False)
        z=d.copy();z[1:]-=r*d[:-1];filtered=z.copy();filtered[1:]-=p1*(1-p2)*z[:-1];filtered[2:]-=p2*z[:-2]
        sol=cho_solve_banded((chol,True),filtered,check_finite=False);qsum+=float(np.sum(filtered*sol));ldsum+=d.shape[1]*2*float(np.log(chol[0]).sum())
        inv=selected_inverse(chol);gram=np.zeros_like(band)
        gram[0]=np.sum(sol*sol,axis=1)
        for lag in range(1,5):gram[lag,:len(d)-lag]=np.sum(sol[lag:]*sol[:-lag],axis=1)
        dr=np.zeros_like(d);dr[1:]=-d[:-1];dzr=dr.copy();dzr[1:]-=p1*(1-p2)*dr[:-1];dzr[2:]-=p2*dr[:-2]
        dzp1=np.zeros_like(d);dzp1[1:]=-(1-p2)*z[:-1]
        dzp2=np.zeros_like(d);dzp2[1:]=p1*z[:-1];dzp2[2:]-=z[:-2]
        pars=[r,k,p1,p2]
        for j,dz in enumerate((dzr,None,dzp1,dzp2)):
            complex_pars=list(pars);complex_pars[j]+=1e-30j;deriv=np.imag(covariance_band(len(d),*complex_pars))/1e-30
            if j==1:deriv*=k
            dq[axis+j*dim]=(0. if dz is None else 2*float(np.sum(sol*dz)))-band_inner(gram,deriv)
            dld[axis+j*dim]=d.shape[1]*band_inner(inv,deriv)
    if qsum<=0 or not np.isfinite(qsum):raise FloatingPointError('Invalid profile quadratic')
    return float(np.log(qsum/cc.count)+ldsum/cc.count),dq/qsum+dld/cc.count
