"""Independent checks of corrected moments and weak regression scaling."""
import run_fitting as F
R,np=F.R,F.np
from itertools import product


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    # Enumerate the full independent distribution, avoiding MC tolerances.
    signal=np.array([[1.,2.,3.],[-2.,1.,.5],[.2,.7,1.4]])
    a=np.array([[.5,1.,-.2],[.2,0.,.3],[.8,-.1,.2]])
    b=np.array([[.3,-.2,.1],[-.5,.7,.1],[.2,.5,-.3]])
    noise=[i*a+j*b for i,j in product([-1.,1.],repeat=2)]
    average=np.zeros((3,3));ordinary=np.zeros((3,3));n=0
    for e,ea,eb in product(noise,repeat=3):
        observed=signal+e;draws=np.array([signal+ea,signal+eb]);mean=draws.mean(axis=0)
        covariance=sum((d-mean).T@(d-mean) for d in draws)
        shifted=observed-(mean-signal)
        corrected=shifted.T@shifted-1.5*covariance
        average+=corrected;ordinary+=observed.T@observed;n+=1
    average/=n;ordinary/=n
    np.testing.assert_allclose(average,signal.T@signal,atol=2e-14)
    assert np.linalg.norm(ordinary-signal.T@signal)>.1
    # Recover a known linear relation with highly disparate column scales.
    x=np.array([[1.,2.],[-1.,3.],[3.,1.],[2.,-2.]])*np.array([1e-3,1e3])
    truth=np.array([2.,-.4]);y=x@truth;mat=np.column_stack([x,y])
    coef,diag=R.signed_fit(mat.T@mat,mat,[0,1],2)
    np.testing.assert_allclose(coef,truth,rtol=1e-7,atol=1e-7)
    out=dict(enumerated_moment_max_error=float(np.max(np.abs(average-signal.T@signal))),
             scaled_linear_fit_max_error=float(np.max(np.abs(coef-truth))),rank=diag['rank'])
    R.dump(F.ROOT/'numerical_checks.json',out);print(out)


if __name__=='__main__':main()
