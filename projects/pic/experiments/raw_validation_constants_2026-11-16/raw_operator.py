"""Fixed interpolation and Gaussian contractions for degree <=3 weak terms."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'joint_prediction_risk_2026-11-16'))
import run_joint as J
R,Q,F,A,np=J.R,J.Q,J.F,J.A,J.np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix,eye
from itertools import combinations,product


def interpolation(full,idx):
    knots=full[idx]
    # Outside values are filled only to keep zero-weight tensor entries finite.
    # Callers assert that the weak integrands use the common interior domain.
    return CubicSpline(knots,np.eye(len(idx)),bc_type='natural')(np.clip(full,knots[0],knots[-1]))


def map_values(matrix,values):
    return {v:np.tensordot(matrix,y,axes=(1,0)) for v,y in values.items()}


class Covariance:
    def __init__(self,axes,time_map,model):
        self.axes=axes;self.time_map=time_map;self.model=model
        self.maps={};self.axis_cov={};self.cache={}

    def linear_map(self,axis,order):
        key=axis,order
        if key not in self.maps:
            if axis==0:
                m=self.time_map.copy()
                for _ in range(order):m=R.WF._central4(m,self.axes[0],0)
            elif order==0:m=eye(len(self.axes[axis]),format='csr')
            else:
                d=csr_matrix(R.WF._central4(np.eye(len(self.axes[axis])),self.axes[axis],0))
                m=d
                for _ in range(order-1):m=d@m
            self.maps[key]=m
        return self.maps[key]

    def pair(self,left,right):
        if left[0]!=right[0]:return 0.
        key=tuple(sorted((left,right)))
        if key in self.cache:return self.cache[key]
        sig,rhos,*_=self.model[left[0]];value=sig**2
        for axis,(oa,ob,rho) in enumerate(zip(left[1],right[1],rhos)):
            ak=axis,min(oa,ob),max(oa,ob),float(rho)
            if ak not in self.axis_cov:
                aa=self.linear_map(axis,oa);bb=self.linear_map(axis,ob)
                n=aa.shape[1];cov=float(rho)**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
                if axis==0:diag=np.sum((aa@cov)*bb,axis=1)
                else:diag=np.asarray(bb.multiply(aa@cov).sum(axis=1)).ravel()
                self.axis_cov[ak]=diag
            shape=[1]*len(self.axes);shape[axis]=len(self.axes[axis])
            value=value*self.axis_cov[ak].reshape(shape)
        self.cache[key]=value
        return value


def integrate(bank,arr,orders):
    for axis in sorted(range(len(bank.axes)),key=lambda a:bank.shape[a]/arr.shape[a]):
        arr=np.moveaxis(np.tensordot(bank.kernels[axis][orders[axis]],arr,axes=(1,axis)),0,axis)
    return arr


def matrix(bank,values,cov=None):
    fields=R.WF.Fields(values,bank.axes,bank.opaque);points={};integrals={};columns=[]
    for term,monos in zip(bank.terms,bank.monos):
        column=0.
        for (data,coords,orders),factor in monos.items():
            assert len(data)<=3
            key=data,coords,term.opaque
            if key not in points:
                arr=fields.point(data,coords,term.opaque)
                if cov is not None and len(data)>=2:
                    deterministic=fields.point((),coords,term.opaque)
                    for i,j in combinations(range(len(data)),2):
                        correction=cov.pair(data[i],data[j])
                        for k in range(len(data)):
                            if k not in (i,j):correction=correction*fields.get(*data[k])
                        arr=arr-deterministic*correction
                points[key]=arr
            ik=key,orders
            if ik not in integrals:integrals[ik]=integrate(bank,points[key],orders)
            column=column+factor*integrals[ik]
        columns.append(np.asarray(column).reshape(-1))
    out=np.column_stack(columns);assert np.isfinite(out).all();return out


def checks():
    # Gauss-Hermite integrates the relevant polynomial moments exactly.
    nodes,weights=np.polynomial.hermite.hermgauss(4);weights=weights/np.sqrt(np.pi)
    mu=np.array([.7,-1.2,.3]);cc=np.array([[.3,.04,-.02],[.04,.2,.01],[-.02,.01,.1]])
    chol=np.linalg.cholesky(cc);means=np.zeros(2)
    for ids in product(range(4),repeat=3):
        w=np.prod(weights[list(ids)]);y=mu+chol@(np.sqrt(2)*nodes[list(ids)])
        means+=w*np.array([y[0]*y[1]-cc[0,1],y.prod()-cc[0,1]*y[2]-cc[0,2]*y[1]-cc[1,2]*y[0]])
    np.testing.assert_allclose(means,[mu[0]*mu[1],mu.prod()],atol=2e-15)
    axes=[np.linspace(0,1,13),np.linspace(-1,1,7)];idx=np.arange(0,13,3);tm=interpolation(axes[0],idx)
    model={'u':(.2,[.3,-.2],[])};cov=Covariance(axes,tm,model)
    ta=tm;tb=R.WF._central4(tm,axes[0],0);sa=np.eye(7);sb=R.WF._central4(sa,axes[1],0)
    ct=.3**np.abs(np.arange(len(idx))[:,None]-np.arange(len(idx))[None,:]);cx=(-.2)**np.abs(np.arange(7)[:,None]-np.arange(7)[None,:])
    dense=np.kron(ta,sa)@(.2**2*np.kron(ct,cx))@np.kron(tb,sb).T
    expected=np.diag(dense).reshape(13,7)
    np.testing.assert_allclose(cov.pair(('u',(0,0)),('u',(1,1))),expected,atol=2e-15)
    terms=[R.WF.WeakTerm([], (0,0)),R.WF.WeakTerm([('u',(0,0))],(0,0)),
           R.WF.WeakTerm([('u',(1,0))],(0,0)),R.WF.WeakTerm([('u',(0,0))]*3,(0,0))]
    plan=dict(q=(3,3),cut=(0,0),ms=(4,3),ss=(4,3),shape=(1,1));bank=Q.Bank(axes,terms,{},plan=plan)
    values={'u':np.full((13,7),2.)};plain=bank.matrix(values);corrected=matrix(bank,values,cov)
    np.testing.assert_allclose(matrix(bank,values),plain,atol=2e-13)
    np.testing.assert_allclose(plain[0,:3],[1.,2.,0.],atol=2e-13)
    assert corrected[0,0]==plain[0,0] and corrected[0,1]==plain[0,1]
    # Two independent signs: cross products have no squared-noise floor.
    cross=np.mean([(2+.4*a)*(3+.7*b) for a,b in product([-1,1],repeat=2)])
    np.testing.assert_allclose(cross,6.,atol=1e-15)
    R.dump(ROOT/'operator_checks.json',dict(gaussian_contractions=means,expected=[mu[0]*mu[1],mu.prod()],
        covariance_dense_max_error=float(np.max(np.abs(cov.pair(('u',(0,0)),('u',(1,1)))-expected))),
        constant_column=float(plain[0,0]),constant_derivative=float(plain[0,2]),independent_cross_mean=float(cross)))


if __name__=='__main__':checks()
