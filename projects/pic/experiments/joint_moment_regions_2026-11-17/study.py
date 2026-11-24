import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'residual_moment_uncertainty_2026-11-17'))
import experiment as E
import iid_studentization as I
R,O,N,np=E.R,E.O,E.N,E.np
import json
from math import comb
PROFILES={'iid':(0,0.),'positive':(1,.4),'negative':(2,-.4)}
RCOND=1e-12


def source_hashes():
    pp=[Path(__file__),ROOT/'protocol.txt',ROOT/'run_case.py',Path(I.__file__),Path(R.SV.__file__)]
    return {**E.sources(),**{str(p):E.sha(p) for p in pp}}


def amplitude(fields,scales,orders,rhos):
    total=0.;count=0
    for v,y in fields.items():
        for axis,k in enumerate(orders[v]):
            if k:
                d=np.array([(-1)**j*comb(k,j) for j in range(k+1)],float)
                c=float(rhos[axis])**np.abs(np.arange(k+1)[:,None]-np.arange(k+1)[None,:]);factor=float(d@c@d);assert factor>0
                diff=np.diff(y,n=k,axis=axis);total+=float(np.sum(diff*diff))/(factor*scales[v]**2);count+=diff.size
    if count==0 or total<=0:raise ValueError('No usable amplitude estimate')
    return float(np.sqrt(total/count))


def gram(m):return m.T@m


def metric(s,active):
    sub=s[np.ix_(active,active)];scale=np.sqrt(np.diag(sub))
    if np.any(scale<=0):return None,dict(status='zero_stochastic_variance')
    g=sub/scale[:,None]/scale[None,:];eig,u=np.linalg.eigh((g+g.T)/2);rank=int(np.sum(eig>RCOND*eig[-1]))
    meta=dict(status='ok' if rank==len(active) else 'rank_deficient',rank=rank,dimension=len(active),eigenvalues=eig)
    if rank!=len(active):return None,meta
    w=(u/np.sqrt(eig)[None,:])/scale[:,None]
    np.testing.assert_allclose(w.T@sub@w,np.eye(len(active)),rtol=2e-6,atol=2e-6)
    return w,meta


def operator_scores(gg,w,active):
    sub=gg[...,active,:][...,:,active];trans=np.matmul(w.T,np.matmul(sub,w));trans=(trans+np.swapaxes(trans,-1,-2))/2
    return np.linalg.eigvalsh(trans)[...,-1]


def quadratic_region(a,gobs,s,candidate,cutoff):
    xi,yi=candidate['xi'],candidate['yi'];p=len(xi);yy=float(a[yi,yi])
    if p==0:return dict(status='point' if yy<=0 else 'empty',feasible=bool(yy<=0),witness=[] if yy<=0 else None,minimum=yy)
    scale=np.sqrt(np.maximum(np.diag(gobs)[xi]+cutoff*np.diag(s)[xi],0.));scale=np.where(scale>0,scale,1.)
    h=a[np.ix_(xi,xi)]/scale[:,None]/scale[None,:];g=a[xi,yi]/scale;h=(h+h.T)/2
    eig,u=np.linalg.eigh(h);tol=RCOND*max(float(np.max(np.abs(eig))),np.finfo(float).tiny)
    out=dict(eigenvalues=eig,scale=scale,feasible=None,witness=None)
    if eig[0]<-tol:
        direction=u[:,0];component=float(g@direction)
        t=(1. if component>=0 else -1.)*2*np.sqrt(max(yy,0.)/(-eig[0])) if yy>0 else 0.
        witness=t*direction/scale
        out.update(status='unbounded',feasible=True,witness=witness,negative_direction=direction/scale)
    elif eig[0]>tol:
        center_n=u@((u.T@g)/eig);center=center_n/scale;minimum=float(yy-g@center_n)
        uncertainty=RCOND*(abs(yy)+abs(float(g@center_n))+np.finfo(float).tiny)
        out.update(center=center,minimum=minimum)
        if abs(minimum)<=uncertainty:out.update(status='boundary_ambiguous')
        elif minimum>0:out.update(status='empty',feasible=False)
        else:
            inverse=(u/eig[None,:])@u.T;radius=np.sqrt(-minimum*np.diag(inverse))/scale
            out.update(status='ellipsoid',feasible=True,witness=center,lower=center-radius,upper=center+radius)
    else:
        bn=np.linalg.lstsq(h,g,rcond=RCOND)[0];beta=bn/scale;v=E.vector(candidate,beta,len(a));value=float(v@a@v)
        if value<0:out.update(status='near_singular_with_witness',feasible=True,witness=beta,witness_value=value)
        elif yy<=0:out.update(status='near_singular_with_witness',feasible=True,witness=np.zeros(p),witness_value=yy)
        else:out.update(status='unresolved_singular')
    if out['witness'] is not None:
        v=E.vector(candidate,out['witness'],len(a));lhs=float(v@gobs@v);rhs=float(cutoff*(v@s@v))
        assert lhs<=rhs+2e-9*(abs(lhs)+abs(rhs)+np.finfo(float).tiny),(candidate['id'],out,lhs,rhs)
        out['witness_margin']=lhs-rhs
    return out


class Study:
    def __init__(self,name,seed,profile,kind):
        self.name,self.seed,self.profile,self.kind=name,seed,profile,kind
        self.p=R.datasets.load(name);self.axes=R.axes_of(self.p);self.cs,self.labels=R.panel(self.p);self.terms,self.opaque=R.symbols(self.p,self.labels)
        base,self.old=E.F.baseline(name,0);self.inputs={str(base):E.sha(base)}
        self.profile_id,self.rho=PROFILES[profile]
        self.raw=self.p.noisy(5,seed) if profile=='iid' else self.true_field(seed,anchor=True)
        self.ar={v:R.WF.noise_model(y) for v,y in self.raw.items()};self.orders={v:self.ar[v][2] for v in self.raw}
        self.scales={v:float(np.std(y)) for v,y in self.raw.items()}
        if kind=='ar':self.model=self.ar;self.delta=None
        else:
            rho=self.rho if kind=='known_rho' else 0.;rhos=[rho]*len(self.axes)
            self.delta=amplitude(self.raw,self.scales,self.orders,rhos)
            self.model={v:[self.delta*self.scales[v],rhos,self.orders[v]] for v in self.raw}
        self.pilot,self.info=R.reconstruct(self.raw,self.axes,np.arange(len(self.axes[0])))
        self.cov=O.Covariance(self.axes,np.eye(len(self.axes[0])),self.model)
        self.specs={'local':N.spec_from_plan(self.axes,self.old['base_plan']),
            'broad':N.common_spec([self.axes],self.terms,self.old['base_plan']['q'],k=2)}
        self.banks={k:N.Bank(self.axes,self.terms,self.opaque,v) for k,v in self.specs.items()};self.basis=E.M.RealBasis(2,len(self.axes))
        self.active=[i for i,t in enumerate(self.terms) if t.data];self.deterministic=[i for i in range(len(self.terms)) if i not in self.active]

    def true_field(self,b,anchor=False):
        if self.profile=='iid':return self.p.noisy(5,1500000+1000*self.seed+b)
        key=[940100,self.profile_id,self.seed] if anchor else [940103,self.profile_id,self.seed,b]
        rng=np.random.default_rng(np.random.SeedSequence(key))
        return {v:y+R.WF.ar_noise(rng,y.shape,.05*float(np.std(y)),[self.rho]*y.ndim) for v,y in self.p.data.items()}

    def matrices(self,fields,correct=True):
        out={}
        for banktag,bank in self.banks.items():
            m=O.matrix(bank,fields,self.cov) if correct else bank.matrix(fields)
            out[banktag]=self.basis.convert(m) if banktag=='broad' else m.real
        return out

    def draw(self,rng):
        noise={v:R.WF.ar_noise(rng,y.shape,*self.model[v][:2]) for v,y in self.raw.items()}
        return {v:self.pilot[v]+y for v,y in noise.items()},noise
