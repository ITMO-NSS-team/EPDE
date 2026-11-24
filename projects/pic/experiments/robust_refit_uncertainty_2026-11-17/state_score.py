"""Exact AR2-curvature contrast likelihood via stationary Kalman innovations.

State=(noise_t,noise_t-1,noise_t-2,curvature_t,curvature_t-1).
No candidate dynamics are integrated; this evaluates a statistical covariance.
"""
from common import np
H=np.array([1.,-2.,1.,1.,0.])
def axis_score(data,r,k,p1,p2):
    n,reps=data.shape;a1=p1*(1-p2);v=(1-p1*p1)*(1-p2*p2)
    trans=np.zeros((5,5));trans[0,0]=r;trans[1,0]=trans[2,1]=trans[4,3]=1;trans[3,3]=a1;trans[3,4]=p2
    dt=np.zeros((4,5,5));dt[0,0,0]=1;dt[2,3,3]=1-p2;dt[3,3,3]=-p1;dt[3,3,4]=1
    noise=np.zeros((5,5));noise[0,0]=1-r*r;noise[3,3]=k*v
    dn=np.zeros((4,5,5));dn[0,0,0]=-2*r;dn[1,3,3]=k*v;dn[2,3,3]=-2*k*p1*(1-p2*p2);dn[3,3,3]=-2*k*p2*(1-p1*p1)
    lag=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:]);cov=np.zeros((5,5));cov[:3,:3]=r**lag;cov[3:,3:]=k*np.array([[1.,p1],[p1,1.]])
    dc=np.zeros((4,5,5));mask=lag>0;dc[0,:3,:3][mask]=lag[mask]*r**(lag[mask]-1);dc[1,3:,3:]=cov[3:,3:];dc[2,3,4]=dc[2,4,3]=k
    x=np.zeros((5,reps));dx=np.zeros((4,5,reps));q=ld=0.;dq=np.zeros(4);dld=np.zeros(4)
    for y in data:
        c=cov@H;f=float(H@c);ddc=dc@H;df=ddc@H
        if f<=0 or not np.isfinite(f):raise FloatingPointError('Nonpositive innovations variance')
        e=y-H@x;de=-np.einsum('i,kij->kj',H,dx);ee=float(e@e)
        q+=ee/f;ld+=reps*np.log(f);dq+=2*(de@e)/f-ee*df/(f*f);dld+=reps*df/f
        gain=c/f;dg=ddc/f-df[:,None]*c[None,:]/(f*f)
        post=x+gain[:,None]*e;dpost=dx+dg[:,:,None]*e[None,None,:]+gain[None,:,None]*de[:,None,:]
        # Joseph form avoids cancellation in the posterior covariance.
        j=np.eye(5)-np.outer(gain,H);dj=-dg[:,:,None]*H[None,None,:]
        pc=j@cov@j.T;dpc=dj@cov@j.T+j@dc@j.T+j@cov@np.swapaxes(dj,-1,-2)
        cov=trans@pc@trans.T+noise;dc=dt@pc@trans.T+trans@dpc@trans.T+trans@pc@np.swapaxes(dt,-1,-2)+dn
        x=trans@post;dx=dt@post+trans@dpost
    return q,ld,dq,dld
def profile(cc,rhos,kappas,p1s,p2s):
    dim=len(cc.fields);q=ld=0.;dq=np.zeros(4*dim);dld=np.zeros(4*dim)
    for axis,(data,r,k,p1,p2) in enumerate(zip(cc.fields,rhos,kappas,p1s,p2s)):
        qq,ll,gg,hh=axis_score(data,r,k,p1,p2);q+=qq;ld+=ll;dq[axis::dim]=gg;dld[axis::dim]=hh
    return float(np.log(q/cc.count)+ld/cc.count),dq/q+dld/cc.count,q
