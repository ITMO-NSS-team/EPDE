from calibration_context import *
import native_score as Native
from scipy.optimize import minimize


def likelihood(data,z,original=False):
    log_sigma,ar,lk,ap1,ap2=z;r,p1,p2=np.tanh([ar,ap1,ap2]);k=np.exp(lk);variance=np.exp(2*log_sigma);n=data.size
    q,ld,dq,dld=(Native.ORIGINAL if original else Native.axis_score)(data,r,k,p1,p2)
    jac=np.array([1-r*r,1.,1-p1*p1,1-p2*p2]);gradient=np.r_[1-q/(n*variance),.5*(dld+dq/variance)*jac/n]
    return float(log_sigma+.5*(ld+q/variance)/n),gradient


def region_fit(y,model,info):
    data=F.A.Contrasts(y).fields[0];z=np.r_[np.log(model[0]),info['transformed']];value,gradient=likelihood(data,z,True);hs=[]
    for step in (2e-4,1e-4):
        h=[]
        for j in range(5):
            delta=np.zeros(5);delta[j]=step;h.append((likelihood(data,z+delta)[1]-likelihood(data,z-delta)[1])/(2*step))
        h=np.array(h).T*data.size;hs.append((h+h.T)/2)
    h=hs[1];ev=np.linalg.eigvalsh(h);relative_difference=float(np.linalg.norm(hs[0]-h)/max(np.linalg.norm(h),1e-300));status='ok';cov=None
    if ev[0]<=1e-9*max(abs(ev)) or np.linalg.cond(h)>1e12:status='nonpositive_or_ill_conditioned_information'
    elif relative_difference>2e-3:status='unstable_hessian'
    else:cov=np.linalg.inv(h)[:2,:2]
    return dict(status=status,theta=z[:2],parameters=z,negative_log_likelihood_per_contrast=value,gradient=gradient,hessian=h,hessian_eigenvalues=ev,hessian_relative_difference=relative_difference,noise_covariance=cov)


def restricted(y,theta,fitinfo):
    data=F.A.Contrasts(y).fields[0];bounds=[(-40.,20.),(-F.P.BOUND,F.P.BOUND),(-F.P.BOUND,F.P.BOUND)];lo=np.array([v[0] for v in bounds]);hi=np.array([v[1] for v in bounds])
    starts=[np.array(fitinfo['transformed'])[1:]]+[F.P.transform([0.,*s],1)[1:] for s in ((-5.,.99,-.8),(-3.,.95,-.5),(-11.,.9,0.),(-15.,.9,0.))]
    def fn(u):
        try:v,g=likelihood(data,np.r_[theta,u]);return v,g[2:]
        except (FloatingPointError,np.linalg.LinAlgError):return 1e30,np.zeros(3)
    rows=[]
    for index,u in enumerate(starts):
        opt=minimize(fn,u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));v,g=fn(opt.x);pg=float(max(abs(opt.x-np.clip(opt.x-g,lo,hi))));rows.append(dict(start=index,nuisance=opt.x,value=v,projected_gradient=pg))
    best=min(rows,key=lambda r:r['value']);u=np.array(best['nuisance']);v,g=likelihood(data,np.r_[theta,u],True);pg=float(max(abs(u-np.clip(u-g[2:],lo,hi))))
    return dict(status='ok' if pg<=2e-5 else 'unresolved_restricted_fit',theta=theta,nuisance=u,value=v,gradient=g,projected_gradient=pg,native_value_difference=abs(v-best['value']),starts=rows)


def evaluate(y,model,info,region,theta):
    profile=restricted(y,theta,info);raw_lr=2*(len(y)-2)*(profile['value']-region['negative_log_likelihood_per_contrast']);lr_status=profile['status']
    if info['status']!='ok':lr_status='unresolved_unrestricted_fit'
    if raw_lr < -1e-5:lr_status='restricted_better_than_unrestricted'
    lr=max(0.,raw_lr) if lr_status=='ok' else None
    delta=np.array(theta)-region['theta'];wald=float(delta@np.linalg.solve(region['noise_covariance'],delta)) if region['status']=='ok' and info['status']=='ok' else None
    return dict(lr=lr,raw_lr=raw_lr,lr_status=lr_status,wald=wald,wald_status=region['status'] if info['status']=='ok' else 'unresolved_unrestricted_fit',profile=profile)
