from common import *
import paired_study_fast as E
from scipy.linalg import eigvalsh
import argparse,time
MC=32
H=.001
TRUST=.05
class Studentizer:
    def __init__(self,e):
        self.e=e;self.variables=list(e.raw);self.center=self.encode(e.model);self.pilot=e.matrices(e.pilot,correct=False);self.cross={};self.exact_calls=0;self.response_calls=0;self.force_exact=False
    def encode(self,model):return np.concatenate([np.r_[np.log(model[v][0]),np.arctanh(model[v][1])] for v in self.variables])
    def decode(self,t):
        out={};j=0
        for v in self.variables:
            dim=self.e.raw[v].ndim;out[v]=[float(np.exp(t[j])),np.tanh(t[j+1:j+dim+1]).tolist(),[3]*dim];j+=dim+1
        return out
    def errors(self,t):
        model=self.decode(t);stacks={b:[] for b in self.e.banks}
        for j in range(MC):
            rng=np.random.default_rng(np.random.SeedSequence([981105,self.e.pid,self.e.seed,j]));noise={v:R.WF.ar_noise(rng,self.e.raw[v].shape,*model[v][:2]) for v in self.variables};fields={v:self.e.pilot[v]+noise[v] for v in self.variables}
            for b,m in self.e.matrices(fields,model).items():stacks[b].append(m-self.pilot[b])
        return {b:np.stack(x) for b,x in stacks.items()}
    def direct(self,t):
        self.exact_calls+=1;return {b:np.einsum('nri,nrj->ij',x,x)/MC for b,x in self.errors(t).items()}
    def build(self):
        base=self.errors(self.center);response={b:[x] for b,x in base.items()}
        for j in range(len(self.center)):
            shift=np.zeros_like(self.center);shift[j]=H;plus=self.errors(self.center+shift);minus=self.errors(self.center-shift)
            for b in base:response[b].append((plus[b]-minus[b])/(2*H))
        for b,parts in response.items():
            p=len(parts);m=parts[0].shape[-1];x=np.stack(parts,axis=2).reshape(-1,p*m);self.cross[b]=(x.T@x/MC).reshape(p,m,p,m).transpose(0,2,1,3)
    def evaluate(self,t):
        delta=t-self.center
        if self.force_exact or np.max(np.abs(delta))>TRUST:return self.direct(t)
        self.response_calls+=1;c=np.r_[1.,delta];return {b:np.einsum('i,j,ijab->ab',c,c,g,optimize=True) for b,g in self.cross.items()}
    def validate(self,parameters):
        candidates=[t for t in parameters if np.max(np.abs(t-self.center))<=TRUST]
        if not candidates:return dict(status='no_in_radius_check_parameters',relative_error=None)
        t=max(candidates,key=lambda t:np.max(np.abs(t-self.center)));approx=self.evaluate(t);exact=self.direct(t);errors=[]
        for b in exact:
            g=exact[b][np.ix_(self.e.active,self.e.active)];a=approx[b][np.ix_(self.e.active,self.e.active)];sc=np.sqrt(np.diag(g));g=g/sc[:,None]/sc[None,:];a=a/sc[:,None]/sc[None,:];eig=eigvalsh((a+a.T)/2,(g+g.T)/2);errors.append(float(np.max(np.abs(eig-1))))
        self.force_exact=max(errors)>.01;return dict(status='direct_all' if self.force_exact else 'response_validated',relative_error=max(errors),theta=t)
def run(name,profile,seed):
    start=time.monotonic();key=f'{name}_{profile}_seed{seed}';path=ROOT/'results'/(key+'.json');d=read(path);e=E.Study(name,seed,profile);student=Studentizer(e);student.build();params=[student.encode(r['model']) for r in d['refits']['check']];validation=student.validate(params);anchor={b:g[0,0] for b,g in student.cross.items()};arrays={b+'_response_cross':g for b,g in student.cross.items()};scores={tag:{b:[] for b in e.banks} for tag in ('check','actual')}
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    metrics={};regions={};counts={}
    for j,t in enumerate(params):
        covs=student.evaluate(t)
        for b,cov in covs.items():
            w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['check'][b].append(float(S.operator_scores(a[f'refit_{b}_check_grams'][j:j+1],w,e.active)[0]))
    for b in e.banks:
        cut=float(np.quantile(scores['check'][b],.95));metrics[b]=dict(cutoff=cut);g=a[f'refit_{b}_observed_gram'];regions[b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*anchor[b],g,anchor[b],c,cut)) for c in e.cs];arrays[b+'_anchor_sigma']=anchor[b];arrays[b+'_check_scores']=np.array(scores['check'][b])
    fp=ROOT/'studentized_frozen'/(key+'.json');src=hashes('studentize.py','studentization_protocol.txt','paired_study_fast.py','local6_fit.py','fast_covariance.py');dump(fp,dict(system=name,profile=profile,seed=seed,validation=validation,metrics=metrics,regions=regions,center=student.center,check_models_sha256=R.digest(np.stack(params)),array_hashes={k:R.digest(v) for k,v in arrays.items()},source_sha256=src,input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))}));fhash=sha(fp)
    for j,row in enumerate(d['refits']['actual']):
        covs=student.evaluate(student.encode(row['model']))
        for b,cov in covs.items():
            w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['actual'][b].append(float(S.operator_scores(a[f'refit_{b}_actual_error_grams'][j:j+1],w,e.active)[0]))
    for b in e.banks:arrays[b+'_actual_scores']=np.array(scores['actual'][b]);counts[b]=int(np.sum(arrays[b+'_actual_scores']<=metrics[b]['cutoff']))
    ap=ROOT/'studentized'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays)
    result=dict(system=name,profile=profile,seed=seed,status='COMPLETE',coverage_counts=counts,validation=validation,exact_metric_calls=student.exact_calls,response_metric_calls=student.response_calls,metrics=metrics,regions=regions,seconds=time.monotonic()-start,frozen_sha256=fhash,array_sha256=sha(ap),source_sha256=src,input_sha256={str(path):sha(path)})
    dump(ap.with_suffix('.json'),result);print(key,counts,validation['status'],'exact',student.exact_calls,'seconds',round(result['seconds'],1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system');p.add_argument('profile');p.add_argument('seed',type=int);a=p.parse_args();run(a.system,a.profile,a.seed)
