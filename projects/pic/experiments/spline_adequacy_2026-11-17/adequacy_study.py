from adequacy_context import *
import adequacy_fit as A
import adjacent_fit as B
from concurrent.futures import ProcessPoolExecutor,as_completed

PROFILES=('iid','positive','negative')
FREQUENCIES=(1,2,4,6,8,12)
RHOS=(-.4,0.,.4,.8)
JOBS=[dict(cohort='primary',name=name,profile=p,seed=s) for name,ss in [('ode',range(1800,1864)),('vdp',range(1800,1816)),('duffing',range(1800,1808))] for s in ss for p in PROFILES]
JOBS += [dict(cohort='stress',name='ode',profile='positive',seed=s) for s in range(2000,3024)]
JOBS += [dict(cohort='analytic',name=f'frequency{freq}',frequency=freq,profile=p,seed=s) for freq in FREQUENCIES for p in PROFILES for s in range(16)]
JOBS += [dict(cohort='control',name=f'n{n}',n=n,rho_index=k,profile=f'rho{rho}',seed=s) for n in (320,1001) for k,rho in enumerate(RHOS) for s in range(32)]

def sources():return src('adequacy_fit.py','adjacent_fit.py','adequacy_study.py','protocol.txt')
def key_of(j):return f"{j['cohort']}_{j['name']}_{j['profile']}_seed{j['seed']}"
def compact(f):return {k:v for k,v in f.items() if k not in ('gram','beta')}

def generate(j):
    if j['cohort'] in ('primary','stress'):
        e=Point.case(j['name'],j['profile'],j['seed']);return e.raw['u'],e.p.data['u'],.05*np.std(e.p.data['u']),e.rho,e.inputs
    if j['cohort']=='analytic':
        x=np.linspace(0,1,320);freq=j['frequency'];clean=np.sin(2*np.pi*freq*x)+.3*np.cos(4*np.pi*freq*x)+.2*x;sigma=.05*np.std(clean);pid,rho=S.PROFILES[j['profile']];rng=np.random.default_rng(np.random.SeedSequence([991101,freq,pid,j['seed']]))
    else:
        clean=np.zeros(j['n']);sigma=1.;rho=RHOS[j['rho_index']];rng=np.random.default_rng(np.random.SeedSequence([991102,j['n'],j['rho_index'],j['seed']]))
    return clean+R.WF.ar_noise(rng,clean.shape,sigma,[rho]),clean,sigma,rho,{}

def observe(j):
    y,clean,sigma,rho,inputs=generate(j);model,info,mean=G.estimate(y);own,oc=A.analyze(y,info);adj,ac=B.analyze(y,info);selections=dict(cv=dict(selected_p=info['selected_p'],status=info['status']),fixed32=dict(selected_p=32,status=info['full_fits']['32']['status']));selections.update({'own_'+k:v for k,v in oc.items()});selections.update({'adj_'+k:v for k,v in ac.items()});info['full_fits']={p:compact(f) for p,f in info['full_fits'].items()};key=key_of(j);path=ROOT/'frozen'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,raw=y,cv_mean=mean)
    dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,job=j,n=len(y),cv_info=info,own_screens=own,adj_screens=adj,selections=selections,raw_sha256=R.digest(y),array_sha256=sha(path),source_sha256=sources(),input_sha256=inputs));return key

def evaluate(key):
    path=ROOT/'frozen'/(key+'.json');d=read(path);y,clean,sigma,rho,inputs=generate(d['job']);assert R.digest(y)==d['raw_sha256'];rows={};computed={}
    for method,sel in d['selections'].items():
        p=sel['selected_p']
        if p is None:rows[method]=dict(status=sel['status'],selected_p=None,coverage=None,unbounded_certified=None,sigma_ratio=None,rho_error=None);continue
        if p not in computed:
            fit=d['cv_info']['full_fits'][str(p)];reg=G.Regression(y,p);value=H.likelihood(reg,sigma,rho);base=reg.df*(fit['objective']+1);limit=next(r for r in d['own_screens'] if r['p']==p)['limit_lr'];computed[p]=dict(sigma_ratio=fit['sigma']/sigma,rho_error=fit['rho']-rho,lr=value-base,truth_nll=value,nll0=base,coverage=value-base<=H.CUTOFF,unbounded_certified=limit<H.CUTOFF-1e-6,limit_lr=limit,near_unit=abs(fit['rho'])>.95)
        rows[method]=dict(status=sel['status'],selected_p=p,**computed[p])
    dump(ROOT/'results'/(key+'.json'),dict(status='COMPLETE',key=key,job=d['job'],true_sigma=sigma,true_rho=rho,methods=rows,source_sha256=sources(),input_sha256={str(path):sha(path)}));return key

def main():
    assert read(ROOT/'checks.json')['status']=='PASS';assert len(JOBS)==1832;dump(ROOT/'launch.json',dict(status='FROZEN',jobs=JOBS,source_sha256=sources()));keys=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(observe,j) for j in JOBS]):
            keys.append(f.result())
            if len(keys)%64==0:print('fitted',len(keys),'of1832',flush=True)
        dump(ROOT/'all_frozen.json',dict(status='FROZEN',source_sha256=sources(),input_sha256={str(ROOT/'frozen'/(k+'.json')):sha(ROOT/'frozen'/(k+'.json')) for k in keys}))
        for f in as_completed([pool.submit(evaluate,key) for key in keys]):f.result()
    dump(ROOT/'process.json',dict(status='COMPLETE',cases=len(keys),source_sha256=sources()));print('STUDY COMPLETE',len(keys),flush=True)

if __name__=='__main__':main()
