"""Adapt the previous audited protocol without changing archived files."""
from spline_context import *

source=(PREVIOUS/'bootstrap_regions.py').read_text(encoding='utf-8')
source=source.replace('from run_profiles import *', '''from spline_context import *
import spline_fit as G
from point_study import case
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import chi2
Old=Old.Old
PROFILES=('iid','positive','negative')
NOMINAL=float(chi2.ppf(.975,2))
def src(*names):return sources('spline_fit.py','point_study.py','protocol.txt',*names)
def generate(kind,p,s):
    assert kind=='ode';e=case('ode',p,s);return e.raw['u'],np.array([np.log(.05*np.std(e.p.data['u'])),np.arctanh(e.rho)]),{}
def estimate(y,method):
    if method=='guarded':return Fit.estimate(y)
    assert method=='spline';model,info,mean=G.estimate(y);return model,info''')
source=source.replace("JOBS=[(p,s) for s in range(264,272) for p in PROFILES]", "JOBS=[(p,s,m) for s in range(400,404) for p in PROFILES for m in ('guarded','spline')]")
source=source.replace("src('bootstrap_regions.py','bootstrap_protocol.txt')", "src('spline_bootstrap.py')")
source=source.replace("def bkey(p,s):return f'ode_{p}_seed{s}'", "def bkey(p,s,method):return f'ode_{p}_seed{s}_{method}'")
source=source.replace('def observe(p,s):','def observe(p,s,method):').replace('model,info=Fit.estimate(y);e,_,_', 'model,info=estimate(y,method);e,_,_').replace('key=bkey(p,s);','key=bkey(p,s,method);')
source=source.replace("key=key,profile=p,seed=s,model=model", "key=key,profile=p,seed=s,method=method,model=model")
source=source.replace('model,info=Fit.estimate(field);','model,info=estimate(field,d[\'method\']);')
source=source.replace("key=key,profile=d['profile'],seed=d['seed'],true_noise", "key=key,profile=d['profile'],seed=d['seed'],method=d['method'],true_noise")
source=source.replace("read(ROOT/'process.json')", "read(ROOT/'point_process.json')")
(ROOT/'spline_bootstrap.py').write_text(source,encoding='utf-8')

source=(PREVIOUS/'propagate.py').read_text(encoding='utf-8').replace('from bootstrap_regions import *','from spline_bootstrap import *')
source=source.replace("src('bootstrap_regions.py','bootstrap_protocol.txt','propagate.py','propagation_protocol.txt')", "src('spline_bootstrap.py','spline_propagate.py')")
source=source.replace("key=key,profile=e.profile,seed=e.seed,coverage", "key=key,profile=e.profile,seed=e.seed,method=d['method'],coverage")
(ROOT/'spline_propagate.py').write_text(source,encoding='utf-8')

source=(PREVIOUS/'audit_propagation.py').read_text(encoding='utf-8').replace('from propagate import *','from spline_propagate import *')
source=source.replace("src('audit_propagation.py','propagate.py','propagation_protocol.txt')", "src('audit_spline_propagation.py','spline_propagate.py')")
(ROOT/'audit_spline_propagation.py').write_text(source,encoding='utf-8')
print('Prepared new bootstrap, propagation and independent audit; archives unchanged.')
