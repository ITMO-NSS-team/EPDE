from common import *
import local6_fit as F
def sources():return {**S.source_hashes(),**hashes('paired_study.py','run_case.py','local6_fit.py','common.py','protocol.txt','local6_protocol.txt','coverage_decision.txt')}
class Study:
    def __init__(self,name,seed,profile):
        self.name,self.seed,self.profile=name,seed,profile;self.p=R.datasets.load(name);self.axes=R.axes_of(self.p);self.pid,self.rho=S.PROFILES[profile]
        self.cs,self.labels=R.panel(self.p);self.terms,self.opaque=R.symbols(self.p,self.labels);base,self.old=S.E.F.baseline(name,0);self.inputs={str(base):sha(base)}
        self.raw=self.true_field(0,True);self.model,self.fit_info=self.estimate(self.raw)
        self.pilot,self.pilot_info=R.reconstruct(self.raw,self.axes,np.arange(len(self.axes[0])))
        self.specs={'local':N.spec_from_plan(self.axes,self.old['base_plan']),'broad':N.common_spec([self.axes],self.terms,self.old['base_plan']['q'],k=2)}
        self.banks={k:N.Bank(self.axes,self.terms,self.opaque,s) for k,s in self.specs.items()};self.basis=S.E.M.RealBasis(2,len(self.axes))
        self.active=[i for i,t in enumerate(self.terms) if t.data];self.deterministic=[i for i in range(len(self.terms)) if i not in self.active]
    def estimate(self,fields):
        models={};infos={}
        for v,y in fields.items():models[v],infos[v]=F.estimate(y)
        return models,infos
    def true_field(self,b,anchor=False):
        key=[980101,self.pid,self.seed] if anchor else [980104,self.pid,self.seed,b];rng=np.random.default_rng(np.random.SeedSequence(key))
        return {v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[self.rho]*y.ndim) for v,y in self.p.data.items()}
    def draw(self,rng):
        noise={v:R.WF.ar_noise(rng,y.shape,*self.model[v][:2]) for v,y in self.raw.items()}
        return {v:self.pilot[v]+y for v,y in noise.items()},noise
    def matrices(self,fields,model=None,correct=True):
        cov=O.Covariance(self.axes,np.eye(len(self.axes[0])),model or self.model) if correct else None
        out={}
        for tag,bank in self.banks.items():
            m=O.matrix(bank,fields,cov) if correct else bank.matrix(fields)
            out[tag]=self.basis.convert(m) if tag=='broad' else m.real
        return out
