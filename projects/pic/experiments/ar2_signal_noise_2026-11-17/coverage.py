from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parent
import score_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S
R,O,N=S.R,S.O,S.N
spec=importlib.util.spec_from_file_location('spectral_intersection',S.ROOT/'intersect_banks.py')
J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

def sources():
    paths=[ROOT/p for p in ('coverage.py','run_case.py','fit.py','score_fit.py','score_core.py','score_protocol.txt','protocol.txt','coverage_decision.txt')]+[F.PREVIOUS/p for p in ('latent_fit.py','correlated_fit.py')]
    return {**S.source_hashes(),str(S.ROOT/'intersect_banks.py'):S.E.sha(S.ROOT/'intersect_banks.py'),**{str(p):S.E.sha(p) for p in paths}}

class Study:
    def __init__(self,name,seed,profile,kind):
        self.name,self.seed,self.profile,self.kind=name,seed,profile,kind;self.p=R.datasets.load(name);self.axes=R.axes_of(self.p)
        self.cs,self.labels=R.panel(self.p);self.terms,self.opaque=R.symbols(self.p,self.labels);base,self.old=S.E.F.baseline(name,0);self.inputs={str(base):S.E.sha(base)}
        self.profile_id,self.rho=S.PROFILES[profile];self.raw=self.true_field(0,anchor=True)
        self.ar={v:R.WF.noise_model(y) for v,y in self.raw.items()};self.orders={v:self.ar[v][2] for v in self.raw};self.scales={v:float(np.std(y)) for v,y in self.raw.items()}
        self.fit_info={};self.delta=None
        if kind=='ar2':
            self.model={}
            for v,y in self.raw.items():
                m,info=F.estimate(y);self.model[v]=m;self.fit_info[v]=info
        elif kind=='ar':self.model=self.ar
        else:
            rho=self.rho if kind=='known_rho' else 0.;rr=[rho]*len(self.axes);self.delta=S.amplitude(self.raw,self.scales,self.orders,rr)
            self.model={v:[self.delta*self.scales[v],rr,self.orders[v]] for v in self.raw}
        self.valid=all(m is not None and m[0]>0 for m in self.model.values()) and all(i['status']=='ok' for i in self.fit_info.values())
        if not self.valid:return
        self.pilot,self.pilot_info=R.reconstruct(self.raw,self.axes,np.arange(len(self.axes[0])));self.cov=O.Covariance(self.axes,np.eye(len(self.axes[0])),self.model)
        self.specs={'local':N.spec_from_plan(self.axes,self.old['base_plan']),'broad':N.common_spec([self.axes],self.terms,self.old['base_plan']['q'],k=2)}
        self.banks={k:N.Bank(self.axes,self.terms,self.opaque,v) for k,v in self.specs.items()};self.basis=S.E.M.RealBasis(2,len(self.axes))
        self.active=[i for i,t in enumerate(self.terms) if t.data];self.deterministic=[i for i in range(len(self.terms)) if i not in self.active]

    def true_field(self,b,anchor=False):
        key=[970101,self.profile_id,self.seed] if anchor else [970104,self.profile_id,self.seed,b];rng=np.random.default_rng(np.random.SeedSequence(key))
        return {v:y+R.WF.ar_noise(rng,y.shape,.05*float(np.std(y)),[self.rho]*y.ndim) for v,y in self.p.data.items()}

    matrices=S.Study.matrices
    draw=S.Study.draw
