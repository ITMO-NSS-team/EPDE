"""Direct weak-moment noise regions without fitting a coefficient-noise map."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'native_moments_2026-11-16'))
import native_quadrature as N
R,O,M,F,np=N.R,N.O,N.M,N.F,N.np
import json,hashlib


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sources():
    paths=[Path(__file__),ROOT/'run_case.py',ROOT/'protocol.txt',Path(N.__file__),Path(O.__file__),Path(N.Q.__file__),
           Path(M.__file__),Path(R.__file__),Path(R.WF.__file__),Path(F.__file__),Path(N.E.__file__),Path(R.datasets.__file__)]
    return {str(p):sha(p) for p in paths}


def score(residual,variance):
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):ss=residual**2/variance
    ss=np.where((residual==0)&(variance==0),0.,ss)
    assert not np.isnan(ss).any()
    return np.mean(ss,axis=-1)


def vector(c,coef,size):
    v=np.zeros(size);v[c['yi']]=1.;v[c['xi']]-=coef;return v


class Study:
    def __init__(self,name,seed):
        self.name,self.seed=name,seed;self.p=R.datasets.load(name);self.axes=R.axes_of(self.p)
        path,self.old=F.baseline(name,0);self.inputs={str(path):sha(path)}
        allcs,alllabels=R.panel(self.p);fullterms,_=R.symbols(self.p,alllabels)
        needed={i for c in allcs if c['registered'] for i in [c['yi'],*c['xi']]};needed.add(alllabels.index(()))
        ids=sorted(needed);self.labels=[alllabels[i] for i in ids];remap={old:new for new,old in enumerate(ids)}
        self.cs=[dict(c,xi=[remap[i] for i in c['xi']],yi=remap[c['yi']]) for c in allcs if c['registered']]
        self.terms,self.opaque=R.symbols(self.p,self.labels);self.raw=self.p.noisy(5,seed)
        self.model={v:R.WF.noise_model(y) for v,y in self.raw.items()}
        self.pilot,self.info=R.reconstruct(self.raw,self.axes,np.arange(len(self.axes[0])))
        self.cov=O.Covariance(self.axes,np.eye(len(self.axes[0])),self.model)
        local=N.spec_from_plan(self.axes,self.old['base_plan'])
        broad=N.common_spec([self.axes],fullterms,self.old['base_plan']['q'],k=2)
        self.specs={'local':local,'broad':broad};self.banks={k:N.Bank(self.axes,self.terms,self.opaque,spec) for k,spec in self.specs.items()}
        self.basis=M.RealBasis(2,len(self.axes));self.pilot_moments=self.matrices(self.pilot,corrected=False)

    def matrices(self,fields,corrected=True):
        result={}
        for tag,bank in self.banks.items():
            convert=self.basis.convert if tag=='broad' else lambda x:x.real
            result[tag+'_raw']=convert(bank.matrix(fields))
            if corrected:result[tag+'_corrected']=convert(O.matrix(bank,fields,self.cov))
        return result


def statistics(arrays,queries):
    results=[]
    for arm in ('local_raw','local_corrected','broad_raw','broad_corrected'):
        bank=arm.split('_')[0];pilot=arrays[bank+'_pilot'];clean=arrays[bank+'_clean']
        for query in queries[bank]:
            if query['vector'] is None:
                results.append(dict(arm=arm,**query,status='infeasible_fit'));continue
            v=np.asarray(query['vector']);cal=arrays[arm+'_calibration']@v
            var=np.var(cal,axis=0,ddof=1)
            boot=(arrays[arm+'_check']-pilot)@v;actual=(arrays[arm+'_actual']-clean)@v
            qb=score(boot,var);qa=score(actual,var);radius=float(np.quantile(qb,.95))
            null=score(arrays[arm+'_actual']@v,var);obs=score(arrays[arm+'_observed']@v,var)
            deterministic=score(clean@v,var)
            finite=bool(np.isfinite(qb).all() and np.isfinite(qa).all() and np.isfinite(null).all() and np.isfinite(obs) and np.isfinite(deterministic))
            intervals=np.quantile(boot,[.025,.975],axis=0)
            shift=score(actual.mean(axis=0),var)
            denom=actual.std(axis=0,ddof=1);valid=denom>0
            row=dict(arm=arm,candidate=query['candidate'],label=query['label'],slot=query['slot'],kind=query['kind'],
                status='ok' if finite else 'nonfinite',radius=radius,noise_coverage_count=int(np.sum(qa<=radius)),
                zero_acceptance_count=int(np.sum(null<=radius)),anchor_acceptance=bool(obs<=radius),anchor_score=float(obs),
                clean_score=float(deterministic),mean_shift_score=float(shift),
                coordinate_coverage=float(np.mean((actual>=intervals[0])&(actual<=intervals[1]))),
                median_sd_ratio=float(np.median(boot.std(axis=0,ddof=1)[valid]/denom[valid])) if valid.any() else None,
                zero_variance_rows=int(np.sum(var==0)),rows=len(var))
            results.append(row)
    return results
