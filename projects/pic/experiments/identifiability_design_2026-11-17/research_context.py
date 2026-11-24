from pathlib import Path
import os,sys,json,hashlib,importlib.util
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'full_pipeline_uncertainty_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import bridge as B
import compressed_time as F
import full_pipeline_fast as Legacy
from exact_metric_pair import paired
np,R,S,O,N=B.np,B.R,B.S,B.O,B.N
Geometry,Covariance,Pilot=B.Geometry,B.Covariance,B.Pilot
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):R.dump(Path(path),value)
def sources(*names):
    return {**B.sources('full_pipeline_fast.py','compressed_time.py','exact_metric_pair.py'),
            **{str(ROOT/n):sha(ROOT/n) for n in ('research_context.py',*names)}}

def primary(name,profile,seed):
    path=PREVIOUS/'results'/f'{name}_{profile}_seed{seed}.json';d=read(path)
    e=Legacy.Experiment.__new__(Legacy.Experiment)
    e.name,e.profile,e.seed=name,profile,seed;e.system_id=B.SYSTEMS.index(name);e.pid,e.rho=S.PROFILES[profile]
    e.p=R.datasets.load(name);e.axes=R.axes_of(e.p);e.cs,e.labels=d['candidates'],d['labels'];e.terms,e.opaque=R.symbols(e.p,e.labels)
    base,e.old=S.E.F.baseline(name,0);e.inputs={str(base):sha(base)};e.model=d['model'];e.fit_info=d['fit_info'];e.pilot_info=d['pilot_info']
    e.active,e.deterministic=d['active'],d['deterministic'];e.raw=e.true_field(0,True)
    assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256']
    with np.load(path.with_suffix('.npz')) as z:e.pilot={v:z['anchor_pilot_'+v] for v in e.raw}
    assert {v:R.digest(y) for v,y in e.pilot.items()}==d['pilot_sha256']
    e.specs={'local':N.spec_from_plan(e.axes,e.old['base_plan']),'broad':N.common_spec([e.axes],e.terms,e.old['base_plan']['q'],k=2)}
    e.banks={k:N.Bank(e.axes,e.terms,e.opaque,s) for k,s in e.specs.items()};e.basis=S.E.M.RealBasis(2,len(e.axes));e.geometry=Geometry(e.axes)
    return e,d,path
