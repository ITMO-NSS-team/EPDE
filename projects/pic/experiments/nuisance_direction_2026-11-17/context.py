from pathlib import Path
import os,sys,json,hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'identifiability_design_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import research_context as C
import certified_starts as Repair
import design_engine as D
from run_repair import Fresh
np,R,S,O,N,B=C.np,C.R,C.S,C.O,C.N,C.B
Legacy,Pilot,F=C.Legacy,C.Pilot,C.F
sys.path.insert(0,str(ROOT))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):R.dump(Path(path),data)
def sources(*names):return {**C.sources('certified_starts.py','design_engine.py','design_metric.py','run_repair.py'),**{str(ROOT/n):sha(ROOT/n) for n in ('context.py','protocol.txt',*names)}}

def matrices(e,fields,model=None,correct=True):return D.matrices(e,fields,model,correct)
def metric(e,model):return D.all_metrics(e,{'fixed':e.pilot},model)['fixed']

class Experiment(Fresh):
    def __init__(self,name,seed,profile,design=None):
        super().__init__(name,seed,profile)
        if design is not None:
            screen=read(design);specs=dict(self.specs);specs['selected']=screen['specs'][screen['selected']];D.build(self,specs)
    def matrices(self,fields,model=None,correct=True):return matrices(self,fields,model,correct)

def load_saved(path):
    path=Path(path);d=read(path)
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    e,_,_=C.primary(d['system'],'iid',51);e.seed=d['seed'];e.profile=d['profile'];e.pid,e.rho=S.PROFILES[e.profile];e.model=d['model'];e.raw=e.true_field(0,True);e.pilot={v:a['anchor_pilot_'+v] for v in e.raw}
    D.build(e,d['specs'])
    return e,d,a
