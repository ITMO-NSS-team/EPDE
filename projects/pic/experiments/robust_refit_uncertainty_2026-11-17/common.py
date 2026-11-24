import os,sys,hashlib,json
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'ar2_signal_noise_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import score_fit as A
np=A.np
import importlib.util
spec=importlib.util.spec_from_file_location('previous_ar2_coverage',PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R,O,N=C.S,C.R,C.O,C.N
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']
JOBS=[(n,'iid') for n in SYSTEMS]+[(n,p) for n in ('ac','kdv_cossin','ks','wave') for p in ('positive','negative')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,d):R.dump(Path(p),d)
def hashes(*names):return {str(ROOT/n):sha(ROOT/n) for n in names}
