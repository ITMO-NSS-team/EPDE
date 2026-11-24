from pathlib import Path
import sys,os,json,hashlib,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'spline_compatibility_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import compat_context as Prior
import compat_fit as H
np,R,S,C,G,Point=Prior.np,Prior.R,Prior.S,Prior.C,Prior.G,Prior.Point
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):R.dump(Path(p),d)
def src(*names):return {**Prior.src('compat_fit.py'),**{str(ROOT/n):sha(ROOT/n) for n in ('adequacy_context.py',*names)}}
