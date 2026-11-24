from pathlib import Path
import sys,os,json,hashlib,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'spline_noise_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import spline_context as Previous
import spline_fit as G
import point_study as Point
np,R,S,C,Pilot=Previous.np,Previous.R,Previous.S,Previous.C,Previous.Pilot
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):R.dump(Path(p),d)
def src(*names):return {**Previous.sources('spline_fit.py','point_study.py'),**{str(ROOT/n):sha(ROOT/n) for n in ('compat_context.py',*names)}}
