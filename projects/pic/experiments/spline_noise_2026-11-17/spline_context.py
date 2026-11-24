from pathlib import Path
import sys,os,json,hashlib,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nuisance_calibration_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import calibration_context as Old
np,R,S,C,F,Pilot,Fit=Old.np,Old.R,Old.S,Old.C,Old.F,Old.Pilot,Old.Fit
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):R.dump(Path(p),d)
def sources(*names):return {**Old.src(),**{str(ROOT/n):sha(ROOT/n) for n in ('spline_context.py',*names)}}
