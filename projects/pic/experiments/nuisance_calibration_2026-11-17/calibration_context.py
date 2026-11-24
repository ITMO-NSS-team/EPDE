from pathlib import Path
import sys,json,hashlib,time,os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'safeguard_envelope_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import pipeline as Old
np,R,S,C,F,Pilot,Fit=Old.np,Old.R,Old.S,Old.C,Old.F,Old.Pilot,Old.Fit
sys.path.insert(0,str(ROOT))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):R.dump(Path(path),data)
def src(*names):return {**Old.src(),**{str(ROOT/n):sha(ROOT/n) for n in ('calibration_context.py',*names)}}
