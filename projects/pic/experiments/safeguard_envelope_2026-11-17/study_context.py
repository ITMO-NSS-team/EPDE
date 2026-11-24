from pathlib import Path
import os,sys,json,hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nuisance_direction_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import context as Old
import profile_guard as Guard
np,R,S,O,N,B=Old.np,Old.R,Old.S,Old.O,Old.N,Old.B
F,Legacy,Pilot,C,D=Old.F,Old.Legacy,Old.Pilot,Old.C,Old.D
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):R.dump(Path(path),value)
def sources(*names):return {**Old.sources('profile_guard.py','guard_protocol.txt'),**{str(ROOT/n):sha(ROOT/n) for n in ('study_context.py','protocol.txt',*names)}}
