import os,sys,hashlib,json,importlib.util
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'robust_refit_uncertainty_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import common as B
np,R,S,O,N=B.np,B.R,B.S,B.O,B.N
import time_fit as Temporal
import local6_fit as Local
import correlated_pilot as Pilot
from fast_covariance import Geometry,Covariance
SYSTEMS,JOBS=B.SYSTEMS,B.JOBS
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):R.dump(Path(path),value)
def sources(*names):
    files=[ROOT/'bridge.py',PREVIOUS/'common.py',PREVIOUS/'time_fit.py',PREVIOUS/'local6_fit.py',PREVIOUS/'correlated_pilot.py',PREVIOUS/'state_score.py',PREVIOUS/'repair_fit.py',PREVIOUS/'fast_covariance.py']+[ROOT/n for n in names]
    return {**S.source_hashes(),**{str(p):sha(p) for p in files}}
