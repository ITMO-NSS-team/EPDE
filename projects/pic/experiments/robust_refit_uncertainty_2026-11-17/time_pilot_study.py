from common import *
import paired_study_fast as Base
import time_fit as F
import correlated_pilot as P

def sources():
    return {**S.source_hashes(),**hashes('time_pilot_study.py','time_pilot_case.py','time_pilot_student.py','time_pilot_coverage_protocol.txt','time_fit.py','correlated_pilot.py','state_score.py','repair_fit.py','fast_covariance.py','common.py','paired_study_fast.py','studentize.py','patch_studentizer.py')}

class Study(Base.Study):
    def __init__(self,name,seed,profile):
        super().__init__(name,seed,profile)
        self.pilot,self.pilot_info=P.reconstruct(self.raw,self.axes,self.model)
    def estimate(self,fields):
        models={};infos={}
        for v,y in fields.items():models[v],infos[v]=F.estimate(y)
        return models,infos
    def true_field(self,b,anchor=False):
        key=[983101,self.pid,self.seed] if anchor else [983104,self.pid,self.seed,b];rng=np.random.default_rng(np.random.SeedSequence(key))
        return {v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[self.rho]*y.ndim) for v,y in self.p.data.items()}
