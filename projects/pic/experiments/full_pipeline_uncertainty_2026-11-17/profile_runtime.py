from bridge import *
from full_pipeline_saved import Experiment
import time

def main():
    timings={};start=time.monotonic();e=Experiment('ks',50,'positive');timings['initialize']=time.monotonic()-start;rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,50]));fields,_=e.draw(rng)
    start=time.monotonic();model,info=e.estimate(fields);timings['estimate']=time.monotonic()-start
    start=time.monotonic();pilot,pi=Pilot.reconstruct(fields,e.axes,model);timings['reconstruct']=time.monotonic()-start
    start=time.monotonic();e.metric(e.pilot,model);timings['fixed_metric']=time.monotonic()-start
    start=time.monotonic();e.metric(pilot,model);timings['refit_metric']=time.monotonic()-start
    dump(ROOT/'runtime_profile.json',dict(timings=timings,source_sha256=sources('profile_runtime.py','full_pipeline_saved.py'),not_primary=True));print(timings,flush=True)
if __name__=='__main__':main()
