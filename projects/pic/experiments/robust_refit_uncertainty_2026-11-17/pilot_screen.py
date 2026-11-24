from common import *
import time_fit as F
import correlated_pilot as P
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(seed):
    start=time.monotonic();path=ROOT/'time_validation_frozen'/f'ac_positive_seed{seed}.json';f=read(path);p=R.datasets.load('ac');pid,rho=S.PROFILES['positive'];rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==f['raw_sha256'];models={v:mm[0] for v,mm in f['time'].items()};pilot,info=P.reconstruct(raw,R.axes_of(p),models);rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]));rows=[]
    for j in range(16):
        fields={v:y+R.WF.ar_noise(rng,y.shape,*models[v][:2]) for v,y in pilot.items()};fits={v:F.estimate(y) for v,y in fields.items()};rows.append(dict(draw=j,fit=fits,field_sha256={v:R.digest(y) for v,y in fields.items()}))
    source=hashes('pilot_screen.py','correlated_pilot.py','time_fit.py','pilot_protocol.txt');fp=ROOT/'pilot_screen_frozen'/path.name;dump(fp,dict(system='ac',profile='positive',seed=seed,pilot_info=info,pilot_sha256={v:R.digest(y) for v,y in pilot.items()},rows=rows,source_sha256=source,input_sha256={str(path):sha(path)}))
    old,_=R.reconstruct(raw,R.axes_of(p),np.arange(len(R.axes_of(p)[0])));result=dict(system='ac',profile='positive',seed=seed,ratios_old=[r['time']['u'][0][0]/models['u'][0] for r in f['bootstrap']],ratios_new=[r['fit']['u'][0][0]/models['u'][0] for r in rows],statuses=[r['fit']['u'][1]['status'] for r in rows],relative_pilot_rmse={k:float(np.sqrt(np.mean((v['u']-p.data['u'])**2))/models['u'][0]) for k,v in [('old',old),('new',pilot)]},source_sha256=source,frozen_sha256=sha(fp),seconds=time.monotonic()-start);dump(ROOT/'pilot_screen'/path.name,result);return result

def main():
    assert read(ROOT/'pilot_checks.json')['status']=='PASS'
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,s) for s in (45,46,47)]):r=f.result();print(r['seed'],np.mean(r['ratios_old']),np.mean(r['ratios_new']),r['relative_pilot_rmse'],flush=True)
if __name__=='__main__':main()
