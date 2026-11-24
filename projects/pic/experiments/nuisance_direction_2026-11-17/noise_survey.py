from context import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(profile,seed):
    start=time.monotonic();e=Experiment('ode',seed,profile);key=f'ode_{profile}_seed{seed}';model=e.model;obs=e.matrices(e.raw);cov=metric(e,model)
    meta=dict(system='ode',profile=profile,seed=seed,model=model,fit_info=e.fit_info,pilot_info=e.pilot_info,raw_sha256={v:R.digest(y) for v,y in e.raw.items()},source_sha256=sources('noise_survey.py'))
    fp=ROOT/'survey_frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN'));fh=sha(fp)
    truth={v:[.05*np.std(y),[e.rho]*y.ndim,model[v][2]] for v,y in e.p.data.items()};clean=e.matrices(e.p.data,correct=False);oracle=e.matrices(e.raw,truth);ocov=metric(e,truth);ss={}
    for b in e.banks:ss[b]=dict(fitted=Legacy.score(e,S.gram(obs[b]-clean[b]),cov[b])[0],oracle_noise=Legacy.score(e,S.gram(oracle[b]-clean[b]),ocov[b])[0])
    out=dict(**meta,status='COMPLETE',truth_for_evaluation=truth,sigma_ratio={v:model[v][0]/truth[v][0] for v in model},rho_error={v:model[v][1][0]-e.rho for v in model},scores=ss,frozen_sha256=fh,seconds=time.monotonic()-start)
    assert sha(fp)==fh;dump(ROOT/'survey'/(key+'.json'),out);print('survey',profile,seed,'sigma',out['sigma_ratio']['u'],'rhoerror',out['rho_error']['u'],flush=True);return dict(profile=profile,seed=seed,status=e.fit_info['u']['status'])

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,p,s) for p in ('iid','positive','negative') for s in range(100,132)]):rows.append(f.result())
    dump(ROOT/'survey_process.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('noise_survey.py')))

if __name__=='__main__':main()
