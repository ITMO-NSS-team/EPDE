from study_context import *
import guarded_fit as Fit
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,argparse

JOBS=[('positive',s) for s in (57,58,59)]+[(p,s) for p in ('iid','positive','negative') for s in (60,61,62)]
def key_of(profile,seed):return f'ode_{profile}_seed{seed}'
def moments(e,fields,model=None,correct=True):return D.matrices(e,fields,model,correct)
def covariance(e,model):return D.all_metrics(e,{'fixed':e.pilot},model)['fixed']
def src(*names):return sources('pipeline.py','guarded_fit.py','native_score.py','native_score.c','native_score.dll',*names)

def load_case(key):
    path=ROOT/'anchors'/(key+'.json');d=read(path);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.profile=d['profile'];e.pid,e.rho=S.PROFILES[e.profile];e.model=d['model'];e.raw=e.true_field(0,True)
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    e.pilot={v:a['pilot_'+v] for v in e.raw};assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256'];return e,d,a

def observe(profile,seed):
    start=time.monotonic();e,_,_=C.primary('ode','iid',51);e.seed=seed;e.profile=profile;e.pid,e.rho=S.PROFILES[profile];e.raw=e.true_field(0,True);e.model={};infos={}
    for v,y in e.raw.items():e.model[v],infos[v]=Fit.estimate(y)
    e.pilot,pinfo=Pilot.reconstruct(e.raw,e.axes,e.model);cov=covariance(e,e.model);obs=moments(e,e.raw);pilot=moments(e,e.pilot,correct=False);arrays={}
    for v,y in e.pilot.items():arrays['pilot_'+v]=y
    for b in e.banks:arrays[b+'_observed']=obs[b];arrays[b+'_pilot']=pilot[b];arrays[b+'_covariance']=cov[b]
    key=key_of(profile,seed);ap=ROOT/'anchors'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays)
    out=dict(status='FROZEN',system='ode',profile=profile,seed=seed,model=e.model,fit_info=infos,pilot_info=pinfo,raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,specs=e.specs,array_sha256=sha(ap),array_hashes={k:R.digest(y) for k,y in arrays.items()},source_sha256=src(),input_sha256=e.inputs,seconds=time.monotonic()-start)
    dump(ap.with_suffix('.json'),out);print('observed',key,'model',e.model,'seconds',round(out['seconds'],1),flush=True);return key

def chunk(key,start):
    began=time.monotonic();e,d,a=load_case(key);rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]));rows=[];arrays={};grams={b:[] for b in e.banks};covs={b:[] for b in e.banks}
    for j in range(start+8):
        fields,_=e.draw(rng)
        if j<start:continue
        model={};info={}
        for v,y in fields.items():model[v],info[v]=Fit.estimate(y)
        mm=moments(e,fields,model);cv=covariance(e,model)
        for b in e.banks:
            error=mm[b]-a[b+'_pilot'];assert np.all(error[:,e.deterministic]==0);grams[b].append(S.gram(error));covs[b].append(cv[b])
        rows.append(dict(draw=j,model=model,info=info,field_sha256={v:R.digest(y) for v,y in fields.items()}))
    for b in e.banks:arrays[b+'_grams']=np.stack(grams[b]);arrays[b+'_covariances']=np.stack(covs[b])
    ap=ROOT/'chunks'/f'{key}_start{start:03d}.npz';ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays);out=dict(status='COMPLETE',key=key,start=start,rows=rows,array_sha256=sha(ap),source_sha256=src(),input_sha256={str(ROOT/'anchors'/(key+'.json')):sha(ROOT/'anchors'/(key+'.json'))},seconds=time.monotonic()-began);dump(ap.with_suffix('.json'),out)
    return dict(key=key,start=start,seconds=out['seconds'],guard_triggers=sum(i['guard_triggered'] for r in rows for i in r['info'].values()),warnings=sum(i['status']!='ok' for r in rows for i in r['info'].values()))

def assemble(key):
    e,d,a=load_case(key);fits=[];grams={b:[] for b in e.banks};covs={b:[] for b in e.banks};inputs={str(ROOT/'anchors'/(key+'.json')):sha(ROOT/'anchors'/(key+'.json'))}
    for start in range(0,128,8):
        p=ROOT/'chunks'/f'{key}_start{start:03d}.json';r=read(p);assert sha(p.with_suffix('.npz'))==r['array_sha256'];fits.extend(r['rows']);inputs[str(p)]=sha(p)
        with np.load(p.with_suffix('.npz')) as z:
            for b in e.banks:grams[b].append(z[b+'_grams']);covs[b].append(z[b+'_covariances'])
    scores={m:{} for m in ('studentized','fixed_metric')};cuts={};joint={};regions={}
    for b in e.banks:
        a[b+'_bootstrap_grams']=gg=np.concatenate(grams[b]);a[b+'_bootstrap_covariances']=cc=np.concatenate(covs[b]);scores['studentized'][b]=[Legacy.score(e,g,c)[0] for g,c in zip(gg,cc)];scores['fixed_metric'][b]=[Legacy.score(e,g,a[b+'_covariance'])[0] for g in gg]
    for mode,ss in scores.items():
        cuts[mode]={b:Legacy.cutoff(q) for b,q in ss.items()};regions[mode]={}
        for b in e.banks:
            a[mode+'_'+b+'_scores']=np.array(ss[b]);g=S.gram(a[b+'_observed']);cv=a[b+'_covariance'];cut=cuts[mode][b];regions[mode][b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*cv,g,cv,c,cut)) for c in e.cs]
        cut=Legacy.cutoff(np.maximum(ss['local'],ss['broad']));joint[mode]=dict(cutoff=cut,certificates=[B.B.C.J.certificate([S.gram(a[b+'_observed']) for b in e.banks],[a[b+'_covariance'] for b in e.banks],cut,c,e.seed) for c in e.cs])
    ap=ROOT/'point_frozen'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**a);out=dict(status='FROZEN',system='ode',profile=e.profile,seed=e.seed,model=e.model,fit_info=d['fit_info'],refits=fits,candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,cuts=cuts,joint=joint,regions=regions,array_hashes={k:R.digest(v) for k,v in a.items()},array_sha256=sha(ap),source_sha256=src(),input_sha256=inputs)
    dump(ap.with_suffix('.json'),out);print('point frozen',key,'cuts',cuts,flush=True);return key

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();assert read(ROOT/'native_arithmetic_checks.json')['status']=='PASS';assert read(ROOT/'native_fit_checks.json')['status']=='PASS'
    dump(ROOT/'launch.json',dict(status='FROZEN',jobs=JOBS,workers=a.workers,source_sha256=src()))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        keys=[]
        for f in as_completed([pool.submit(observe,*j) for j in JOBS]):keys.append(f.result())
        keys=sorted(keys);rows=[]
        for f in as_completed([pool.submit(chunk,key,start) for start in range(0,128,8) for key in keys]):
            r=f.result();rows.append(r)
            if len(rows)%12==0:print('chunks',len(rows),'of',len(keys)*16,'recent',r,flush=True)
        for f in as_completed([pool.submit(assemble,key) for key in keys]):f.result()
    dump(ROOT/'point_process.json',dict(status='COMPLETE',jobs=JOBS,chunks=rows,source_sha256=src()))

if __name__=='__main__':main()
