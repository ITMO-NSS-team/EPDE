from context import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,argparse

def run(name,profile,seed):
    start=time.monotonic();design=ROOT/'design_screen.json' if name=='kdv_cossin' else None;e=Experiment(name,seed,profile,design);key=f'{name}_{profile}_seed{seed}'
    anchor=metric(e,e.model);obs=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);arrays={};fits=[];hashes=[];warnings=[]
    scores={mode:{b:[] for b in e.banks} for mode in ('studentized','fixed_metric')};grams={b:[] for b in e.banks};covs={b:[] for b in e.banks}
    def check(info,j):
        for v,i in info.items():
            if i['status']!='ok':warnings.append(dict(draw=j,variable=v,status=i['status']))
    check(e.fit_info,-1)
    for v,y in e.pilot.items():arrays['anchor_pilot_'+v]=y
    for b in e.banks:arrays[b+'_observed']=obs[b];arrays[b+'_pilot']=pilot[b];arrays[b+'_covariance']=anchor[b]
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,seed]))
    for j in range(128):
        fields,_=e.draw(rng);model,info=e.estimate(fields);check(info,j);mat=e.matrices(fields,model);cv=metric(e,model);hashes.append({v:R.digest(y) for v,y in fields.items()});fits.append(dict(model=model,info=info))
        for b in e.banks:
            error=mat[b]-pilot[b];assert np.all(error[:,e.deterministic]==0);g=S.gram(error);grams[b].append(g);covs[b].append(cv[b]);scores['studentized'][b].append(Legacy.score(e,g,cv[b])[0]);scores['fixed_metric'][b].append(Legacy.score(e,g,anchor[b])[0])
        if (j+1)%32==0:print(key,'bootstrap',j+1,round(time.monotonic()-start,1),flush=True)
    pairs={'original':['local','broad']}
    if design:pairs['replacement']=['local','selected']
    cuts={};joint={};regions={}
    for b in e.banks:arrays[b+'_bootstrap_grams']=np.stack(grams[b]);arrays[b+'_bootstrap_covariances']=np.stack(covs[b])
    for mode,ss in scores.items():
        cuts[mode]={b:Legacy.cutoff(q) for b,q in ss.items()};joint[mode]={};regions[mode]={}
        for b in e.banks:
            arrays[mode+'_'+b+'_scores']=np.array(ss[b]);g=S.gram(obs[b]);cut=cuts[mode][b];regions[mode][b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*anchor[b],g,anchor[b],c,cut)) for c in e.cs]
        for pair,bb in pairs.items():
            cut=Legacy.cutoff(np.maximum(*[ss[b] for b in bb]));joint[mode][pair]=dict(banks=bb,cutoff=cut,certificates=[B.B.C.J.certificate([S.gram(obs[b]) for b in bb],[anchor[b] for b in bb],cut,c,seed) for c in e.cs])
    meta=dict(system=name,profile=profile,seed=seed,model=e.model,fit_info=e.fit_info,pilot_info=e.pilot_info,refits=fits,
        raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},bootstrap_field_sha256=hashes,
        candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,specs=e.specs,cuts=cuts,joint=joint,regions=regions,warnings=warnings,
        source_sha256=sources('run.py'),input_sha256={**e.inputs,**({str(design):sha(design)} if design else {})})
    fp=ROOT/'frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN',array_hashes={k:R.digest(v) for k,v in arrays.items()}));fh=sha(fp)
    clean=e.matrices(e.p.data,correct=False);actual={};coverage={};references=[]
    for b in e.banks:arrays[b+'_clean']=clean[b];arrays[b+'_error_gram']=S.gram(obs[b]-clean[b]);actual[b]=Legacy.score(e,arrays[b+'_error_gram'],anchor[b])[0]
    for mode in scores:
        coverage[mode]={b:actual[b]<=cuts[mode][b] for b in e.banks};coverage[mode].update({pair:max(actual[b] for b in bb)<=joint[mode][pair]['cutoff'] for pair,bb in pairs.items()})
        for b in e.banks:
            for c in e.cs:
                if c['label']!='reference':continue
                v=S.E.vector(c,c['exact'],len(e.labels));bound=cuts[mode][b]*float(v@anchor[b]@v);references.append(dict(mode=mode,bank=b,candidate=c['id'],zero_accepted=float(v@S.gram(obs[b])@v)<=bound,noise_covered=float(v@arrays[b+'_error_gram']@v)<=bound))
    ap=ROOT/'results'/(key+'.npz');ap.parent.mkdir(exist_ok=True);assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(ap,**arrays);assert sha(fp)==fh
    out=dict(**meta,status='COMPLETE_WITH_WARNINGS' if warnings else 'COMPLETE',actual_scores=actual,coverage=coverage,references=references,array_sha256=sha(ap),frozen_sha256=fh,seconds=time.monotonic()-start)
    dump(ap.with_suffix('.json'),out);print(key,'DONE',coverage,round(out['seconds'],1),flush=True);return dict(key=key,status=out['status'],coverage=coverage)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();jobs=[(n,p,s) for p in ('iid','positive','negative') for s in (57,58,59) for n in ('ode','kdv_cossin')];rows=[]
    dump(ROOT/'launch.json',dict(status='FROZEN',jobs=jobs,workers=a.workers,source_sha256=sources('run.py'),screen_sha256=sha(ROOT/'design_screen.json')))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,*j) for j in jobs]):rows.append(f.result())
    dump(ROOT/'process.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('run.py')))

if __name__=='__main__':main()
