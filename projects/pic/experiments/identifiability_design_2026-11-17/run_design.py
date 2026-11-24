from design_engine import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,argparse

PAIRS={'original':['local','broad'],'replacement':['local','selected']}
class Experiment(Legacy.Experiment):
    def __init__(self,profile,seed):
        super().__init__('kdv_cossin',seed,profile)
        screen=read(ROOT/'design_screen.json');specs=dict(self.specs);specs['selected']=screen['specs'][screen['selected']];build(self,specs)
    def matrices(self,fields,model=None,correct=True):return matrices(self,fields,model,correct)

def run(profile,seed):
    start=time.monotonic();e=Experiment(profile,seed);key=f'kdv_cossin_{profile}_seed{seed}';warnings=[]
    cov=all_metrics(e,{'fixed':e.pilot},e.model)['fixed'];obs=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False)
    arrays={};fits=[];hashes=[];scores={b:[] for b in e.banks};grams={b:[] for b in e.banks};metrics={b:[] for b in e.banks}
    def check(info,draw):
        for v,i in info.items():
            if i['status']!='ok':warnings.append(dict(variable=v,draw=draw,info=i))
    check(e.fit_info,-1)
    for v,y in e.pilot.items():arrays['anchor_pilot_'+v]=y
    for b in e.banks:arrays[b+'_observed']=obs[b];arrays[b+'_pilot']=pilot[b];arrays[b+'_covariance']=cov[b]
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,seed]))
    for j in range(128):
        fields,_=e.draw(rng);model,info=e.estimate(fields);check(info,j);mat=e.matrices(fields,model);cv=all_metrics(e,{'fixed':e.pilot},model)['fixed']
        hashes.append({v:R.digest(y) for v,y in fields.items()});fits.append(dict(model=model,info=info))
        for b in e.banks:
            err=mat[b]-pilot[b];assert np.all(err[:,e.deterministic]==0);g=S.gram(err)
            grams[b].append(g);metrics[b].append(cv[b]);scores[b].append(Legacy.score(e,g,cv[b])[0])
        if (j+1)%32==0:print(key,j+1,round(time.monotonic()-start,1),flush=True)
    cuts={b:Legacy.cutoff(q) for b,q in scores.items()};regions={};joint={}
    for b in e.banks:
        arrays[b+'_bootstrap_grams']=np.stack(grams[b]);arrays[b+'_bootstrap_covariances']=np.stack(metrics[b]);arrays[b+'_scores']=np.array(scores[b])
        g=S.gram(obs[b]);regions[b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cuts[b]*cov[b],g,cov[b],c,cuts[b])) for c in e.cs]
    for mode,bb in PAIRS.items():
        cut=Legacy.cutoff(np.maximum(*[scores[b] for b in bb]));joint[mode]=dict(banks=bb,cutoff=cut,certificates=[B.B.C.J.certificate([S.gram(obs[b]) for b in bb],[cov[b] for b in bb],cut,c,seed) for c in e.cs])
    meta=dict(system=e.name,profile=profile,seed=seed,status='FROZEN',model=e.model,fit_info=e.fit_info,pilot_info=e.pilot_info,refits=fits,
              raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},bootstrap_field_sha256=hashes,
              candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,specs=e.specs,cuts=cuts,regions=regions,joint=joint,warnings=warnings,
              source_sha256=src('run_design.py','protocol.txt'),input_sha256={**e.inputs,str(ROOT/'design_screen.json'):sha(ROOT/'design_screen.json')})
    fp=ROOT/'frozen'/(key+'.json');dump(fp,dict(**meta,array_hashes={k:R.digest(v) for k,v in arrays.items()}));fh=sha(fp)
    clean=e.matrices(e.p.data,correct=False);actual={};refs=[]
    for b in e.banks:
        err=S.gram(obs[b]-clean[b]);arrays[b+'_clean']=clean[b];arrays[b+'_error_gram']=err;actual[b]=Legacy.score(e,err,cov[b])[0]
        for c in e.cs:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(e.labels));bound=cuts[b]*float(v@cov[b]@v)
            refs.append(dict(bank=b,candidate=c['id'],noise_covered=float(v@err@v)<=bound,zero_accepted=float(v@S.gram(obs[b])@v)<=bound))
    coverage={b:actual[b]<=cuts[b] for b in e.banks};coverage.update({mode:max(actual[b] for b in bb)<=joint[mode]['cutoff'] for mode,bb in PAIRS.items()})
    ap=ROOT/'results'/(key+'.npz');ap.parent.mkdir(exist_ok=True);assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(ap,**arrays);assert sha(fp)==fh
    out=dict(meta,status='COMPLETE_WITH_WARNINGS' if warnings else 'COMPLETE',actual_scores=actual,coverage=coverage,references=refs,frozen_sha256=fh,array_sha256=sha(ap),seconds=time.monotonic()-start)
    dump(ap.with_suffix('.json'),out);print(key,coverage,round(out['seconds'],1),flush=True);return dict(key=key,status=out['status'],coverage=coverage,seconds=out['seconds'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();jobs=[(p,s) for p in ('iid','positive','negative') for s in (54,55,56)];rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,*j) for j in jobs]):rows.append(f.result())
    dump(ROOT/'design_process.json',dict(status='COMPLETE',rows=rows,source_sha256=src('run_design.py')))

if __name__=='__main__':main()
