from research_context import *
import certified_starts as Repair
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,argparse

def source():return sources('certified_starts.py','run_repair.py','repair_protocol.txt')

def regression(name,seed):
    start=time.monotonic();e,d,path=primary(name,'iid',seed);rows=[]
    for v,y in e.raw.items():
        model,info=Repair.estimate(y,(d['model'][v],d['fit_info'][v]));assert info['improvement']>=-1e-10
        rows.append(dict(variable=v,old_model=d['model'][v],model=model,info=info))
    out=dict(status='COMPLETE',system=name,seed=seed,rows=rows,raw_sha256=d['raw_sha256'],seconds=time.monotonic()-start,source_sha256=source(),input_sha256={str(path):sha(path)})
    dump(ROOT/'repair_regression'/f'{name}_seed{seed}.json',out);print('regression',name,seed,[(r['variable'],Repair.ccgain(r['info'])) for r in rows],flush=True);return dict(kind='regression',name=name,seed=seed)

class Fresh(Legacy.Experiment):
    def estimate(self,fields):
        models={};infos={};old={}
        for v,y in fields.items():
            om,oi=F.estimate(y);models[v],infos[v]=Repair.estimate(y,(om,oi));old[v]=dict(model=om,info=oi)
        self.last_original=old;return models,infos

def fresh(seed):
    start=time.monotonic();e=Fresh('ode',seed,'iid');old_anchor=e.last_original;cov=e.metric(e.pilot,e.model);obs=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False)
    arrays={};fits=[];hashes=[];scores={b:[] for b in e.banks};grams={b:[] for b in e.banks};metrics={b:[] for b in e.banks}
    for v,y in e.pilot.items():arrays['anchor_pilot_'+v]=y
    for b in e.banks:arrays[b+'_observed']=obs[b];arrays[b+'_pilot']=pilot[b];arrays[b+'_covariance']=cov[b]
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,seed]))
    for j in range(128):
        fields,_=e.draw(rng);model,info=e.estimate(fields);mat=e.matrices(fields,model);cv=e.metric(e.pilot,model)
        fits.append(dict(model=model,info=info,original=e.last_original));hashes.append({v:R.digest(y) for v,y in fields.items()})
        for b in e.banks:
            err=mat[b]-pilot[b];assert np.all(err[:,e.deterministic]==0);g=S.gram(err);grams[b].append(g);metrics[b].append(cv[b]);scores[b].append(Legacy.score(e,g,cv[b])[0])
        if (j+1)%32==0:print('fresh ode',seed,j+1,round(time.monotonic()-start,1),flush=True)
    cuts={b:Legacy.cutoff(q) for b,q in scores.items()};cut=Legacy.cutoff(np.maximum(scores['local'],scores['broad']));joint=[B.B.C.J.certificate([S.gram(obs[b]) for b in e.banks],[cov[b] for b in e.banks],cut,c,seed) for c in e.cs]
    for b in e.banks:arrays[b+'_bootstrap_grams']=np.stack(grams[b]);arrays[b+'_bootstrap_covariances']=np.stack(metrics[b]);arrays[b+'_scores']=np.array(scores[b])
    meta=dict(system='ode',profile='iid',seed=seed,model=e.model,fit_info=e.fit_info,original_anchor=old_anchor,pilot_info=e.pilot_info,refits=fits,
              raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},bootstrap_field_sha256=hashes,
              candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,cuts=cuts,joint=dict(cutoff=cut,certificates=joint),source_sha256=source(),input_sha256=e.inputs)
    key=f'ode_iid_seed{seed}';fp=ROOT/'repair_frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN',array_hashes={k:R.digest(v) for k,v in arrays.items()}));fh=sha(fp)
    clean=e.matrices(e.p.data,correct=False);actual={}
    for b in e.banks:arrays[b+'_clean']=clean[b];arrays[b+'_error_gram']=S.gram(obs[b]-clean[b]);actual[b]=Legacy.score(e,arrays[b+'_error_gram'],cov[b])[0]
    cover={b:actual[b]<=cuts[b] for b in e.banks};cover['joint']=max(actual.values())<=cut
    ap=ROOT/'repair_results'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays);assert sha(fp)==fh
    out=dict(**meta,status='COMPLETE',actual_scores=actual,coverage=cover,frozen_sha256=fh,array_sha256=sha(ap),seconds=time.monotonic()-start)
    dump(ap.with_suffix('.json'),out);print('fresh ode done',seed,cover,round(out['seconds'],1),flush=True);return dict(kind='fresh',seed=seed,coverage=cover)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();rows=[]
    dump(ROOT/'repair_launch.json',dict(status='FROZEN',source_sha256=source(),regression_jobs=45,fresh_seeds=[54,55,56],bootstrap=128,metric_draws=16))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(fresh,s) for s in (54,55,56)]+[pool.submit(regression,n,s) for n in B.SYSTEMS for s in (51,52,53)]
        for f in as_completed(tasks):rows.append(f.result())
    dump(ROOT/'repair_process.json',dict(status='COMPLETE',rows=rows,source_sha256=source()))

if __name__=='__main__':main()
