import time_pilot_study as E
from common import *
import time,argparse
DRAWS=256
MODES=('fixed','refit')
def brief(info):return {v:{k:i[k] for k in ('status','objective','projected_gradient')} for v,i in info.items()}
def run(name,seed,profile):
    start=time.monotonic();e=E.Study(name,seed,profile);key=f'{name}_{profile}_seed{seed}';src=E.sources();warnings=[]
    meta=dict(system=name,seed=seed,profile=profile,draws=DRAWS,model=e.model,fit_info=e.fit_info,raw_sha256={v:R.digest(y) for v,y in e.raw.items()},source_sha256=src,input_sha256=e.inputs)
    for v,i in e.fit_info.items():
        if i['status']!='ok':warnings.append(dict(stream='anchor',draw=-1,variable=v,status=i['status']))
    observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);arrays={};fits={};field_hashes={};metrics={};regions={};joint={}
    for b in e.banks:arrays[b+'_observed']=observed[b];arrays[b+'_pilot']=pilot[b]
    for stream,domain in [('calibration',983102),('check',983103)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.pid,seed]));gg={(mode,b):[] for mode in MODES for b in e.banks};fits[stream]=[];field_hashes[stream]=[]
        for j in range(DRAWS):
            fields,noise=e.draw(rng);model,info=e.estimate(fields);fits[stream].append(dict(model=model,info=brief(info)))
            for v,i in info.items():
                if i['status']!='ok':warnings.append(dict(stream=stream,draw=j,variable=v,status=i['status']))
            if j in (0,DRAWS-1):field_hashes[stream].append({v:R.digest(y) for v,y in fields.items()})
            for mode,m in [('fixed',e.model),('refit',model)]:
                for b,a in e.matrices(fields,m).items():
                    error=a-pilot[b];assert np.all(error[:,e.deterministic]==0);gg[mode,b].append(S.gram(error))
        for (mode,b),g in gg.items():arrays[f'{mode}_{b}_{stream}_grams']=np.stack(g)
        print(key,stream,'DONE',round(time.monotonic()-start,1),flush=True)
    for mode in MODES:
        metrics[mode]={};regions[mode]={}
        for b in e.banks:
            prefix=f'{mode}_{b}';sigma=arrays[prefix+'_calibration_grams'].mean(axis=0);g=S.gram(observed[b]);w,mi=S.metric(sigma,e.active);assert w is not None,mi
            q=S.operator_scores(arrays[prefix+'_check_grams'],w,e.active);cut=float(np.quantile(q,.95))
            arrays[prefix+'_sigma']=sigma;arrays[prefix+'_white']=w;arrays[prefix+'_check_scores']=q;arrays[prefix+'_observed_gram']=g
            metrics[mode][b]=dict(mi,cutoff=cut);regions[mode][b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*sigma,g,sigma,c,cut)) for c in e.cs]
        cut=float(np.quantile(np.maximum(arrays[f'{mode}_local_check_scores'],arrays[f'{mode}_broad_check_scores']),.95))
        joint[mode]=dict(cutoff=cut,certificates=[C.J.certificate([arrays[f'{mode}_{b}_observed_gram'] for b in e.banks],[arrays[f'{mode}_{b}_sigma'] for b in e.banks],cut,c,seed) for c in e.cs])
    fp=ROOT/'time_pilot_frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN',pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},specs=e.specs,candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,
        metrics=metrics,regions=regions,joint=joint,refits=fits,field_hashes=field_hashes,array_hashes={k:R.digest(v) for k,v in arrays.items()},warnings=warnings));fh=sha(fp)
    clean=e.matrices(e.p.data,correct=False);actual={(mode,b):[] for mode in MODES for b in e.banks};errors={k:[] for k in actual};fits['actual']=[];field_hashes['actual']=[]
    for b in e.banks:arrays[b+'_clean']=clean[b]
    for j in range(DRAWS):
        fields=e.true_field(j);model,info=e.estimate(fields);fits['actual'].append(dict(model=model,info=brief(info)))
        for v,i in info.items():
            if i['status']!='ok':warnings.append(dict(stream='actual',draw=j,variable=v,status=i['status']))
        if j in (0,DRAWS-1):field_hashes['actual'].append({v:R.digest(y) for v,y in fields.items()})
        for mode,m in [('fixed',e.model),('refit',model)]:
            for b,a in e.matrices(fields,m).items():actual[mode,b].append(S.gram(a));errors[mode,b].append(S.gram(a-clean[b]))
    stats=[]
    for mode,b in actual:
        prefix=f'{mode}_{b}';g=np.stack(actual[mode,b]);err=np.stack(errors[mode,b]);arrays[prefix+'_actual_grams']=g;arrays[prefix+'_actual_error_grams']=err
        q=S.operator_scores(err,arrays[prefix+'_white'],e.active);arrays[prefix+'_actual_scores']=q;cut=metrics[mode][b]['cutoff'];sigma=arrays[prefix+'_sigma'];refs=[]
        for c in e.cs:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(e.labels));bound=cut*float(v@sigma@v);check=np.einsum('i,bij,j->b',v,arrays[prefix+'_check_grams'],v);threshold=float(np.quantile(check,.95));z=np.einsum('i,bij,j->b',v,g,v);noise=np.einsum('i,bij,j->b',v,err,v)
            refs.append(dict(candidate=c['id'],anchor_accepted=bool(v@arrays[prefix+'_observed_gram']@v<=bound),pointwise_coverage_count=int(np.sum(noise<=threshold)),uniform_coverage_count=int(np.sum(noise<=bound)),zero_acceptance_count=int(np.sum(z<=bound))))
        stats.append(dict(mode=mode,bank=b,operator_coverage_count=int(np.sum(q<=cut)),cutoff=cut,references=refs))
    for mode in MODES:
        joint[mode]['operator_coverage_count']=int(np.sum(np.maximum(arrays[f'{mode}_local_actual_scores'],arrays[f'{mode}_broad_actual_scores'])<=joint[mode]['cutoff']))
    assert all(np.isfinite(a).all() for a in arrays.values());ap=ROOT/'time_pilot_results'/(key+'.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays);assert sha(fp)==fh
    status='COMPLETE_WITH_FIT_WARNINGS' if warnings else 'COMPLETE'
    dump(ap.with_suffix('.json'),dict(**meta,status=status,metrics=metrics,regions=regions,joint=joint,statistics=stats,refits=fits,field_hashes=field_hashes,warnings=warnings,frozen_sha256=fh,array_sha256=sha(ap),seconds=time.monotonic()-start))
    print(key,status,[(r['mode'],r['bank'],r['operator_coverage_count']) for r in stats],round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system');p.add_argument('seed',type=int);p.add_argument('profile');a=p.parse_args();run(a.system,a.seed,a.profile)
