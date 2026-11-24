from bridge import *
import paired_study_fast as Base
import batch_moments as Batch
import time
from math import ceil

MC=16
BOOTSTRAP=128
class Experiment(Base.Study):
    def __init__(self,name,seed,profile):
        self.system_id=SYSTEMS.index(name)
        super().__init__(name,seed,profile)
        self.pilot,self.pilot_info=Pilot.reconstruct(self.raw,self.axes,self.model)
    def estimate(self,fields):
        models={};infos={}
        for v,y in fields.items():models[v],infos[v]=Temporal.estimate(y)
        return models,infos
    def true_field(self,b,anchor=False):
        domain=984101 if anchor else 984104;rng=np.random.default_rng(np.random.SeedSequence([domain,self.system_id,self.pid,self.seed,b]))
        return {v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[self.rho]*y.ndim) for v,y in self.p.data.items()}
    def metric(self,pilot,model,mc=MC):
        baseline=self.matrices(pilot,correct=False);parts={b:[] for b in self.banks}
        for j in range(0,mc,4):
            stack={v:[] for v in pilot}
            for i in range(j,min(j+4,mc)):
                rng=np.random.default_rng(np.random.SeedSequence([984105,self.system_id,self.pid,i]))
                for v,y in pilot.items():stack[v].append(y+R.WF.ar_noise(rng,y.shape,*model[v][:2]))
            for bank,m in Batch.matrices(self,{v:np.stack(y) for v,y in stack.items()},model).items():
                error=m-baseline[bank][None];error[:,:,self.deterministic]=0.;parts[bank].append(error)
        return {bank:np.einsum('nri,nrj->ij',z,z)/mc for bank,ll in parts.items() for z in [np.concatenate(ll)]}

def score(e,gram,cov):
    w,info=S.metric(cov,e.active)
    if w is None:raise ValueError(info)
    return float(S.operator_scores(gram[None],w,e.active)[0]),info

def cutoff(values):
    # Finite Monte Carlo rank correction; bootstrap exchangeability is not assumed.
    rank=min(len(values),ceil(.95*(len(values)+1)));return float(np.sort(values)[rank-1])

def source_files():return {**sources('full_pipeline.py','coverage_decision.txt','run_full.py'),**{str(PREVIOUS/n):sha(PREVIOUS/n) for n in ('batch_moments.py','paired_study_fast.py')}}

def run(name,profile,seed):
    start=time.monotonic();e=Experiment(name,seed,profile);key=f'{name}_{profile}_seed{seed}';src=source_files();warnings=[]
    def fit_warnings(info,stream,draw):
        for v,i in info.items():
            if i['status']!='ok':warnings.append(dict(stream=stream,draw=draw,variable=v,status=i['status']))
    fit_warnings(e.fit_info,'observed',-1);anchor_cov=e.metric(e.pilot,e.model);observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,correct=False);arrays={};fits=[];metrics={mode:{b:[] for b in e.banks} for mode in ('fixed_pilot','refit_pilot')};scores={mode:{b:[] for b in e.banks} for mode in metrics};grams={b:[] for b in e.banks};pilot_hashes=[]
    for b in e.banks:arrays[b+'_observed']=observed[b];arrays[b+'_pilot']=pilot[b];arrays[b+'_anchor_covariance']=anchor_cov[b]
    rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,seed]));field_hashes=[]
    for j in range(BOOTSTRAP):
        fields,_=e.draw(rng);model,info=e.estimate(fields);fit_warnings(info,'bootstrap',j);newpilot,newinfo=Pilot.reconstruct(fields,e.axes,model);fit=dict(model=model,info={v:{k:i[k] for k in ('status','objective','projected_gradient')} for v,i in info.items()},pilot_info=newinfo);fits.append(fit);pilot_hashes.append({v:R.digest(y) for v,y in newpilot.items()});field_hashes.append({v:R.digest(y) for v,y in fields.items()});mat=e.matrices(fields,model)
        covariance={'fixed_pilot':e.metric(e.pilot,model),'refit_pilot':e.metric(newpilot,model)}
        for b in e.banks:
            g=S.gram(mat[b]-pilot[b]);assert np.all((mat[b]-pilot[b])[:,e.deterministic]==0);grams[b].append(g)
            for mode in covariance:
                cv=covariance[mode][b];q,mi=score(e,g,cv);metrics[mode][b].append(cv);scores[mode][b].append(q)
        if (j+1)%16==0:print(key,'bootstrap',j+1,round(time.monotonic()-start,1),flush=True)
    cuts={};regions={};joint={}
    for b in e.banks:arrays[b+'_bootstrap_grams']=np.stack(grams[b])
    for mode in scores:
        cuts[mode]={};regions[mode]={}
        for b in e.banks:
            arrays[mode+'_'+b+'_covariances']=np.stack(metrics[mode][b]);arrays[mode+'_'+b+'_scores']=np.array(scores[mode][b]);cut=cuts[mode][b]=cutoff(scores[mode][b]);g=S.gram(observed[b]);regions[mode][b]=[dict(candidate=c['id'],label=c['label'],**S.quadratic_region(g-cut*anchor_cov[b],g,anchor_cov[b],c,cut)) for c in e.cs]
        cut=cutoff(np.maximum(scores[mode]['local'],scores[mode]['broad']));joint[mode]=dict(cutoff=cut,certificates=[B.C.J.certificate([S.gram(observed[b]) for b in e.banks],[anchor_cov[b] for b in e.banks],cut,c,seed) for c in e.cs])
    meta=dict(system=name,profile=profile,seed=seed,bootstrap=BOOTSTRAP,metric_draws=MC,model=e.model,fit_info=e.fit_info,pilot_info=e.pilot_info,raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},refits=fits,bootstrap_field_sha256=field_hashes,refitted_pilot_sha256=pilot_hashes,candidates=e.cs,labels=e.labels,active=e.active,deterministic=e.deterministic,cuts=cuts,regions=regions,joint=joint,warnings=warnings,source_sha256=src,input_sha256=e.inputs)
    fp=ROOT/'frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN',array_hashes={k:R.digest(v) for k,v in arrays.items()}));fh=sha(fp)
    # Only now evaluate clean-field errors; the method and all cutoffs are fixed.
    clean=e.matrices(e.p.data,correct=False);actual_scores={};refs=[];counts={}
    for b in e.banks:
        arrays[b+'_clean']=clean[b];err=S.gram(observed[b]-clean[b]);arrays[b+'_error_gram']=err;actual_scores[b]=score(e,err,anchor_cov[b])[0]
        for c in e.cs:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(e.labels));variance=float(v@anchor_cov[b]@v);z=float(v@S.gram(observed[b])@v);noise=float(v@err@v)
            for mode in cuts:refs.append(dict(mode=mode,bank=b,candidate=c['id'],noise_covered=noise<=cuts[mode][b]*variance,zero_accepted=z<=cuts[mode][b]*variance))
    for mode in cuts:counts[mode]={b:actual_scores[b]<=cuts[mode][b] for b in e.banks};counts[mode]['both']=max(actual_scores.values())<=joint[mode]['cutoff']
    ap=ROOT/'results'/(key+'.npz');ap.parent.mkdir(exist_ok=True);assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(ap,**arrays);assert sha(fp)==fh
    out=dict(**meta,status='COMPLETE_WITH_WARNINGS' if warnings else 'COMPLETE',actual_scores=actual_scores,coverage=counts,references=refs,frozen_sha256=fh,array_sha256=sha(ap),seconds=time.monotonic()-start);dump(ap.with_suffix('.json'),out);print(key,out['status'],counts,round(out['seconds'],1),flush=True)
