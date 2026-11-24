from adequacy_study import *
from functools import lru_cache
Pilot=Prior.Pilot
BOOTJOBS=[dict(cohort='bootstrap',name='ode',profile='positive',seed=s) for s in range(3200,3456)]

def bsrc():return src('adequacy_fit.py','adjacent_fit.py','adequacy_study.py','selection_bootstrap.py','target_bootstrap.py','bootstrap_protocol.txt','target_protocol.txt')
def bkey(j):return f"ode_{j['profile']}_seed{j['seed']}"
def field(j):return generate(dict(j,cohort='primary'))

def estimate(y):
    m,info,mean=G.estimate(y);screens,choices=B.analyze(y,info);selected=choices['cv_a01'];p=selected['selected_p'];model=None if p is None else [info['full_fits'][str(p)]['sigma'],[info['full_fits'][str(p)]['rho']],[2]];info['full_fits']={p:compact(f) for p,f in info['full_fits'].items()};return model,dict(cv_info=info,screens=screens,selection=selected)

def observe(j):
    y,clean,sigma,rho,inputs=field(j);model,info=estimate(y);key=bkey(j);path=ROOT/'target_anchors'/(key+'.npz');path.parent.mkdir(exist_ok=True);a=dict(raw=y);pinfo=None
    if model is not None:
        e=Point.case('ode',j['profile'],j['seed']);pilot,pinfo=Pilot.reconstruct({'u':y},e.axes,{'u':model});a['pilot']=pilot['u']
    np.savez_compressed(path,**a);dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,job=j,model=model,info=info,pilot_info=pinfo,raw_sha256=R.digest(y),array_sha256=sha(path),source_sha256=bsrc(),input_sha256=inputs));return key

@lru_cache(None)
def load(key):
    p=ROOT/'target_anchors'/(key+'.json');d=read(p);assert sha(p.with_suffix('.npz'))==d['array_sha256']
    with np.load(p.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    e=Point.case('ode',d['job']['profile'],d['job']['seed']);return d,a,e.system_id,e.pid

def score(y,model,info):
    p=info['selection']['selected_p']
    if p is None:return None
    fit=info['cv_info']['full_fits'][str(p)];reg=G.Regression(y,p);return float(H.likelihood(reg,model[0],model[1][0])-reg.df*(fit['objective']+1))

def chunk(key,start):
    d,a,sid,pid=load(key);rng=np.random.default_rng(np.random.SeedSequence([991201,sid,pid,d['job']['seed']]));rows=[]
    for j in range(start+8):
        y=a['pilot']+R.WF.ar_noise(rng,a['pilot'].shape,*d['model'][:2])
        if j<start:continue
        model,info=estimate(y);value=score(y,d['model'],info);rows.append(dict(index=j,model=model,info=info,score=value,score_kind='infinite_abstention' if value is None else 'finite',field_sha256=R.digest(y)))
    path=ROOT/'target_chunks'/f'{key}_start{start:03d}.json';dump(path,dict(status='COMPLETE',key=key,start=start,rows=rows,source_sha256=bsrc(),input_sha256={str(ROOT/'target_anchors'/(key+'.json')):sha(ROOT/'target_anchors'/(key+'.json'))}));return key,start

def freeze(key):
    d,a,sid,pid=load(key);rows=[];inputs={}
    if d['model'] is not None:
        for start in range(0,128,8):
            p=ROOT/'target_chunks'/f'{key}_start{start:03d}.json';rows.extend(read(p)['rows']);inputs[str(p)]=sha(p)
    scores=[float('inf') if r['score'] is None else r['score'] for r in rows];cut=sorted(scores)[125] if scores else None;bad=[r['index'] for r in rows if r['score'] is not None and r['score']<-1e-5];status='unresolved_observed_selection' if d['model'] is None else 'unresolved_negative_statistic' if bad else 'FROZEN';dump(ROOT/'target_frozen'/(key+'.json'),dict(status=status,key=key,cutoff=None if cut is None or not np.isfinite(cut) else cut,cutoff_kind='undefined' if cut is None else 'infinite' if not np.isfinite(cut) else 'finite',abstentions=sum(r['score'] is None for r in rows),negative_statistics=bad,draws=len(rows),source_sha256=bsrc(),input_sha256=inputs));return key

def evaluate(key):
    d,a,sid,pid=load(key);p=ROOT/'target_frozen'/(key+'.json');fr=read(p);y,clean,sigma,rho,inputs=field(d['job']);assert np.array_equal(y,a['raw']);result=dict(status='COMPLETE',key=key,job=d['job'],numerical_status=fr['status'],abstentions=fr['abstentions'],cutoff=fr['cutoff'],cutoff_kind=fr['cutoff_kind'],coverage=dict(nominal=None,calibrated=None),unbounded_certified=dict(nominal=None,calibrated=None),source_sha256=bsrc(),input_sha256={str(p):sha(p)})
    if d['model'] is not None:
        value=score(y,[sigma,[rho],[2]],d['info']);selected=d['info']['selection']['selected_p'];tail=next(r for r in d['info']['screens'] if r['p']==selected)['limit_lr'];result.update(selected_p=selected,lr=value,limit_lr=tail,sigma_ratio=d['model'][0]/sigma,rho_error=d['model'][1][0]-rho)
        if fr['status']=='FROZEN':
            cut=fr['cutoff'] if fr['cutoff_kind']=='finite' else float('inf');result['coverage']=dict(nominal=value<=H.CUTOFF,calibrated=value<=cut);result['unbounded_certified']=dict(nominal=tail<H.CUTOFF-1e-6,calibrated=tail<cut-1e-6)
    dump(ROOT/'target_results'/(key+'.json'),result);return key

def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';dump(ROOT/'target_launch.json',dict(status='FROZEN',jobs=BOOTJOBS,source_sha256=bsrc()));keys=[bkey(j) for j in BOOTJOBS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(observe,j) for j in BOOTJOBS]):print('bootstrap observed',f.result(),flush=True)
        active=sorted([key for key in keys if read(ROOT/'target_anchors'/(key+'.json'))['info']['selection']['selected_p']==24],key=lambda key:read(ROOT/'target_anchors'/(key+'.json'))['job']['seed'])[:8];dump(ROOT/'target_selection.json',dict(status='FROZEN',selected=active,scanned=keys,source_sha256=bsrc(),input_sha256={str(ROOT/'target_anchors'/(key+'.json')):sha(ROOT/'target_anchors'/(key+'.json')) for key in keys}));jobs=[pool.submit(chunk,key,start) for start in range(0,128,8) for key in active]
        for i,f in enumerate(as_completed(jobs),1):
            f.result()
            if i%48==0:print('bootstrap chunks',i,'of',len(jobs),flush=True)
        for f in as_completed([pool.submit(freeze,key) for key in active]):f.result()
        dump(ROOT/'target_all_frozen.json',dict(status='FROZEN',source_sha256=bsrc(),input_sha256={str(ROOT/'target_frozen'/(key+'.json')):sha(ROOT/'target_frozen'/(key+'.json')) for key in active}))
        for f in as_completed([pool.submit(evaluate,key) for key in active]):f.result()
    dump(ROOT/'target_process.json',dict(status='COMPLETE',cases=len(keys),active=len(active),draws=len(active)*128,source_sha256=bsrc()));print('BOOTSTRAP COMPLETE',len(keys),len(active)*128,flush=True)

if __name__=='__main__':main()
