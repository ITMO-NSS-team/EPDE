from pipeline import *
import envelope_batch as Batch
from scipy.optimize import minimize_scalar
import traceback

def envsrc():return src('envelope.py','envelope_batch.py')

def prepare(key):
    path=ROOT/'point_frozen'/(key+'.json');d=read(path);points=[];base=d['model']['u'];theta=np.array([np.log(base[0]),np.arctanh(base[1][0])])
    candidates=[theta]+[2*theta-np.array([np.log(r['model']['u'][0]),np.arctanh(r['model']['u'][1][0])]) for r in d['refits']]
    for j,t in enumerate(candidates):
        with np.errstate(over='ignore',invalid='ignore'):sigma=float(np.exp(t[0]));rho=float(np.tanh(t[1]))
        valid=np.isfinite(sigma) and sigma>0 and np.isfinite(rho) and abs(rho)<1
        points.append(dict(index=j,transformed=t,status='ok' if valid else 'invalid_nuisance_point',model={'u':[sigma,[rho],base[2]]} if valid else None))
    dump(ROOT/'nuisance_points'/(key+'.json'),dict(status='FROZEN',key=key,points=points,source_sha256=envsrc(),input_sha256={str(path):sha(path)}));return key

def scalar_bound(e,dg,cv,anchor,cut):
    w,info=S.metric(anchor,e.active);assert w is not None;ix=np.ix_(e.active,e.active);g=w.T@dg[ix]@w;s=w.T@cv[ix]@w;g=(g+g.T)/2;s=(s+s.T)/2
    def fun(logt):
        tau=np.exp(logt);h=(1+tau)*g+(1+1/tau)*cut*s;return float(np.linalg.eigvalsh((h+h.T)/2)[-1])
    opt=minimize_scalar(fun,bounds=(-10.,10.),method='bounded',options=dict(xatol=1e-10));logt=min((-10.,10.,float(opt.x)),key=fun);tau=float(np.exp(logt));value=fun(logt);h=(1+tau)*g+(1+1/tau)*cut*s
    minimum=float(np.linalg.eigvalsh(value*np.eye(len(e.active))-h)[0]);assert minimum>=-1e-8*max(1,value)
    return dict(bound=value,tau=tau,log_tau=logt,minimum_whitened_slack=minimum)

def point(e,a,p):
    if p['status']!='ok':return dict(index=p['index'],status=p['status']),None
    model=p['model'];cv=covariance(e,model);observed=moments(e,e.raw,model);gg={b:[] for b in e.banks};hashes=[];deterministic_roundoff=0.
    for start in range(0,128,16):
        values={v:[] for v in e.raw}
        for j in range(start,start+16):
            rng=np.random.default_rng(np.random.SeedSequence([985802,e.system_id,e.pid,e.seed,j]));fields={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in e.pilot.items()}
            if j in (0,127):hashes.append({v:R.digest(y) for v,y in fields.items()})
            for v,y in fields.items():values[v].append(y)
        mm=Batch.matrices(e,{v:np.stack(y) for v,y in values.items()},model)
        for b in e.banks:
            # These columns depend only on coordinates. Use the canonical
            # single-field values after checking batch summation roundoff.
            target=a[b+'_pilot'][:,e.deterministic]
            discrepancy=float(np.max(abs(mm[b][:,:,e.deterministic]-target[None])))
            assert discrepancy<=1e-12*max(1.,float(np.max(abs(target))))
            deterministic_roundoff=max(deterministic_roundoff,discrepancy)
            mm[b][:,:,e.deterministic]=target[None]
            error=mm[b]-a[b+'_pilot'][None];assert np.all(error[:,:,e.deterministic]==0);gg[b].append(np.einsum('bri,brj->bij',error,error))
    scores={};arrays={}
    for b in e.banks:
        arrays[b+'_grams']=np.concatenate(gg[b]);arrays[b+'_covariance']=cv[b];arrays[b+'_correction_gram']=S.gram(a[b+'_observed']-observed[b]);w,info=S.metric(cv[b],e.active)
        if w is None:raise ValueError(info)
        scores[b]=S.operator_scores(arrays[b+'_grams'],w,e.active);arrays[b+'_scores']=scores[b]
    cut=float(np.sort(np.maximum(scores['local'],scores['broad']))[125]);bounds={b:scalar_bound(e,arrays[b+'_correction_gram'],cv[b],a[b+'_covariance'],cut) for b in e.banks}
    return dict(index=p['index'],status='ok',model=model,conditional_cutoff=cut,bounds=bounds,field_hashes=hashes,deterministic_roundoff=deterministic_roundoff),arrays

def chunk_env(key,start):
    e,d,a=load_case(key);ppath=ROOT/'nuisance_points'/(key+'.json');p=read(ppath);rows=[];stacks={};indices=[]
    for item in p['points'][start:start+4]:
        try:r,arrays=point(e,a,item)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,AssertionError):r=dict(index=item['index'],status='unresolved_numerical_point',error=traceback.format_exc());arrays=None
        rows.append(r)
        if arrays is not None:
            indices.append(item['index'])
            for k,v in arrays.items():stacks.setdefault(k,[]).append(v)
    arrays={k:np.stack(v) for k,v in stacks.items()};arrays['indices']=np.array(indices,dtype=int);ap=ROOT/'envelope_chunks'/f'{key}_start{start:03d}.npz';ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays)
    dump(ap.with_suffix('.json'),dict(status='COMPLETE',key=key,start=start,rows=rows,array_sha256=sha(ap),source_sha256=envsrc(),input_sha256={str(ppath):sha(ppath)}));return dict(key=key,start=start,unresolved=sum(r['status']!='ok' for r in rows))

def finish(key):
    e,d,a=load_case(key);pointpath=ROOT/'point_frozen'/(key+'.json');p=read(pointpath);rows=[];inputs={str(pointpath):sha(pointpath)}
    for start in range(0,129,4):
        path=ROOT/'envelope_chunks'/f'{key}_start{start:03d}.json';r=read(path);rows.extend(r['rows']);inputs[str(path)]=sha(path)
    assert [r['index'] for r in rows]==list(range(129));bad=[r['index'] for r in rows if r['status']!='ok'];base=max(v['cutoff'] for v in p['joint'].values())
    if bad:out=dict(status='UNRESOLVED',unresolved_points=bad,cutoff=None,certificates=[])
    else:
        cut=max([base]+[v['bound'] for r in rows for v in r['bounds'].values()]);cert=[B.B.C.J.certificate([S.gram(a[b+'_observed']) for b in e.banks],[a[b+'_covariance'] for b in e.banks],cut,c,e.seed) for c in e.cs];out=dict(status='FROZEN',unresolved_points=[],cutoff=cut,point_cutoff_maximum=base,certificates=cert)
    dump(ROOT/'envelope_frozen'/(key+'.json'),dict(**out,key=key,points=rows,source_sha256=envsrc(),input_sha256=inputs));print('envelope frozen',key,out['status'],out['cutoff'],flush=True);return dict(key=key,status=out['status'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();assert read(ROOT/'point_process.json')['status']=='COMPLETE';assert read(ROOT/'envelope_checks.json')['status']=='PASS';keys=[key_of(*j) for j in JOBS]
    for key in keys:prepare(key)
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(chunk_env,key,start) for start in range(0,129,4) for key in keys]):
            records.append(f.result())
            if len(records)%24==0:print('envelope chunks',len(records),'of396',flush=True)
        final=[f.result() for f in as_completed([pool.submit(finish,key) for key in keys])]
    dump(ROOT/'envelope_process.json',dict(status='COMPLETE',rows=final,chunks=records,source_sha256=envsrc()))

if __name__=='__main__':main()
