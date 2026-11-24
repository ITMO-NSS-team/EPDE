from bootstrap_regions import *
import envelope as E
import audit_new as A


def psrc():return {**E.envsrc(),**src('bootstrap_regions.py','bootstrap_protocol.txt','propagate.py','propagation_protocol.txt')}
def getcase(key):
    d,y,pilot,sid=load(key);e,_,_=C.primary('ode','iid',51);e.profile=d['profile'];e.seed=d['seed'];e.pid,e.rho=S.PROFILES[e.profile];e.raw={'u':y};e.pilot={'u':pilot};e.model={'u':d['model']};return e,d


def primary(key):
    e,d=getcase(key);r=read(ROOT/'bootstrap_frozen'/(key+'.json'));assert r['status']=='FROZEN',(key,r['status']);cov=Old.covariance(e,e.model);obs=Old.moments(e,e.raw);pil=Old.moments(e,e.pilot,correct=False);a={};gg={b:[] for b in e.banks};cc={b:[] for b in e.banks};pid,rho=S.PROFILES[e.profile];rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,pid,e.seed]))
    for b in e.banks:a[b+'_observed']=obs[b];a[b+'_pilot']=pil[b];a[b+'_covariance']=cov[b]
    for j,fit in enumerate(r['refits']):
        fields={'u':e.pilot['u']+R.WF.ar_noise(rng,e.pilot['u'].shape,*d['model'][:2])};assert R.digest(fields['u'])==fit['field_sha256'];model={'u':fit['model']};mat=Old.moments(e,fields,model);cv=Old.covariance(e,model)
        for b in e.banks:gg[b].append(S.gram(mat[b]-pil[b]));cc[b].append(cv[b])
    cuts={};certs={};scores={}
    for b in e.banks:a[b+'_bootstrap_grams']=np.stack(gg[b]);a[b+'_bootstrap_covariances']=np.stack(cc[b])
    for mode in ('studentized','fixed_metric'):
        ss={b:np.array([Old.Legacy.score(e,g,c)[0] for g,c in zip(gg[b],cc[b] if mode=='studentized' else [cov[b]]*128)]) for b in e.banks};cut=Old.Legacy.cutoff(np.maximum(ss['local'],ss['broad']));cuts[mode]=cut
        for b in e.banks:a[mode+'_'+b+'_scores']=ss[b]
        certs[mode]=[Old.B.B.C.J.certificate([S.gram(obs[b]) for b in e.banks],[cov[b] for b in e.banks],cut,c,e.seed) for c in e.cs]
    theta=np.array(r['theta']);center=np.array(r['basic_center']);vals,vec=np.linalg.eigh(r['covariance']);root=(vec*np.sqrt(vals))@vec.T;rad=np.sqrt(r['empirical_cutoff']);points=[center,theta]
    for radius in (.5,1.):
        for angle in np.arange(32)*2*np.pi/32:points.append(center+radius*rad*root@np.array([np.cos(angle),np.sin(angle)]))
    points=[dict(index=j,status='ok',transformed=t,model={'u':[float(np.exp(t[0])),[float(np.tanh(t[1]))],[2]]}) for j,t in enumerate(points)]
    for p in points:
        if not np.isfinite(p['model']['u'][0]) or p['model']['u'][0]<=0 or abs(p['model']['u'][1][0])>=1:p['status']='invalid_nuisance_point'
    path=ROOT/'propagation_primary'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**a)
    dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,candidates=e.cs,labels=e.labels,active=e.active,cuts=cuts,certificates=certs,points=points,array_sha256=sha(path),source_sha256=psrc(),input_sha256={str(ROOT/'bootstrap_frozen'/(key+'.json')):sha(ROOT/'bootstrap_frozen'/(key+'.json'))}));return key


@lru_cache(None)
def primary_arrays(key):
    p=ROOT/'propagation_primary'/(key+'.json');d=read(p);assert sha(p.with_suffix('.npz'))==d['array_sha256']
    with np.load(p.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    return d,a


def scenario_chunk(key,start):
    e,d=getcase(key);p,a=primary_arrays(key);c=next(c for c in e.cs if c['label']=='reference');rows=[];stacks={};indices=[]
    for item in p['points'][start:start+3]:
        row,arrays=E.point(e,a,item)
        if arrays is not None:
            obs=Old.moments(e,e.raw,item['model']);grams=[S.gram(obs[b]) for b in e.banks];cv=[arrays[b+'_covariance'] for b in e.banks];cert=Old.B.B.C.J.certificate(grams,cv,row['conditional_cutoff'],c,e.seed);A.certcheck(cert,grams,cv,row['conditional_cutoff'],c);row['reference_certificate']=cert
            for b in e.banks:arrays[b+'_observed']=obs[b]
            indices.append(item['index'])
            for name,v in arrays.items():stacks.setdefault(name,[]).append(v)
        rows.append(row)
    path=ROOT/'propagation_chunks'/f'{key}_start{start:03d}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,indices=np.array(indices),**{k:np.stack(v) for k,v in stacks.items()})
    dump(path.with_suffix('.json'),dict(status='COMPLETE',key=key,start=start,rows=rows,array_sha256=sha(path),source_sha256=psrc(),input_sha256={str(ROOT/'propagation_primary'/(key+'.json')):sha(ROOT/'propagation_primary'/(key+'.json'))}));return dict(key=key,start=start)


def freeze_union(key):
    e,d=getcase(key);p,a=primary_arrays(key);rows=[];inputs={}
    for start in range(0,66,3):
        path=ROOT/'propagation_chunks'/f'{key}_start{start:03d}.json';rows.extend(read(path)['rows']);inputs[str(path)]=sha(path)
    c=next(c for c in e.cs if c['label']=='reference');components=[next(cert for cert in cc if cert['label']=='reference') for cc in p['certificates'].values()];components += [r['reference_certificate'] for r in rows if r['status']=='ok'];status='FROZEN' if all(r['status']=='ok' for r in rows) else 'UNRESOLVED';box=None
    if any(c['geometry']=='unbounded_certified' for c in components):geometry='unbounded_certified'
    elif len(components)==68 and all(c['geometry']=='bounded_outer' for c in components):
        geometry='bounded_outer';box=dict(lower=np.min([c['outer_region']['lower'] for c in components],axis=0),upper=np.max([c['outer_region']['upper'] for c in components],axis=0))
    else:geometry='unresolved'
    cut=max(list(p['cuts'].values())+[b['bound'] for r in rows if r['status']=='ok' for b in r['bounds'].values()]);scalar=Old.B.B.C.J.certificate([S.gram(a[b+'_observed']) for b in e.banks],[a[b+'_covariance'] for b in e.banks],cut,c,e.seed);A.certcheck(scalar,[S.gram(a[b+'_observed']) for b in e.banks],[a[b+'_covariance'] for b in e.banks],cut,c)
    dump(ROOT/'propagation_frozen'/(key+'.json'),dict(status=status,key=key,geometry=geometry,outer_box=box,scalar_cutoff=cut,scalar_certificate=scalar,components=components,points=rows,source_sha256=psrc(),input_sha256=inputs));return key


def evaluate(key):
    e,d=getcase(key);p,a=primary_arrays(key);f=read(ROOT/'propagation_frozen'/(key+'.json'));clean=Old.moments(e,e.p.data,correct=False);actual={b:A.eigen_score(S.gram(a[b+'_observed']-clean[b]),a[b+'_covariance'],e.active) for b in e.banks};coverage={m:max(actual.values())<=cut for m,cut in p['cuts'].items()};coverage['scalar']=max(actual.values())<=f['scalar_cutoff'];coverage['finite_union']=any(coverage[m] for m in p['cuts']);covering=[];best=float('inf');c=next(c for c in e.cs if c['label']=='reference');v=S.E.vector(c,c['exact'],len(e.labels));accepted=False
    for start in range(0,66,3):
        path=ROOT/'propagation_chunks'/f'{key}_start{start:03d}.npz';meta=read(path.with_suffix('.json'));assert sha(path)==meta['array_sha256']
        with np.load(path) as z:
            for row in meta['rows']:
                if row['status']!='ok':continue
                pos=list(z['indices']).index(row['index']);ratios=[A.eigen_score(S.gram(z[b+'_observed'][pos]-clean[b]),z[b+'_covariance'][pos],e.active)/row['conditional_cutoff'] for b in e.banks];best=min(best,max(ratios))
                if max(ratios)<=1:covering.append(row['index'])
                accepted=accepted or all(v@S.gram(z[b+'_observed'][pos])@v<=row['conditional_cutoff']*float(v@z[b+'_covariance'][pos]@v) for b in e.banks)
    coverage['finite_union']=coverage['finite_union'] or bool(covering);accepted=accepted or any(all(v@S.gram(a[b+'_observed'])@v<=cut*float(v@a[b+'_covariance']@v) for b in e.banks) for cut in p['cuts'].values())
    out=dict(status='COMPLETE',key=key,profile=e.profile,seed=e.seed,coverage=coverage,actual_scores=actual,best_conditional_ratio=best,covering_conditional_points=covering,reference_accepted=accepted,primary_geometry={m:next(c['geometry'] for c in cc if c['label']=='reference') for m,cc in p['certificates'].items()},finite_union_geometry=f['geometry'],scalar_geometry=f['scalar_certificate']['geometry'],source_sha256=psrc(),input_sha256={str(ROOT/'propagation_frozen'/(key+'.json')):sha(ROOT/'propagation_frozen'/(key+'.json'))});dump(ROOT/'propagation_results'/(key+'.json'),out);print('propagation evaluated',key,coverage,f['geometry'],flush=True);return key


def main():
    assert read(ROOT/'bootstrap_process.json')['status']=='COMPLETE';keys=[bkey(*j) for j in JOBS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(primary,key) for key in keys]):print('primary frozen',f.result(),flush=True)
        rows=[]
        for f in as_completed([pool.submit(scenario_chunk,key,start) for start in range(0,66,3) for key in keys]):
            rows.append(f.result())
            if len(rows)%48==0:print('propagation chunks',len(rows),'of528',flush=True)
        for f in as_completed([pool.submit(freeze_union,key) for key in keys]):f.result()
        assert all(read(ROOT/'propagation_frozen'/(key+'.json'))['status']=='FROZEN' for key in keys),'Unresolved scenario retained; do not score a partial union as complete.'
        dump(ROOT/'propagation_all_frozen.json',dict(status='FROZEN',input_sha256={str(ROOT/'propagation_frozen'/(key+'.json')):sha(ROOT/'propagation_frozen'/(key+'.json')) for key in keys},source_sha256=psrc()))
        for f in as_completed([pool.submit(evaluate,key) for key in keys]):f.result()
    dump(ROOT/'propagation_process.json',dict(status='COMPLETE',source_sha256=psrc()))


if __name__=='__main__':main()
