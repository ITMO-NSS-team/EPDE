from run_profiles import *
from functools import lru_cache
from scipy.spatial import ConvexHull

JOBS=[(p,s) for s in range(264,272) for p in PROFILES]
def bsrc():return src('bootstrap_regions.py','bootstrap_protocol.txt')
def bkey(p,s):return f'ode_{p}_seed{s}'


def observe(p,s):
    y,truth,design=generate('ode',p,s);model,info=Fit.estimate(y);e,_,_=C.primary('ode','iid',51);pilot,pinfo=Pilot.reconstruct({'u':y},e.axes,{'u':model});key=bkey(p,s);path=ROOT/'bootstrap_anchors'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,raw=y,pilot=pilot['u'])
    dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,profile=p,seed=s,model=model,fit_info=info,pilot_info=pinfo,raw_sha256=R.digest(y),pilot_sha256=R.digest(pilot['u']),array_sha256=sha(path),source_sha256=bsrc()));return key


@lru_cache(None)
def load(key):
    path=ROOT/'bootstrap_anchors'/(key+'.json');d=read(path);assert sha(path.with_suffix('.npz'))==d['array_sha256']
    with np.load(path.with_suffix('.npz')) as a:y=a['raw'];pilot=a['pilot']
    e,_,_=C.primary('ode','iid',51);return d,y,pilot,e.system_id


def chunk(key,start):
    d,y,pilot,sid=load(key);pid,rho=S.PROFILES[d['profile']];rng=np.random.default_rng(np.random.SeedSequence([984102,sid,pid,d['seed']]));rows=[]
    for j in range(start+8):
        field=pilot+R.WF.ar_noise(rng,pilot.shape,*d['model'][:2])
        if j<start:continue
        model,info=Fit.estimate(field);rows.append(dict(draw=j,model=model,info=info,field_sha256=R.digest(field)))
    path=ROOT/'bootstrap_chunks'/f'{key}_start{start:03d}.json';dump(path,dict(status='COMPLETE',key=key,start=start,rows=rows,source_sha256=bsrc(),input_sha256={str(ROOT/'bootstrap_anchors'/(key+'.json')):sha(ROOT/'bootstrap_anchors'/(key+'.json'))}));return dict(key=key,start=start,warnings=sum(r['info']['status']!='ok' for r in rows))


def freeze_region(key):
    d,y,pilot,sid=load(key);rows=[];inputs={}
    for start in range(0,128,8):
        path=ROOT/'bootstrap_chunks'/f'{key}_start{start:03d}.json';rows.extend(read(path)['rows']);inputs[str(path)]=sha(path)
    theta=np.array([np.log(d['model'][0]),np.arctanh(d['model'][1][0])]);samples=np.array([[np.log(r['model'][0]),np.arctanh(r['model'][1][0])] for r in rows]);mean=samples.mean(axis=0);cov=np.cov(samples.T);ev=np.linalg.eigvalsh(cov);status='FROZEN';radius=None
    if ev[0]<=1e-12*ev[-1]:status='unresolved_covariance'
    elif d['fit_info']['status']!='ok' or any(r['info']['status']!='ok' for r in rows):status='unresolved_selected_fit'
    if ev[0]>0:
        centered=samples-mean;dist=np.einsum('bi,ib->b',centered,np.linalg.solve(cov,centered.T));radius=float(sorted(dist)[125])
    out=dict(status=status,key=key,theta=theta,basic_center=2*theta-mean,covariance=cov,samples=samples,refits=rows,empirical_cutoff=radius,nominal_cutoff=NOMINAL,source_sha256=bsrc(),input_sha256=inputs);dump(ROOT/'bootstrap_frozen'/(key+'.json'),out);return key


def evaluate_region(key):
    d,y,pilot,sid=load(key);f=read(ROOT/'bootstrap_frozen'/(key+'.json'));yy,truth,design=generate('ode',d['profile'],d['seed']);assert np.array_equal(y,yy);theta=np.array(f['theta']);center=np.array(f['basic_center']);cov=np.array(f['covariance']);coverage={};regions={}
    for mode,cen,cut in [('uncorrected',theta,NOMINAL),('basic_nominal',center,NOMINAL),('basic_empirical',center,f['empirical_cutoff'])]:
        if f['status']!='FROZEN':coverage[mode]=None;continue
        delta=truth-cen;score=float(delta@np.linalg.solve(cov,delta));rad=np.sqrt(cut*np.diag(cov));lo=cen-rad;hi=cen+rad;coverage[mode]=score<=cut;regions[mode]=dict(center=cen,cutoff=cut,score=score,area_transformed=float(np.pi*cut*np.sqrt(np.linalg.det(cov))),sigma_bounds=[float(np.exp(lo[0])),float(np.exp(hi[0]))],rho_bounds=[float(np.tanh(lo[1])),float(np.tanh(hi[1]))])
    xy=np.vstack([theta,2*theta-np.array(f['samples'])]);hull=ConvexHull(xy);coverage['finite_hull']=bool(np.all(hull.equations[:,:2]@truth+hull.equations[:,2]<=1e-10))
    out=dict(status='COMPLETE',key=key,profile=d['profile'],seed=d['seed'],true_noise=truth,model=d['model'],numerical_status=f['status'],coverage=coverage,regions=regions,sigma_ratio=d['model'][0]/np.exp(truth[0]),rho_error=d['model'][1][0]-np.tanh(truth[1]),basic_center_sigma_ratio=float(np.exp(center[0]-truth[0])),basic_center_rho_error=float(np.tanh(center[1])-np.tanh(truth[1])),source_sha256=bsrc(),input_sha256={str(ROOT/'bootstrap_frozen'/(key+'.json')):sha(ROOT/'bootstrap_frozen'/(key+'.json'))});dump(ROOT/'bootstrap_results'/(key+'.json'),out);print('bootstrap evaluated',key,coverage,flush=True);return key


def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';dump(ROOT/'bootstrap_launch.json',dict(status='FROZEN',jobs=JOBS,workers=8,source_sha256=bsrc()));keys=[bkey(*j) for j in JOBS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(observe,*j) for j in JOBS]):print('bootstrap observed',f.result(),flush=True)
        rows=[]
        for f in as_completed([pool.submit(chunk,key,start) for start in range(0,128,8) for key in keys]):
            rows.append(f.result())
            if len(rows)%24==0:print('bootstrap chunks',len(rows),'of384',rows[-1],flush=True)
        for f in as_completed([pool.submit(freeze_region,key) for key in keys]):f.result()
        frozen={str(ROOT/'bootstrap_frozen'/(key+'.json')):sha(ROOT/'bootstrap_frozen'/(key+'.json')) for key in keys};dump(ROOT/'bootstrap_all_frozen.json',dict(status='FROZEN',input_sha256=frozen,source_sha256=bsrc()))
        for f in as_completed([pool.submit(evaluate_region,key) for key in keys]):f.result()
    dump(ROOT/'bootstrap_process.json',dict(status='COMPLETE',rows=rows,source_sha256=bsrc()))


if __name__=='__main__':main()
