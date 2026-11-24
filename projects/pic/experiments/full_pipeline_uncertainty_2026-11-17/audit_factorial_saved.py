from bridge import *
from concurrent.futures import ProcessPoolExecutor,as_completed
spec=importlib.util.spec_from_file_location('previous_temporal_audit',PREVIOUS/'audit_time.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)

def one(seed):
    name=f'ks_positive_seed{seed}.json';fpath=ROOT/'factorial_frozen'/name;f=read(fpath);d=read(ROOT/'factorial'/name);parent=read(PREVIOUS/'time_validation_frozen'/name);assert sha(fpath)==d['frozen_sha256'];links=Q.links(f)+Q.links(d);p=R.datasets.load('ks');pid,rho=S.PROFILES['positive'];rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==parent['raw_sha256'];model=f['model'];pilots={};pilots['ordinary'],_=R.reconstruct(raw,R.axes_of(p),np.arange(len(R.axes_of(p)[0])));pilots['correlated'],_=Pilot.reconstruct(raw,R.axes_of(p),model);rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]));replays=0
    repro=read(ROOT/'pilot_reproducibility.json');match=next(r for r in repro['comparisons'] if r['seed']==seed and r['matches_original']);path=Path(match['canonical'])
    with np.load(path) as z:saved={v:z[v] for v in z.files}
    for v,y in saved.items():np.testing.assert_allclose(pilots['correlated'][v],y,rtol=0.,atol=1e-9*np.max(np.abs(y)))
    pilots['correlated']=saved
    assert {tag:{v:R.digest(y) for v,y in pil.items()} for tag,pil in pilots.items()}==f['pilot_sha256'],('pilot_hash',seed,{tag:{v:R.digest(y) for v,y in pil.items()} for tag,pil in pilots.items()},f['pilot_sha256'])
    for j in range(16):
        noise={v:R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in raw.items()}
        for pilot in pilots:
            fields={v:y+noise[v] for v,y in pilots[pilot].items()}
            for method in ('time','local6'):
                row=next(r for r in f['rows'] if (r['draw'],r['pilot'],r['method'])==(j,pilot,method));assert {v:R.digest(y) for v,y in fields.items()}==row['field_sha256'],('field_hash',seed,j,pilot,method);assert {v:R.digest(y) for v,y in noise.items()}==row['noise_sha256'],('noise_hash',seed,j,pilot,method)
                if pilot=='ordinary':assert row['fits']==parent['bootstrap'][j][method]
                for v,y in fields.items():
                    if method=='time':Q.temporal(y,*row['fits'][v])
                    else:Q.checkfit(Local,y,*row['fits'][v])
                    replays+=1
    out=dict(status='PASS',seed=seed,score_replays=replays,hash_links=links,input_sha256={str(fpath):sha(fpath),str(ROOT/'factorial'/name):sha(ROOT/'factorial'/name)});dump(ROOT/'factorial_audit_saved'/name,out);return out

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,s) for s in (45,46,47)]):rows.append(f.result())
    dump(ROOT/'audit_factorial_saved.json',dict(status='PASS',records=3,score_replays=sum(r['score_replays'] for r in rows),hash_links=sum(r['hash_links'] for r in rows),rows=rows,source_sha256=sources('audit_factorial_saved.py','pilot_roundoff_protocol.txt')));print(rows,flush=True)
if __name__=='__main__':main()
