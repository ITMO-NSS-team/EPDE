from audit_new import *

def main():
    checks=0;records=0;fields=0
    for p in sorted((ROOT/'repair_regression').glob('*.json')):
        d=read(p);e,old,_=primary(d['system'],'iid',d['seed']);assert d['raw_sha256']==old['raw_sha256']
        for r in d['rows']:exactfit(e.raw[r['variable']],r['model'],r['info']);assert r['info']['improvement']>=-1e-10;checks+=1
        records+=1
    assert records==45 and checks==60
    for p in (ROOT/'repair_characterization').glob('*.json'):
        d=read(p)
        with np.load(ROOT/'noise_inputs'/(d['key']+'.npz')) as z:y=z['field']
        exactfit(y,d['model'],d['info']);checks+=1
    e,old,path=primary('ode','iid',52);rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]));pilot=e.matrices(e.pilot,correct=False);qs={b:[] for b in e.banks}
    for j in range(128):
        d=read(ROOT/'archive_repair_draws'/f'draw{j:03d}.json');ff,_=e.draw(rng);assert {v:R.digest(y) for v,y in ff.items()}==d['field_sha256'];fields+=len(ff)
        for v,y in ff.items():exactfit(y,d['model'][v],d['info'][v]);checks+=1
        for b in e.banks:
            q=eigen_score(np.array(d['grams'][b]),np.array(d['covariances'][b]),e.active);close(q,d['scores'][b]);qs[b].append(d['scores'][b])
        if j in (0,112,127):
            mat=e.matrices(ff,d['model']);cv=V.Legacy.direct_metric(e,e.pilot,d['model'])
            for b in e.banks:close(S.gram(mat[b]-pilot[b]),d['grams'][b]);close(cv[b],d['covariances'][b])
    d=read(ROOT/'archive_repair_summary.json')
    for b in e.banks:assert sorted(qs[b])[122]==d['cutoffs'][b]
    assert sorted(np.maximum(qs['local'],qs['broad']))[122]==d['cutoffs']['joint']
    dump(ROOT/'audit_optimizer.json',dict(status='PASS',regression_records=records,exact_fit_replays=checks,archive_field_hashes=fields,archive_scores=256,source_sha256=sources('audit_optimizer.py','audit_new.py','certified_starts.py')))
    print('optimizer audit PASS',checks,flush=True)

if __name__=='__main__':main()
