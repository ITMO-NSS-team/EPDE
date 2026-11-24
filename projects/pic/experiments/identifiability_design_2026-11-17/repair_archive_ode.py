from research_context import *
import certified_starts as Repair
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def block(start):
    e,d,path=primary('ode','iid',52);rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]));rows=[]
    with np.load(path.with_suffix('.npz')) as z:pilot={b:z[b+'_pilot'] for b in e.banks}
    for j in range(start+32):
        fields,_=e.draw(rng)
        if j<start:continue
        assert {v:R.digest(y) for v,y in fields.items()}==d['bootstrap_field_sha256'][j];models={};infos={}
        for v,y in fields.items():
            om,oi=F.estimate(y);np.testing.assert_allclose(om[0],d['refits'][j]['model'][v][0],rtol=1e-7)
            models[v],infos[v]=Repair.estimate(y,(om,oi))
        mat=e.matrices(fields,models);cv=e.metric(e.pilot,models);scores={};grams={}
        for b in e.banks:grams[b]=S.gram(mat[b]-pilot[b]);scores[b]=Legacy.score(e,grams[b],cv[b])[0]
        row=dict(draw=j,model=models,info=infos,scores=scores,grams=grams,covariances=cv,field_sha256=d['bootstrap_field_sha256'][j])
        dump(ROOT/'archive_repair_draws'/f'draw{j:03d}.json',dict(**row,source_sha256=sources('repair_archive_ode.py','certified_starts.py','repair_protocol.txt'),input_sha256={str(path):sha(path)}));rows.append(row)
    print('archive block complete',start,flush=True);return rows

def main():
    e,d,path=primary('ode','iid',52);model,info=Repair.estimate(e.raw['u'],(d['model']['u'],d['fit_info']['u']));np.testing.assert_allclose(model[0],e.model['u'][0],rtol=1e-12);np.testing.assert_allclose(model[1],e.model['u'][1],rtol=1e-12)
    dump(ROOT/'archive_repair_launch.json',dict(status='FROZEN',scope='all128 original paired bootstrap fields; observed model unchanged',input_sha256={str(path):sha(path)},source_sha256=sources('repair_archive_ode.py','certified_starts.py')))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(block,s) for s in (0,32,64,96)]):rows.extend(f.result())
    rows.sort(key=lambda r:r['draw']);q={b:[r['scores'][b] for r in rows] for b in e.banks};cuts={b:Legacy.cutoff(v) for b,v in q.items()};cuts['joint']=Legacy.cutoff(np.maximum(q['local'],q['broad']))
    old=dict(d['cuts']['fixed_pilot']);old['joint']=d['joint']['fixed_pilot']['cutoff'];cover={b:d['actual_scores'][b]<=cuts[b] for b in e.banks};cover['joint']=max(d['actual_scores'].values())<=cuts['joint']
    dump(ROOT/'archive_repair_summary.json',dict(status='COMPLETE',cutoffs=cuts,original_cutoffs=old,cutoff_ratios={b:cuts[b]/old[b] for b in cuts},coverage=cover,
         materially_changed_draws=[r['draw'] for r in rows if any(i['contrast_count']*i['improvement']>1e-3 for i in r['info'].values())],
         selected_fit_statuses=[i['status'] for r in rows for i in r['info'].values()],source_sha256=sources('repair_archive_ode.py','certified_starts.py'),input_sha256={str(p):sha(p) for p in (ROOT/'archive_repair_draws').glob('*.json')}))
    print('archive',cuts,old,cover,flush=True)

if __name__=='__main__':main()
