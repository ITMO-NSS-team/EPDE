from pipeline import *
from scipy.optimize import linprog

def one(key):
    e,d,anchor=load_case(key);pp=ROOT/'point_frozen'/(key+'.json');ep=ROOT/'envelope_frozen'/(key+'.json');p=read(pp);env=read(ep);assert sha(pp.with_suffix('.npz'))==p['array_sha256']
    with np.load(pp.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    clean=moments(e,e.p.data,correct=False);actual={};coverage={};references=[]
    for b in e.banks:actual[b]=Legacy.score(e,S.gram(a[b+'_observed']-clean[b]),a[b+'_covariance'])[0]
    for mode in p['cuts']:
        coverage[mode]={b:actual[b]<=p['cuts'][mode][b] for b in e.banks};coverage[mode]['joint']=max(actual.values())<=p['joint'][mode]['cutoff']
    coverage['envelope']=({**{b:actual[b]<=env['cutoff'] for b in e.banks},'joint':max(actual.values())<=env['cutoff']} if env['status']=='FROZEN' else None)
    for mode in coverage:
        if coverage[mode] is None:continue
        for b in e.banks:
            cut=env['cutoff'] if mode=='envelope' else p['cuts'][mode][b]
            for c in e.cs:
                if c['label']!='reference':continue
                v=S.E.vector(c,c['exact'],len(e.labels));bound=cut*float(v@a[b+'_covariance']@v);references.append(dict(mode=mode,bank=b,candidate=c['id'],noise_covered=float(v@S.gram(a[b+'_observed']-clean[b])@v)<=bound,zero_accepted=float(v@S.gram(a[b+'_observed'])@v)<=bound))
    points=read(ROOT/'nuisance_points'/(key+'.json'));xy=np.array([r['transformed'] for r in points['points']]);truth=np.array([np.log(.05*np.std(e.p.data['u'])),np.arctanh(e.rho)]);lp=linprog(np.zeros(len(xy)),A_eq=np.vstack([xy.T,np.ones(len(xy))]),b_eq=np.r_[truth,1.],bounds=(0,None),method='highs')
    comparison=None
    if e.profile=='positive' and e.seed in (57,58,59):
        oldpath=PREVIOUS/'results'/(key+'.json');old=read(oldpath);assert old['raw_sha256']==d['raw_sha256'];comparison=dict(old_model=old['model'],old_coverage=old['coverage'],old_geometry={m:[c['geometry'] for c in old['joint'][m]['original']['certificates'] if c['label']=='reference'] for m in old['joint']},input_sha256={str(oldpath):sha(oldpath)})
    out=dict(status='COMPLETE' if env['status']=='FROZEN' else 'COMPLETE_WITH_UNRESOLVED_ENVELOPE',key=key,system='ode',profile=e.profile,seed=e.seed,cohort='paired_previous' if comparison else 'fresh',model=e.model,
        sigma_truth_ratio=e.model['u'][0]/(.05*np.std(e.p.data['u'])),rho_error=e.model['u'][1][0]-e.rho,actual_scores=actual,coverage=coverage,references=references,
        point_geometry={m:[c for c in p['joint'][m]['certificates'] if c['label']=='reference'] for m in p['joint']},envelope_geometry=[c for c in env['certificates'] if c['label']=='reference'],
        envelope_cutoff=env['cutoff'],point_joint_cutoffs={m:x['cutoff'] for m,x in p['joint'].items()},nuisance_truth_in_basic_coordinate_box=bool(np.all(truth>=xy.min(axis=0)) and np.all(truth<=xy.max(axis=0))),nuisance_truth_in_basic_convex_hull=bool(lp.success),old_comparison=comparison,
        selected_fit_warnings=[dict(draw=-1,variable=v,status=i['status']) for v,i in d['fit_info'].items() if i['status']!='ok']+[dict(draw=r['draw'],variable=v,status=i['status']) for r in p['refits'] for v,i in r['info'].items() if i['status']!='ok'],
        source_sha256=src('evaluate.py','envelope.py','envelope_batch.py'),input_sha256={str(pp):sha(pp),str(ep):sha(ep),str(ROOT/'nuisance_points'/(key+'.json')):sha(ROOT/'nuisance_points'/(key+'.json'))})
    dump(ROOT/'results'/(key+'.json'),out);print('evaluated',key,coverage,{m:[c['geometry'] for c in rr] for m,rr in out['point_geometry'].items()},'envelope',[c['geometry'] for c in out['envelope_geometry']],flush=True);return out

def main():
    assert read(ROOT/'envelope_process.json')['status']=='COMPLETE';rows=[one(key_of(*j)) for j in JOBS];dump(ROOT/'evaluation_process.json',dict(status='COMPLETE',records=[r['key'] for r in rows],source_sha256=src('evaluate.py')))

if __name__=='__main__':main()
