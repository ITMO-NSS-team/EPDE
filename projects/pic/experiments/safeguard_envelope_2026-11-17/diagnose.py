from envelope import *
import audit_new as A
from scipy.linalg import eigh
from collections import Counter


def union_case(key):
    e,d,a=load_case(key);p=read(ROOT/'point_frozen'/(key+'.json'));env=read(ROOT/'envelope_frozen'/(key+'.json'));cs=[c for c in e.cs if c['label']=='reference'];assert len(cs)==1;c=cs[0]
    rows=[];components=[cert for v in p['joint'].values() for cert in v['certificates'] if cert['candidate']==c['id']]
    for start in range(0,129,4):
        path=ROOT/'envelope_chunks'/f'{key}_start{start:03d}.json';chunk=read(path);assert sha(path.with_suffix('.npz'))==chunk['array_sha256']
        with np.load(path.with_suffix('.npz')) as z:aa={k:z[k] for k in z.files}
        for row in chunk['rows']:
            if row['status']!='ok':rows.append(dict(index=row['index'],status='unresolved'));continue
            pos=list(aa['indices']).index(row['index']);mm=moments(e,e.raw,row['model']);gg=[S.gram(mm[b]) for b in e.banks];cv=[aa[b+'_covariance'][pos] for b in e.banks]
            cert=B.B.C.J.certificate(gg,cv,row['conditional_cutoff'],c,e.seed);A.certcheck(cert,gg,cv,row['conditional_cutoff'],c)
            components.append(cert);rows.append(dict(index=row['index'],status='ok',certificate=cert))
    states=Counter(x['geometry'] for x in components)
    if states['unbounded_certified']:geometry='unbounded_certified'
    elif len(components)==131 and all(x['geometry'] in ('bounded_outer','empty_certified') for x in components):geometry='bounded_outer'
    else:geometry='unresolved'
    box=None
    if geometry=='bounded_outer':
        rr=[x['outer_region'] for x in components if x['geometry']=='bounded_outer'];box=dict(lower=np.min([x['lower'] for x in rr],axis=0),upper=np.max([x['upper'] for x in rr],axis=0))
    out=dict(status='COMPLETE',key=key,geometry=geometry,component_counts=dict(states),outer_box=box,points=rows,primary_components=components[:2],source_sha256=src('diagnose.py','diagnostic_protocol.txt','envelope.py','envelope_batch.py'),input_sha256={str(ROOT/'point_frozen'/(key+'.json')):sha(ROOT/'point_frozen'/(key+'.json')),str(ROOT/'envelope_frozen'/(key+'.json')):sha(ROOT/'envelope_frozen'/(key+'.json'))})
    dump(ROOT/'union_diagnostic'/(key+'.json'),out);print('scenario union',key,geometry,dict(states),flush=True);return dict(key=key,geometry=geometry)


def failure():
    key='ode_negative_seed60';e,d,a=load_case(key);p=read(ROOT/'point_frozen'/(key+'.json'));model={'u':[float(.05*np.std(e.p.data['u'])),[e.rho],e.model['u'][2]]}
    clean=moments(e,e.p.data,correct=False);mm={'estimated':moments(e,e.raw),'known':moments(e,e.raw,model)};cv={'estimated':{b:a[b+'_covariance'] for b in e.banks},'known':covariance(e,model)};scores={}
    for correction,mat in mm.items():
        scores[correction]={metric:{b:A.eigen_score(S.gram(mat[b]-clean[b]),c[b],e.active) for b in e.banks} for metric,c in cv.items()}
    meta,aa=point(e,a,dict(index=-1,status='ok',model=model));directions={}
    for b in e.banks:
        ix=np.ix_(e.active,e.active);g=S.gram(mm['estimated'][b]-clean[b])[ix];s=cv['estimated'][b][ix];sc=np.sqrt(np.diag(s));values,vectors=eigh(g/sc[:,None]/sc,s/sc[:,None]/sc);v=vectors[:,-1]/sc
        full=np.zeros(len(e.labels));full[e.active]=v;A.close(float(full@S.gram(mm['estimated'][b]-clean[b])@full),values[-1]);A.close(float(full@cv['estimated'][b]@full),1.)
        c=next(c for c in e.cs if c['label']=='reference');rv=S.E.vector(c,c['exact'],len(e.labels))
        directions[b]=dict(score=float(values[-1]),unit_covariance_vector=full,labels=e.labels,reference_score=float(rv@S.gram(mm['estimated'][b]-clean[b])@rv/(rv@cv['estimated'][b]@rv)),correction_shift_norm=float(np.linalg.norm((mm['estimated'][b]-mm['known'][b])@full)),known_error_norm_same_direction=float(np.linalg.norm((mm['known'][b]-clean[b])@full)))
    out=dict(status='COMPLETE',key=key,known_model=model,scores=scores,original_cuts=p['cuts'],known_conditional_cutoff=meta['conditional_cutoff'],known_conditional_coverage={b:scores['known']['known'][b]<=meta['conditional_cutoff'] for b in e.banks},directions=directions,source_sha256=src('diagnose.py','diagnostic_protocol.txt','envelope.py','envelope_batch.py'))
    dump(ROOT/'failure_diagnostic.json',out);print('failure diagnostic',scores,'known cutoff',meta['conditional_cutoff'],flush=True)


def main():
    assert read(ROOT/'evaluation_process.json')['status']=='COMPLETE';failure()
    with ProcessPoolExecutor(max_workers=8) as pool:rows=[f.result() for f in as_completed([pool.submit(union_case,key_of(*j)) for j in JOBS])]
    dump(ROOT/'diagnostic_process.json',dict(status='COMPLETE',rows=rows,source_sha256=src('diagnose.py','diagnostic_protocol.txt')))


if __name__=='__main__':main()
