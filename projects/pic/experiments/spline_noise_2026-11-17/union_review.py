"""Account for certified empty components without rewriting frozen results."""
from spline_propagate import *


def main():
    assert read(ROOT/'propagation_audit.json')['status']=='PASS';rows=[]
    for key in [bkey(*j) for j in JOBS]:
        e,d=getcase(key);p,a=primary_arrays(key);path=ROOT/'propagation_frozen'/(key+'.json');u=read(path);certs=u['components'];bounded=[c for c in certs if c['geometry']=='bounded_outer'];empty=[j for j,c in enumerate(certs) if c['geometry']=='empty_certified'];geometry=u['geometry'];box=u['outer_box'];ref=next(c for c in e.cs if c['label']=='reference')
        for idx in empty:
            assert idx>=2;point=p['points'][idx-2];chunk=(idx-2)//3*3;ap=ROOT/'propagation_chunks'/f'{key}_start{chunk:03d}.npz';meta=read(ap.with_suffix('.json'));assert sha(ap)==meta['array_sha256']
            with np.load(ap) as z:
                pos=list(z['indices']).index(idx-2);mm=Old.moments(e,e.raw,point['model']);cv=[z[b+'_covariance'][pos] for b in e.banks];A.certcheck(certs[idx],[S.gram(mm[b]) for b in e.banks],cv,u['points'][idx-2]['conditional_cutoff'],ref)
        if bounded and len(bounded)+len(empty)==len(certs):
            geometry='bounded_outer';box=dict(lower=np.min([c['outer_region']['lower'] for c in bounded],axis=0),upper=np.max([c['outer_region']['upper'] for c in bounded],axis=0))
        elif len(empty)==len(certs):geometry='empty_certified';box=None
        out=dict(status='PASS',key=key,numerical_status=p['numerical_status'],original_geometry=u['geometry'],geometry=geometry,outer_box=box,empty_components=empty,bounded_components=len(bounded),total_components=len(certs),reason='A certified empty set contributes nothing to a union. Unresolved sets are not omitted.',source_sha256=src('union_review.py'),input_sha256={str(path):sha(path),str(ROOT/'propagation_audit.json'):sha(ROOT/'propagation_audit.json')});dump(ROOT/'union_reviews'/(key+'.json'),out);rows.append(out)
    dump(ROOT/'union_review.json',dict(status='PASS',rows=rows,source_sha256=src('union_review.py')));print('UNION REVIEW PASS',[(r['key'],r['original_geometry'],r['geometry'],len(r['empty_components'])) for r in rows if r['empty_components']],flush=True)


if __name__=='__main__':main()
