"""Matched diffusion-positive and reaction-only records on the live search's three window geometries."""
import csv
import json
from types import SimpleNamespace
from search import ROOT,ARMS,SEEDS,np
from transfer import LAWS,tokens
from blocked import fit,wls,block_specs
from study import R
from local_holdouts import fixture


def measure(seed):
    saved=json.loads((ROOT/'results'/f'ac_{seed}_tls.json').read_text())
    plan={k:tuple(v) for k,v in saved['geometry']['native'].items()}
    axes,positive=fixture('ac',False)
    _,null=fixture('ac',True)
    shape=positive['u'].shape
    ref,target=tokens(LAWS['ac'])
    reduced=tuple(term for term in ref if term[0][0]!='d^2u/dx1^2')
    assert len(ref)==4 and len(reduced)==3
    problem=SimpleNamespace(shape=shape,grids=np.meshgrid(*axes,indexing='ij'),data=positive,named_arrays={})
    terms,opaque=R.symbols(problem,ref)
    assert not opaque
    banks=[(s,R.LocalBank([a[z] for a,z in zip(axes,s)],terms,{},p)) for s,p in
           [(tuple(slice(None) for _ in axes),plan),*block_specs(shape,plan,saved['geometry']['axis'])]]
    out=[]
    for j in range(16):
        noise=np.random.default_rng([985104,seed,j]).standard_normal(shape)
        for case,fields in [('positive',positive),('zero',null)]:
            raw=fields['u']+.05*fields['u'].std()*noise
            mats=[b.matrix({'u':raw[s]}) for s,b in banks]
            for arm in ARMS:
                vals=[]
                for support in (ref,reduced):
                    ids=[ref.index(k) for k in support]
                    p=support.index(target)
                    if arm=='wape':
                        m=mats[0][:,ids]
                        v,status=wls(m,p)
                        assert status=='ok'
                        val=np.abs(m@(v/v[p])).sum()/np.abs(m[:,p]).sum()
                    else:
                        held=[]
                        for g in (1,2):
                            m=mats[g][:,ids]
                            v,status=wls(m,p) if arm=='wls' else fit(m)
                            assert status=='ok'
                            scale=np.linalg.norm(np.linalg.norm(m,axis=0)*v)/np.sqrt(len(m))
                            held.append(np.mean((mats[3-g][:,ids]@(v/scale))**2))
                        val=np.mean(held)
                    vals.append(float(val))
                out.append(dict(geometry_seed=seed,repeat=j,case=case,arm=arm,full=vals[0],reaction=vals[1],margin=vals[1]-vals[0]))
    return out


if __name__=='__main__':
    rows=[r for seed in SEEDS for r in measure(seed)]
    with (ROOT/'ac_controls.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for a in ARMS:
        print(a,{c:sum(r['margin']>0 for r in rows if r['arm']==a and r['case']==c) for c in ('positive','zero')})
