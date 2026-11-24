"""Evaluate paired pilot metrics with shared Gaussian/Wick noise features."""
from bridge import *
from batch_moments import Fields,integrate
from itertools import combinations
from collections import Counter

def paired(e,pilots,model,mc=16):
    pilot_fields={tag:Fields({v:y[None] for v,y in pil.items()},e.axes,e.opaque) for tag,pil in pilots.items()};basefields=next(iter(pilot_fields.values()));sums={tag:{b:0. for b in e.banks} for tag in pilots};cov=Covariance(e.geometry,model);plans={};products={tag:{} for tag in pilots}
    for bank in e.banks.values():
        for term,monos in zip(bank.terms,bank.monos):
            for data,coords,orders in monos:
                if len(data)>3:raise ValueError('Exact Wick expansion implemented through cubic degree')
                if data not in plans:
                    counts=Counter()
                    for k in range(1,len(data)+1):
                        for sub in combinations(range(len(data)),k):counts[(tuple(data[j] for j in sub),tuple(data[j] for j in range(len(data)) if j not in sub))]+=1
                    plans[data]=counts
    for start in range(0,mc,4):
        values={v:[] for v in model};batch=min(4,mc-start)
        for j in range(start,start+batch):
            rng=np.random.default_rng(np.random.SeedSequence([984105,e.system_id,e.pid,j]))
            for v in model:values[v].append(R.WF.ar_noise(rng,e.raw[v].shape,*model[v][:2]))
        noise=Fields({v:np.stack(y) for v,y in values.items()},e.axes,e.opaque);wick={};shared={};points={tag:{} for tag in pilots}
        def hermite(data):
            if data not in wick:
                z=noise.get(*data[0]).copy()
                for atom in data[1:]:z=z*noise.get(*atom)
                for i,j in combinations(range(len(data)),2):
                    term=cov.pair(data[i],data[j])
                    for k,atom in enumerate(data):
                        if k not in (i,j):term=term*noise.get(*atom)
                    z=z-term
                wick[data]=z
            return wick[data]
        for tag,pf in pilot_fields.items():
            for b,bank in e.banks.items():
                integrals={};columns=[]
                for term,monos in zip(bank.terms,bank.monos):
                    column=0.
                    for (data,coords,orders),factor in monos.items():
                        key=data,coords,term.opaque;cache=shared if len(data)<=1 else points[tag]
                        if key not in cache:
                            if not data:arr=np.zeros((batch,)+e.raw[next(iter(e.raw))].shape)
                            else:
                                arr=0.
                                for (nd,pd),mult in plans[data].items():
                                    if pd not in products[tag]:products[tag][pd]=pf.point(pd,(0,)*len(e.axes),()) if pd else 1.
                                    arr=arr+mult*hermite(nd)*products[tag][pd]
                                arr=arr*basefields.point((),coords,term.opaque)
                            cache[key]=arr
                        ik=key,orders
                        if ik not in integrals:integrals[ik]=integrate(bank,cache[key],orders)
                        column=column+factor*integrals[ik]
                    columns.append(np.asarray(column).reshape(batch,-1))
                m=np.stack(columns,axis=-1);m=e.basis.convert(m) if b=='broad' else m.real;m[:,:,e.deterministic]=0.;sums[tag][b]=sums[tag][b]+np.einsum('nri,nrj->ij',m,m)/mc
    return sums
