from research_context import *
from design_metric import paired as all_metrics

def realify(m):return np.concatenate([m.real,m.imag],axis=-2)

def matrices(e,fields,model=None,correct=True):
    cov=Covariance(e.geometry,model or e.model) if correct else None
    return {b:realify(O.matrix(bank,fields,cov) if correct else bank.matrix(fields)) for b,bank in e.banks.items()}

def menu(e):
    specs=dict(e.specs);q=e.old['base_plan']['q'];full=N.common_spec([e.axes],e.terms,q)
    for k in (0,1,4,6):specs[f'full_k{k}']=N.common_spec([e.axes],e.terms,q,k=k)
    for ft,fx in ((.5,1),(1,.5),(.5,.5),(.25,1),(1,.25),(.25,.25)):
        key=f'fraction_{ft:g}_{fx:g}_k2';r=[full['radii'][0]*ft,full['radii'][1]*fx]
        specs[key]=N.common_spec([e.axes],e.terms,q,radii=r,k=2)
    return specs

def build(e,specs):
    e.specs=specs;e.banks={b:N.Bank(e.axes,e.terms,e.opaque,s) for b,s in specs.items()}

def src(*names):return sources('design_metric.py','design_engine.py','design_protocol.txt',*names)
