"""PCR selector with enough nodes for its declared polynomial moments.

The weak quadrature promises degree 2q moments. A coarse half-window needs
at least q points on either side (2q+1 nodes), so m_fine/2 >= q. This is a
quadrature dimension requirement, not a noise threshold or per-system knob.
"""
import run_fitting as F
R,P,np=F.R,F.P,F.np
from itertools import product
import time
import argparse
from pathlib import Path
import hashlib


def selector(axes,terms,opaque,observed,width,seed,targets):
    started=time.monotonic();WF=R.WF
    fields=WF.Fields(observed,axes,opaque);D=len(axes)
    monos=[t.monomials() for t in terms];opq=[t.opaque for t in terms]
    r=np.max([WF.psi_orders(m,D) for m in monos],axis=0).astype(int)
    left=np.max([WF.leftover_orders(m,D) for m in monos],axis=0).astype(int)
    cut=tuple(2*WF.FD_REACH*left);cut_s=tuple(c//2 for c in cut)
    coarse=fields.every_other();rng=np.random.default_rng(seed)
    values={v:y.copy() for v,y in observed.items()}
    for v,y in observed.items():
        sigma,rhos,_=WF.noise_model(y);values[v]=y+WF.ar_noise(rng,y.shape,sigma,rhos)
    drawn=fields.with_values(values)
    n=[n-2*c for n,c in zip(fields.shape,cut)]
    nc=[n-2*c for n,c in zip(coarse.shape,cut_s)]
    lattice=[]
    for n_ in n:
        ms=[];m=2
        while 2*m+1<=n_:ms.append(m);m*=2
        lattice.append(ms)
    table=[];excluded=0;caches=({}, {}, {});js=list(WF.J_START);done=set()
    while True:
        for j in js:
            if j in done:continue
            done.add(j);q=tuple(r+j)
            for ms in product(*lattice):
                msc=tuple(max(1,m//2) for m in ms)
                if any(m<qq for m,qq in zip(msc,q)):
                    excluded+=1;continue
                stride=WF._stride(n,ms,len(terms),lambda shape:R.SV.row_floor('dyad',shape,width) is None)
                if stride is None:continue
                ss,shape=stride
                if any(nn<2*m+1 for nn,m in zip(nc,msc)):continue
                if np.prod(WF.lattice_shape(n,ms,ms))<=len(terms)+1:continue
                mat,_=WF.weak_matrix(monos,opq,fields,cut,ms,ms,q,caches[0])
                draw,_=WF.weak_matrix(monos,opq,drawn,cut,ms,ms,q,caches[1])
                cm,_=WF.weak_matrix(monos,opq,coarse,cut_s,msc,msc,q,caches[2])
                gc=cm.T@cm if len(cm)>cm.shape[1] else np.full((len(terms),len(terms)),np.nan)
                err=draw-mat;g,gd,gn=mat.T@mat,draw.T@draw,err.T@err
                score=float(np.median([WF.pcr_score(g,gd,gc,gn,t) for t in targets]))
                table.append((j,ms,ss,shape,score))
        if not table:raise ValueError(f'No quadrature-complete window on {fields.shape}')
        best=min(table,key=lambda x:(x[4],x[0]))
        if not np.isfinite(best[4]):raise ValueError('No finite complete-quadrature PCR window')
        top=max(js)
        if best[0]<top or not any(x[0]==top and np.isfinite(x[4]) for x in table):break
        if top+1>WF.J_GUARD:raise RuntimeError('Quadrature smoothness guard reached')
        js.append(top+1)
    j,ms,ss,shape,score=best
    selected=WF.WindowPlan(j,r,ms,ss,cut,shape,score,table)
    assert all(m//2>=q for m,q in zip(selected.ms,selected.q))
    return selected,time.monotonic()-started


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=R.datasets.names('core'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    # Local experiment injection; production WF.choose_window is untouched.
    P.selector=selector
    output=F.ROOT/'complete_results'
    for name in args.systems:
        for seed in args.seeds:
            path=output/f'{name}_seed{seed}.json'
            if path.exists():print('SKIP',path.name,flush=True);continue
            F.run_case(name,seed,output)
            # Record the selector's implementation as well as the fit engine.
            import json
            saved=json.loads(path.read_text(encoding='utf-8'))
            saved['selector_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            saved['quadrature_complete']=True
            R.dump(path,saved)


if __name__=='__main__':main()
