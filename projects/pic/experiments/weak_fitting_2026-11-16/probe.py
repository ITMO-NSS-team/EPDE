"""Failure analysis and raw weak regression with training-only window selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'core15_residual_products_2026-11-16'))
import run_screen as R
import analyze_screen as A
import numpy as np
import json
import time
import argparse


def plan_dict(plan):
    return {k:getattr(plan,k) for k in ('q','cut','ms','ss','shape')}


def raw_moments(bank, observed, pilot, model, count, rng):
    """Fixed-window raw measurement bootstrap; no reconstruction in the draws."""
    original=bank.matrix(observed)
    clean_pilot=bank.matrix(pilot)
    total=np.zeros_like(original)
    gram=np.zeros((original.shape[1],original.shape[1]))
    means={}
    for b in range(1,count+1):
        draw={v:y+R.WF.ar_noise(rng,y.shape,model[v][0],model[v][1]) for v,y in pilot.items()}
        mat=bank.matrix(draw)
        total+=mat;gram+=mat.T@mat
        if b in (count//2,count):
            mean=total/b
            bias=mean-clean_pilot
            shifted=original-bias
            covariance=(gram-b*mean.T@mean)/(b-1)
            corrected=shifted.T@shifted-(1+1/b)*covariance
            means[b]=dict(corrected=corrected,shifted=shifted,bias=bias,covariance=covariance)
    return original,clean_pilot,means


def selector(axes,terms,opaque,observed,width,seed,targets):
    fields=R.WF.Fields(observed,axes,opaque)
    before=time.monotonic()
    chosen=R.WF.choose_window(fields,terms,targets,np.random.default_rng(seed),
        rows_ok=lambda shape:R.SV.row_floor('dyad',shape,width) is None)
    return chosen,time.monotonic()-before


def one(name,seed,group,draws):
    start=time.monotonic()
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p)
    terms,opaque=R.symbols(p,labels);width=max(len(c['terms']) for c in cs);A.common_width(width)
    old=json.loads((R.ROOT/'audited'/f'{name}_seed{seed}_b32.json').read_text())
    indices=np.array(old['groups'][group]['indices']);raw=p.noisy(5,seed)
    observed={v:y[indices] for v,y in raw.items()}
    ax=[axes[0][indices],*axes[1:]]
    opq={k:v[indices] for k,v in opaque.items()}
    pilot,info=R.reconstruct(observed,axes,indices)
    pilot={v:y[indices] for v,y in pilot.items()}
    model={v:R.WF.noise_model(y) for v,y in observed.items()}
    chosen,seconds=selector(ax,terms,opq,observed,width,np.random.SeedSequence([692,seed,group]),
                            sorted({c['yi'] for c in cs}))
    plan=plan_dict(chosen)
    print(name,'selected',chosen,'in',round(seconds,2),'s',flush=True)
    bank=R.LocalBank(ax,terms,opq,plan)
    mat,pm,moments=raw_moments(bank,observed,pilot,model,draws,
        np.random.default_rng(np.random.SeedSequence([693,seed,group])))
    clean_local=bank.matrix({v:y[indices] for v,y in p.data.items()})
    frequencies=[]
    for selection in old['groups'][group]['selection']:
        k=selection['loo']['max_mode']
        frequencies.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*selection['spacing'])
    spectral=R.SpectralBank(axes,terms,opaque,frequencies)
    clean=spectral.matrix(p.data)
    rows=[]
    for i,c in enumerate(cs):
        if c['label']!='reference':continue
        scale=float(np.vdot(clean[:,c['yi']],clean[:,c['yi']]).real)
        def report(coef,diag):
            r=clean[:,c['yi']]-clean[:,c['xi']]@coef
            return dict(coefficients=coef,clean_energy=float(np.vdot(r,r).real/scale),fit=diag)
        methods={}
        for label,matrix in [('weak_ols',mat),('weak_pilot_ols',pm),('weak_clean',clean_local)]:
            methods[label]=report(*R.signed_fit(matrix.T@matrix,matrix,c['xi'],c['yi']))
        for b,entry in moments.items():
            methods[f'weak_clsm_{b}']=report(*R.signed_fit(entry['corrected'],mat,c['xi'],c['yi']))
            methods[f'weak_mean_{b}']=report(*R.signed_fit(entry['shifted'].T@entry['shifted'],mat,c['xi'],c['yi']))
        saved=old['folds'][group]['arms']['loo']['counts']['32'][i]
        methods['old_clsm']=report(np.array(saved['coefficients']),saved['fit'])
        col_error=np.linalg.norm(mat-clean_local,axis=0)/np.maximum(np.linalg.norm(clean_local,axis=0),np.finfo(float).tiny)
        rows.append(dict(slot=c['slot'],reference=c,methods=methods,
                         column_relative_errors={str(labels[j]):col_error[j] for j in [c['yi'],*c['xi']]}))
        print(name,c['slot'],{k:round(v['clean_energy'],7) for k,v in methods.items()},flush=True)
    output=dict(system=name,seed=seed,group=group,draws=draws,plan=plan,window_score=chosen.score,
                window_trials=chosen.table,selection_seconds=seconds,noise=model,rows=rows,
                total_seconds=time.monotonic()-start)
    R.dump(ROOT/'probe_results'/f'{name}_seed{seed}_g{group}_b{draws}.json',output)


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['kdv_cossin','ks','pde_compound'])
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--group',type=int,default=0)
    ap.add_argument('--draws',type=int,default=32);args=ap.parse_args()
    for n in args.systems:one(n,args.seed,args.group,args.draws)


if __name__=='__main__':main()
