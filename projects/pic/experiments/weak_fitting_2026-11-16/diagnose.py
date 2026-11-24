"""Distinguish coefficient, quadrature and held-out score failures."""
import run_fitting as F
R,A,np=F.R,F.A,F.np
import json
import argparse


def score_diagnostics(result):
    p=R.datasets.load(result['system']);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    _,old=F.baseline(result['system'],result['seed'])
    grams=F.evaluation_grams(p,old,terms,opaque,labels)
    output=[]
    for c in cs:
        if c['label']!='reference':continue
        i=next(i for i,d in enumerate(cs) if d['id']==c['id'])
        truth=[]
        for g in range(3):truth.append(F.scores([c],[(np.array(c['exact']),{})],grams,f'{g}_loo_')[0])
        exact={m:float(np.mean([t[m] for t in truth])) for m in ('cross','ordinary','clean_energy')}
        entry=dict(system=p.name,seed=result['seed'],slot=c['slot'],exact_coefficients=exact,methods={})
        for method in ['old_clsm','raw_ols','smooth_ols',f"raw_clsm_{old['draws']}",f"smooth_clsm_{old['draws']}"]:
            a=result['aggregates']['loo'][method]
            dominates=a['cross']['dominance'][c['slot']]['unregistered']
            competitors=[]
            for cid in dominates:
                j=next(j for j,d in enumerate(cs) if d['id']==cid)
                competitors.append(dict(id=cid,
                    discrepancy_gap=a['cross']['scores'][j]-a['cross']['scores'][i],
                    clean_energy_gap=a['clean_energy']['scores'][j]-a['clean_energy']['scores'][i],
                    instability_gap=result['instability'][j]-result['instability'][i]))
            entry['methods'][method]=dict(reference_cross=a['cross']['scores'][i],
                reference_clean_energy=a['clean_energy']['scores'][i],competitors=competitors)
        output.append(entry)
    return output


def quadrature(result):
    p=R.datasets.load(result['system']);axes=R.axes_of(p)
    cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels);_,old=F.baseline(p.name,result['seed'])
    grams=F.evaluation_grams(p,old,terms,opaque,labels)
    raw=p.noisy(5,result['seed']);rows=[]
    for g in range(3):
        idx=np.array(old['groups'][g]['indices']);ax=[axes[0][idx],*axes[1:]]
        opq={k:v[idx] for k,v in opaque.items()};plan={k:tuple(v) for k,v in result['plans'][g]['plan'].items()}
        bank=R.LocalBank(ax,terms,opq,plan)
        segment=slice(idx[0],idx[-1]+1)
        fax=[axes[0][segment],*axes[1:]];fopq={k:v[segment] for k,v in opaque.items()}
        fp={k:((v[0]*3,)+v[1:] if k in ('ms','ss','cut') else v) for k,v in plan.items()}
        fine=R.LocalBank(fax,terms,fopq,fp)
        variants={}
        for name,values in [('clean',p.data),('noisy',raw)]:
            observed={v:y[idx] for v,y in values.items()}
            reconstructed,_=R.reconstruct(observed,axes,idx)
            variants[name+'_raw_coarse']=bank.matrix(observed)
            variants[name+'_smooth_coarse']=bank.matrix({v:y[idx] for v,y in reconstructed.items()})
            variants[name+'_smooth_fine']=fine.matrix({v:y[segment] for v,y in reconstructed.items()})
        for c in cs:
            if c['label']!='reference':continue
            entry=dict(system=p.name,seed=result['seed'],slot=c['slot'],group=g,methods={})
            for name,mat in variants.items():
                fit=R.signed_fit(mat.T@mat,mat,c['xi'],c['yi'])
                entry['methods'][name]=F.scores([c],[fit],grams,f'{g}_loo_')[0]
            rows.append(entry)
    return rows


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ks','kdv_cossin','pde_compound','ac','ns'])
    args=ap.parse_args();s=[];q=[]
    for name in args.systems:
        for seed in range(3):
            x=json.loads((F.ROOT/'fit_results'/f'{name}_seed{seed}.json').read_text())
            s.extend(score_diagnostics(x))
        x=json.loads((F.ROOT/'fit_results'/f'{name}_seed0.json').read_text())
        rr=quadrature(x);q.extend(rr)
        for slot in sorted({r['slot'] for r in rr}):
            subset=[r for r in rr if r['slot']==slot]
            print(name,slot,{m:float(np.mean([r['methods'][m]['clean_energy'] for r in subset]))
                  for m in subset[0]['methods']},flush=True)
    R.dump(F.ROOT/'score_diagnostics.json',s);R.dump(F.ROOT/'quadrature_diagnostics.json',q)


if __name__=='__main__':main()
