"""Post-selection algebraic decomposition using clean saved-data controls."""
import run_information as I
B,L,E,Q,F,R,A,np=I.B,I.L,I.E,I.Q,I.F,I.R,I.A,I.np
import json,argparse


def run(name,seed):
    _,old=F.baseline(name,seed);b=old['draws'];p=R.datasets.load(name);axes=R.axes_of(p)
    cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels);ref=next(c for c in cs if c['label']=='reference')
    xi,yi=ref['xi'],ref['yi'];truth=np.array(ref['exact'])
    broad=json.loads((B.ROOT/'results'/f'{name}_seed{seed}.json').read_text())
    plans=json.loads((F.ROOT/'fit_results'/f'{name}_seed{seed}.json').read_text())
    frozen=F.evaluation_grams(p,old,terms,opaque,labels);fields=E.reconstruct_groups(p,old);rows=[]
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,plans['plans'][g]['plan'],axes,opaque)
        banks={'local':Q.Bank(ax,terms,opq,plan=plan)}
        for extra in (0,1):
            chosen=broad['folds'][g]['selection'][str(extra)]
            if not chosen['feasible']:continue
            pp=chosen['plan'];k=chosen['k'];signed=np.r_[np.arange(k+1),-np.arange(1,k+1)]
            freq=[signed*np.pi/(m*(a[1]-a[0])) for m,a in zip(pp['ms'],ax)]
            banks[f'broad{extra}']=Q.Bank(ax,terms,opq,plan=pp,frequencies=freq)
        h=frozen[f'{g}_loo_oracle'][np.ix_(xi,xi)]/frozen[f'{g}_loo_clean_scale'][yi]
        for tag,bank in banks.items():
            gg,nn,mm=I.bank_moments(bank,fields[g],segment);noise=nn[b]
            z=2*mm[0]-mm[1:].mean(axis=0);gc=gg-noise
            coef,fit=R.signed_fit(gc,mm[0],xi,yi)
            clean=bank.matrix({v:y[segment] for v,y in p.data.items()})
            xx=z[:,xi];xc=clean[:,xi];dy=z[:,yi]-clean[:,yi];dx=xx-xc
            rhs=dict(target=np.real(xx.conj().T@dy),
                features=np.real(xx.conj().T@(-dx@truth)),
                clean_discretization=np.real(xx.conj().T@(clean[:,yi]-xc@truth)),
                covariance_correction=-noise[xi,yi]+noise[np.ix_(xi,xi)]@truth)
            parts={}
            for part,rr in rhs.items():
                adjusted=gc.copy();adjusted[xi,yi]=rr;adjusted[yi,xi]=rr
                delta,_=R.signed_fit(adjusted,mm[0],xi,yi)
                parts[part]=dict(coefficient_change=delta,common_clean_impact=float(delta@h@delta))
            total=sum(v['coefficient_change'] for v in parts.values())
            if fit['rank']==len(xi):np.testing.assert_allclose(total,coef-truth,atol=3e-7,rtol=3e-7)
            rows.append(dict(training=g,bank=tag,coefficients=coef,exact_coefficients=truth,fit=fit,parts=parts,
                observed_coefficient_error=coef-truth,decomposition_error=float(np.max(np.abs(total-(coef-truth)))),
                target_relative_error=float(np.linalg.norm(dy)/max(np.linalg.norm(clean[:,yi]),np.finfo(float).tiny)),
                feature_relative_errors=np.linalg.norm(dx,axis=0)/np.maximum(np.linalg.norm(xc,axis=0),np.finfo(float).tiny),
                worst_feature_noise=I.worst_noise(gg,noise,xi)))
    R.dump(I.ROOT/'error_budget'/f'{name}_seed{seed}.json',dict(system=name,seed=seed,rows=rows))
    print(name,seed,'decomposition verified',flush=True)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    for n in args.systems:
        for s in args.seeds:run(n,s)


if __name__=='__main__':main()
