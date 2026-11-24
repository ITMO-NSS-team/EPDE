"""Post-search diagnostic with the catalog topology supplied on the two forcing cases."""
import argparse
import json
from scipy.optimize import minimize_scalar

from frequency_trial import CONFIRMATION, ROOT, folder, live
from assess_swap import read_archive, write_csv, np


def diagnose(dataset,seed):
    archive=read_archive(folder(dataset,seed,'refined'))
    problem=live.load_data(dataset)
    raw=problem.noisy(5,seed)
    _,train,held=live.split(live.R.axes_of(problem))
    target,predictors,_=live.R.parse(problem.truth[0])
    labels=[target,*predictors]
    forcing={(int(p['dim']),p['freq']) for term in labels for name,pars in term
             if name in ('sin','cos') for p in [dict(pars)]}
    (axis,truth),=forcing
    options=[r for r in archive['operator']['profiles'] if r['axis']==axis]
    record=min(options,key=lambda r:abs(r['initial']-truth))
    plan={k:tuple(v) for k,v in archive['plan'].items()}
    def matrix(omega,side):
        termspec=[tuple((name,tuple((k,omega if k=='freq' else value) for k,value in pars))
                       for name,pars in term) for term in labels]
        terms,opaque=live.R.symbols(problem,termspec)
        slices,axes=side
        bank=live.R.LocalBank(axes,terms,{k:v[slices] for k,v in opaque.items()},plan)
        return bank.matrix({k:v[slices] for k,v in raw.items()})
    fits={}
    def fit(omega):
        if omega not in fits:
            m=matrix(omega,train)
            x,y=m[:,1:],m[:,0]
            norms=np.linalg.norm(x,axis=0)
            coef=np.linalg.lstsq(x/norms,y,rcond=None)[0]/norms
            residual=y-x@coef
            fits[omega]=(float(residual@residual),coef,float(np.abs(residual).sum()/np.abs(y).sum()))
        return fits[omega]
    initial=record['initial']
    lo,hi=record['lower'],record['upper']
    for w in (initial,lo,hi):
        fit(w)
    res=minimize_scalar(lambda w:fit(w)[0],bounds=(lo,hi),method='bounded',options={'xatol':(hi-lo)*1e-4,'maxiter':24})
    fit(res.x)
    best=min(fits,key=lambda w:(fits[w][0],abs(initial-w)))
    result=dict(dataset=dataset,seed=seed,registered=truth,initial=initial,search_grid=record['installed'],
                topology_profile=best,within_one_percent=int(abs(best-truth)<=.01*abs(truth)))
    result.update(accepted_min=truth-.01*abs(truth),accepted_max=truth+.01*abs(truth),
                  initial_relative_error_pct=100*abs(initial-truth)/abs(truth),
                  installed_relative_error_pct=100*abs(record['installed']-truth)/abs(truth),
                  profile_relative_error_pct=100*abs(best-truth)/abs(truth))
    for key,w in [('initial',initial),('profiled',best),('registered',truth)]:
        sse,coef,source_wape=fit(w)
        m=matrix(w,held)
        result[key+'_train_sse']=sse
        result[key+'_training_wape']=source_wape
        result[key+'_held_wape']=float(np.abs(m[:,0]-m[:,1:]@coef).sum()/np.abs(m[:,0]).sum())
        result[key+'_coefficients']=json.dumps(coef.tolist())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    seeds=CONFIRMATION if args.confirmation else live.SEEDS
    rows=[diagnose(s,n) for s in ('duffing','ode') for n in seeds]
    out=ROOT/'confirmation' if args.confirmation else ROOT
    out.mkdir(exist_ok=True)
    write_csv(out/'known_topology_diagnostic.csv',rows)
