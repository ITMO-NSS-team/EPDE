"""Signal adequacy evaluated using correlation from the next richer fit."""
from adequacy_context import *
import adequacy_fit as Own
from scipy.linalg import qr
from scipy.stats import f

def screen(y,p,info):
    fit=info['full_fits'][str(p)];source=min(Own.NEXT[p],64);noise=info['full_fits'][str(source)];n=len(y);idx=np.arange(n);v,base=G.Regression(y,p).evaluate(noise['transformed_rho'],True,True);b=Own.union_basis(n,p);wb=G.whiten(b,idx,noise['rho']);wy=G.whiten(y,idx,noise['rho']);u,rr=qr(wb,mode='economic');res=wy-u@(u.T@wy);q1=float(res@res);q0=base['quadratic'];improvement=q0-q1;assert improvement>=-1e-9*max(q0,1e-300);df1=b.shape[1]-p;df2=n-b.shape[1];stat=max(improvement,0)/df1/(q1/df2);pvalue=float(f.sf(stat,df1,df2));limit=Own.boundary(y,p);nll0=(n-p)*(fit['objective']+1);tail=limit['nll']-nll0
    return dict(p=p,next_p=Own.NEXT[p],rho_source_p=source,screen_rho=noise['rho'],source_fit_status=noise['status'],union_rank=b.shape[1],small_quadratic=q0,large_quadratic=q1,improvement=improvement,df1=df1,df2=df2,f_statistic=stat,pvalue=pvalue,limit=limit,nll0=nll0,limit_lr=tail,boundary_excluded=tail>H.CUTOFF+1e-6,fit_status=fit['status'] if noise['status']=='ok' else 'unresolved_screen_noise')

def analyze(y,info):
    rows=[screen(y,p,info) for p in G.SIZES];return rows,Own.select(info,rows)

def main():
    rows=[]
    for key in ('stress_ode_positive_seed1283','stress_ode_positive_seed1334','stress_ode_positive_seed1456','stress_ode_positive_seed1016'):
        d=read(PREVIOUS/'signal_frozen'/(key+'.json'))
        with np.load(PREVIOUS/'signal_frozen'/(key+'.npz')) as z:y=z['raw']
        screens,choices=analyze(y,d['cv_info']);rows.append(dict(key=key,old_p=d['cv_info']['selected_p'],screens=screens,choices=choices));print(key,{k:v['selected_p'] for k,v in choices.items()},[(x['p'],round(x['pvalue'],4),x['rho_source_p']) for x in screens],flush=True)
    dump(ROOT/'development_adjacent.json',dict(status='COMPLETE',role='Archived development; not validation.',rows=rows,source_sha256=src('adequacy_fit.py','adjacent_fit.py')))

if __name__=='__main__':main()
