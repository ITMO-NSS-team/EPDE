from common import *
import repair_fit as P
import state_score as F
from scipy.optimize._numdiff import approx_derivative
def main():
    assert read(ROOT/'state_checks.json')['status']=='PASS';count=0;fd_errors=[];deltas=[];links=0
    for path in sorted((ROOT/'state_polish').glob('*.json')):
        saved=read(path)
        for kind in ('source_sha256','input_sha256'):
            for p,h in saved[kind].items():assert sha(p)==h;links+=1
        old=read(ROOT/'repair'/path.name);p=R.datasets.load(old['system']);pid,rho=S.PROFILES[old['profile']];rng=np.random.default_rng(np.random.SeedSequence([970101,pid,old['seed']]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==saved['raw_sha256']
        for row in saved['rows']:
            y=raw[row['variable']];dim=y.ndim;cc=A.Contrasts(y);scale=np.sqrt(sum(np.sum(d*d) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields];u=np.array(row['transformed'])
            def fun(u):
                t,jac=P.inverse(u,dim);value,g,q=F.profile(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:]);return value,g*jac,q
            value,g,q=fun(u);np.testing.assert_allclose(value,row['objective'],rtol=2e-12,atol=2e-12);np.testing.assert_allclose(g,row['gradient'],atol=2e-10,rtol=2e-8)
            np.testing.assert_allclose(row['model'][0],scale*np.sqrt(q/cc.count),rtol=2e-12);assert row['status']=='ok' and row['projected_gradient']<=2e-5
            if row['optimized']:
                numeric=approx_derivative(lambda u:fun(u)[0],u,abs_step=2e-5).ravel();error=float(np.max(np.abs(g-numeric)));np.testing.assert_allclose(g,numeric,rtol=1e-3,atol=2e-6);fd_errors.append(error)
            prior=next(r for r in old['rows'] if r['variable']==row['variable']);deltas.append(dict(record=path.stem,variable=row['variable'],sigma_relative_change=row['model'][0]/prior['model'][0]-1,max_rho_change=float(np.max(np.abs(np.array(row['model'][1])-prior['model'][1]))),banded_minus_state_initial=row['banded_minus_state_initial'],state_objective_improvement=row['initial_objective']-row['objective']));count+=1
    assert count==84 and len(fd_errors)==11
    dump(ROOT/'audit_state.json',dict(status='PASS',variable_replays=count,finite_difference_checks=len(fd_errors),max_finite_difference_error=max(fd_errors),hash_links=links,deltas=deltas,source_sha256=hashes('audit_state.py')));print('PASS',count,len(fd_errors),max(fd_errors),flush=True)
if __name__=='__main__':main()
