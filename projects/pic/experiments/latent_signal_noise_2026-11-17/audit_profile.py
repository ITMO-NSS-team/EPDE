from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
import audit_followups as A
F,C,S,R,np=A.F,A.C,A.S,A.R,A.np

def main():
    d=json.loads((ROOT/'profile_diagnostic.json').read_text());assert d['status']=='COMPLETE' and len(d['rows'])==9
    links=A.links(d);count=0;cached={}
    for row in d['rows']:
        key=(row['system'],row['seed'],row['profile'])
        if key not in cached:cached[key]=C.Study(*key,'iid')
        e=cached[key];y=e.raw[row['variable']];dim=y.ndim;cc=F.Contrasts(y);scale=np.sqrt(sum(float(np.sum(x*x)) for x in cc.fields)/cc.count);cc.fields=[x/scale for x in cc.fields]
        def evaluate(theta):
            q,ld=cc.evaluate(theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:]);return float(np.log(q/cc.count)+ld/cc.count),scale*np.sqrt(q/cc.count)
        for fit in row['fixed_rho_fits']:
            value,sigma=evaluate(np.r_[np.full(dim,e.rho),fit['theta']]);np.testing.assert_allclose(value,fit['value'],atol=1e-12,rtol=1e-12);count+=1
        good=[f for f in row['fixed_rho_fits'] if f['success'] and f['value']<1e40];best=min(good,key=lambda f:f['value']);assert best['value']==row['fixed_objective']
        value,sigma=evaluate(np.r_[np.full(dim,e.rho),best['theta']]);np.testing.assert_allclose(sigma,row['fixed_sigma'],rtol=1e-12)
        value,_=evaluate(np.asarray(row['free_restart']['theta']));np.testing.assert_allclose(value,row['free_restart']['value'],atol=1e-12,rtol=1e-12);count+=1
        assert row['fixed_minus_original']==row['fixed_objective']-row['original']['objective']
        assert row['restart_minus_original']==row['free_restart']['value']-row['original']['objective']
    R.dump(ROOT/'audit_profile.json',dict(status='PASS',fields=len(d['rows']),objective_replays=count,hash_links=links,
        source_sha256={str(Path(__file__)):S.E.sha(__file__),str(ROOT/'audit_followups.py'):S.E.sha(ROOT/'audit_followups.py')},input_sha256={str(ROOT/'profile_diagnostic.json'):S.E.sha(ROOT/'profile_diagnostic.json')}))
    print('PASS',len(d['rows']),count)

if __name__=='__main__':main()
