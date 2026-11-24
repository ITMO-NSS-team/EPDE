from context import *
import audit_new as A
from types import MethodType
import argparse

def survey():
    rows=[]
    for p in sorted((ROOT/'survey').glob('*.json')):
        d=read(p);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.profile=d['profile'];e.pid,e.rho=S.PROFILES[e.profile];raw=e.true_field(0,True);assert {v:R.digest(y) for v,y in raw.items()}==d['raw_sha256']
        A.exactfit(raw['u'],d['model']['u'],d['fit_info']['u']);assert sha(ROOT/'survey_frozen'/p.name)==d['frozen_sha256'];truth=.05*np.std(e.p.data['u']);A.close(d['sigma_ratio']['u'],d['model']['u'][0]/truth);A.close(d['rho_error']['u'],d['model']['u'][1][0]-e.rho)
        if d['seed'] in (100,115,131):
            e.raw=raw;e.model=d['model'];e.pilot,_=Pilot.reconstruct(raw,e.axes,e.model);e.matrices=MethodType(lambda obj,fields,model=None,correct=True:matrices(obj,fields,model,correct),e);clean=e.matrices(e.p.data,correct=False)
            for label,model in [('fitted',e.model),('oracle_noise',d['truth_for_evaluation'])]:
                mat=e.matrices(raw,model);cv=A.V.Legacy.direct_metric(e,e.pilot,model)
                for b in e.banks:A.close(A.eigen_score(S.gram(mat[b]-clean[b]),cv[b],e.active),d['scores'][b][label])
        rows.append(dict(record=p.name,status='PASS',input_sha256={str(p):sha(p)}))
    assert len(rows)==96;dump(ROOT/'audit_survey.json',dict(status='PASS',rows=rows,source_sha256=sources('audit_auxiliary.py')))

def guard():
    rows=[]
    for p in sorted((ROOT/'guard_results').glob('*.json')):
        d=read(p);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.pid,e.rho=S.PROFILES[d['profile']];y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256'];assert sha(ROOT/'guard_frozen'/p.name)==d['frozen_sha256']
        A.exactfit(y,d['model'],d['info']);assert d['info']['objective']<=d['old_info']['objective']+1e-12
        A.close(d['info']['guard_improvement'],d['old_info']['objective']-d['info']['objective']);A.close(d['sigma_ratio'],d['model'][0]/(.05*np.std(e.p.data['u'])))
        assert d['info']['guard_triggered']==(abs(d['old_model'][1][0])>.95 or d['old_info']['status']!='ok')
        if not d['info']['guard_triggered']:assert d['model']==d['old_model']
        else:assert len(d['info']['guard_rows'])==6;A.close(d['info']['objective'],min(r['objective'] for r in d['info']['guard_rows']))
        rows.append(dict(record=p.name,status='PASS',input_sha256={str(p):sha(p)}))
    assert len(rows)==144;dump(ROOT/'audit_guard.json',dict(status='PASS',rows=rows,source_sha256=sources('audit_auxiliary.py')))

def screen():
    d=read(ROOT/'design_screen.json');e,_,_=C.primary('kdv_cossin','iid',51)
    with np.load(PREVIOUS/'design_screen.npz') as z:a={k:z[k] for k in z.files}
    count=0
    for r in d['rows']:
        m=a[r['bank']+'_pilot'];cv=a[r['bank']+'_covariance'];values=[]
        for item in r['supports']:
            st=item['stochastic_features'];det=item['deterministic_features'];x=m[:,st]
            if det:
                # SVD projection, independent of production least squares.
                u,s,v=np.linalg.svd(m[:,det],full_matrices=False);rank=sum(s>1e-12*s[0]);u=u[:,:rank];x=x-u@(u.T@x)
            w,info=S.metric(cv,st);assert w is not None;g=S.gram(x@w);val=float(np.linalg.eigvalsh((g+g.T)/2)[0]);np.testing.assert_allclose(val,item['minimum'],rtol=3e-7,atol=1e-9);values.append(np.log1p(max(val,0)));count+=1
        A.close(float(np.quantile(values,.25)),r['score']);A.close(float(np.mean(values)),r['mean'])
    assert max((r for r in d['rows'] if r['bank']!='local'),key=lambda r:(r['score'],r['mean']))['bank']==d['selected'];dump(ROOT/'audit_screen.json',dict(status='PASS',supports=count,source_sha256=sources('audit_auxiliary.py'),input_sha256={str(ROOT/'design_screen.json'):sha(ROOT/'design_screen.json')}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['survey','guard','screen']);args=ap.parse_args();globals()[args.kind]();print(args.kind,'PASS',flush=True)

if __name__=='__main__':main()
