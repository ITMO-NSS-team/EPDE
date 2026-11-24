from pathlib import Path
import json,importlib.util
ROOT=Path(__file__).resolve().parent
import axis_fit as F
from score_core import profile_score
np=F.np
spec=importlib.util.spec_from_file_location('axis_audit_previous',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R

def main():
    d=json.loads((ROOT/'axis_diagnostic.json').read_text());assert d['status']=='COMPLETE' and len(d['rows'])==13;cached={};links=0;inputs={}
    for row in d['rows']:
        name,v,axis=row['system'],row['variable'],row['axis'];p=ROOT/'axis_frozen'/f'{name}_{v}_axis{axis}.json';f=json.loads(p.read_text());assert S.E.sha(p)==row['frozen_sha256'];inputs[str(p)]=S.E.sha(p)
        for key in ('source_sha256','input_sha256'):
            for pp,hh in f[key].items():assert S.E.sha(pp)==hh;links+=1
        if name not in cached:cached[name]=C.Study(name,36,'iid','iid')
        e=cached[name];assert f['raw_sha256']=={vv:R.digest(y) for vv,y in e.raw.items()};info=f['info'];model=f['model'];assert model==row['model'] and info['status']==row['status']
        best=min((x for x in info['fits'] if x['success'] and x['value']<1e40),key=lambda x:x['value']);t=np.asarray(best['theta']);cc=F.Contrasts(e.raw[v],axis);scale=np.sqrt(sum(float(np.sum(x*x)) for x in cc.fields)/cc.count);cc.fields=[x/scale for x in cc.fields]
        q,ld=cc.evaluate(t[:1],np.exp(t[1:2]),t[2:3],t[3:]);objective,gradient=profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],F.covariance_band)
        np.testing.assert_allclose([model[0],info['objective']],[scale*np.sqrt(q/cc.count),objective],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(best['gradient'],gradient,rtol=2e-8,atol=2e-10)
        true=.05*float(np.std(e.p.data[v]));assert row['sigma_ratio']==model[0]/true;diff=np.moveaxis(np.diff(e.p.data[v],n=2,axis=axis),axis,0).reshape(e.raw[v].shape[axis]-2,-1);energy=float(np.sum(diff*diff));flat=np.sort((diff*diff).ravel());n_top=max(1,int(np.ceil(len(flat)*.01)))
        expected=[float(np.sum(np.mean(diff,axis=0)**2)*len(diff)/energy),float(flat[-n_top:].sum()/energy),energy/(diff.size*6*true*true)];actual=[row[k] for k in ('clean_curvature_mean_energy_fraction','clean_curvature_top1percent_energy','clean_curvature_to_expected_noise_energy')];np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        gp=ROOT/'score_screen'/f'{name}_iid_seed36.json';inputs[str(gp)]=S.E.sha(gp);g=next(r for r in json.loads(gp.read_text())['rows'] if r['mode']=='raw' and r['variable']==v);assert row['global_model']==g['model'] and row['global_sigma_ratio']==g['sigma_ratio']
    inputs[str(ROOT/'axis_diagnostic.json')]=S.E.sha(ROOT/'axis_diagnostic.json')
    R.dump(ROOT/'audit_axis.json',dict(status='PASS',fields=13,likelihood_replays=13,gradient_replays=13,hash_links=links,boundary=sum(r['status']=='boundary' for r in d['rows']),source_sha256={str(Path(__file__)):S.E.sha(__file__)},input_sha256=inputs));print('PASS',13,links)

if __name__=='__main__':main()
