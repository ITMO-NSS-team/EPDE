from common import *
import repair_fit as P
import state_score as State
import local6_fit as L
import importlib.util
spec=importlib.util.spec_from_file_location('robust_followup_checks',ROOT/'audit_followups.py');checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
checkfit,links=checks.checkfit,checks.links
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def temporal(y,m,info):
    cc=A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale];u=np.array(info['transformed']);t,jac=P.inverse(u,1)
    value,g,q=State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);g*=jac;lo=np.array([-P.BOUND,-40.,-P.BOUND,-P.BOUND]);hi=np.array([P.BOUND,20.,P.BOUND,P.BOUND]);pg=float(np.max(np.abs(u-np.clip(u-g,lo,hi))))
    np.testing.assert_allclose(value,info['objective'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(g,info['gradient'],rtol=1e-9,atol=1e-10);np.testing.assert_allclose(pg,info['projected_gradient'],atol=1e-12);np.testing.assert_allclose(m[0],scale*np.sqrt(q/cc.count),rtol=1e-12);np.testing.assert_allclose(m[1][0],t[0],atol=1e-12)
    if y.ndim>1:
        z=np.diff(y,n=4,axis=0);s=np.sign(z-np.median(z));rr=[]
        for axis in range(1,y.ndim):
            left=[slice(None)]*y.ndim;right=list(left);left[axis]=slice(None,-1);right[axis]=slice(1,None);rr.append(float(np.sin(np.pi*np.mean(s[tuple(left)]*s[tuple(right)])/2)))
        np.testing.assert_allclose(m[1][1:],rr,atol=1e-14)
    status='ok' if pg<=2e-5 else 'unresolved_optimization'
    if max(abs(r) for r in m[1])>1-1e-5 or t[1]>19.99:status='boundary'
    assert status==info['status']

def one(path):
    path=Path(path);d=read(path);count=links(d);replays=0;rawlinks=0
    if path.parent.name=='time_noise':
        for r in d['rows']:
            pid={0.:0,.4:1,-.4:2}[r['true_rho']];rng=np.random.default_rng(np.random.SeedSequence([960105,r['n'],pid,r['draw']]));y=R.WF.ar_noise(rng,(r['n'],),.2,[r['true_rho']]);assert R.digest(y)==r['field_sha256'];temporal(y,r['model'],r['info']);replays+=1;rawlinks+=1
    else:
        screen=path.parent.name=='time_screen';fpath=ROOT/('time_screen_frozen' if screen else 'time_validation_frozen')/path.name;f=read(fpath);assert sha(fpath)==d['frozen_sha256'];count+=links(f)+1
        r=d['rows'][0];name=r['system'] if screen else d['system'];profile=r['profile'] if screen else d['profile'];seed=36 if screen else d['seed'];p=R.datasets.load(name);pid,rho=S.PROFILES[profile];rng=np.random.default_rng(np.random.SeedSequence([960101 if screen else 982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==f['raw_sha256'];rawlinks+=len(raw)
        for r in d['rows']:
            v=r['variable'];noiseonly=(r['mode']=='noise_only') if screen else r['method']=='time_noise_only';y=raw[v]-p.data[v] if noiseonly else raw[v]
            if not screen and r['method']=='local6':checkfit(L,y,r['model'],r['info'])
            else:temporal(y,r['model'],r['info'])
            np.testing.assert_allclose(r['sigma_ratio'],r['model'][0]/(.05*np.std(p.data[v])),rtol=1e-14);replays+=1
        if not screen and f['bootstrap']:
            pilot,_=R.reconstruct(raw,R.axes_of(p),np.arange(len(R.axes_of(p)[0])));assert {v:R.digest(y) for v,y in pilot.items()}==f['pilot_sha256'];rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]))
            for b in f['bootstrap']:
                fields={v:y+R.WF.ar_noise(rng,y.shape,*f['time'][v][0][:2]) for v,y in pilot.items()};assert {v:R.digest(y) for v,y in fields.items()}==b['field_sha256'];rawlinks+=len(fields)
                for v,y in fields.items():
                    temporal(y,*b['time'][v]);checkfit(L,y,*b['local6'][v]);replays+=2
    out=dict(status='PASS',record=str(path.relative_to(ROOT)),hash_links=count,field_hashes=rawlinks,score_replays=replays,input_sha256={str(path):sha(path)});dump(ROOT/'time_audit_records'/(path.parent.name+'_'+path.name),out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();files=[]
    for folder,n in [('time_screen',23),('time_validation',69),('time_noise',9)]:
        ff=list((ROOT/folder).glob('*.json'));assert len(ff)==n,(folder,len(ff));files+=ff
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one,str(p)) for p in files]):r=f.result();rows.append(r);print(r['record'],'PASS',flush=True)
    out=dict(status='PASS',records=len(rows),totals={k:sum(r[k] for r in rows) for k in ('hash_links','field_hashes','score_replays')},rows=rows,source_sha256=hashes('audit_time.py','audit_followups.py'))
    dump(ROOT/'audit_time.json',out);print(out['totals'],flush=True)
if __name__=='__main__':main()
