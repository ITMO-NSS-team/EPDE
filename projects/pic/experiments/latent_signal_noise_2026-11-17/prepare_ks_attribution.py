from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
    t=(ROOT/'failure_attribution.py').read_text(encoding='utf-8')
    t=t.replace('name,seed=job','name,seed,profile=job').replace("{name}_iid_curvature_seed{seed}","{name}_{profile}_curvature_seed{seed}")
    t=t.replace("e=C.Study(name,seed,'iid','curvature');", "e=C.Study(name,seed,profile,'known_rho');e.model=d['model'];e.cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),e.model);")
    t=t.replace('[0.]*y.ndim','[e.rho]*y.ndim')
    t=t.replace("{name}_attribution_seed{seed}","{name}_{profile}_attribution_seed{seed}")
    t=t.replace('system=name,seed=seed','system=name,seed=seed,profile=profile')
    t=t.replace("'failure_attribution_protocol.txt'","'ks_attribution_protocol.txt'")
    t=t.replace("pool.submit(one,(n,s)) for n in ('ns','pde_compound') for s in (36,37,38)","pool.submit(one,('ks',36,p)) for p in ('iid','positive','negative')")
    t=t.replace("'failure_attribution_summary.json'","'ks_attribution_summary.json'")
    p=ROOT/'ks_attribution.py';assert not p.exists();p.write_text(t,encoding='utf-8')
    t=(ROOT/'audit_attribution.py').read_text(encoding='utf-8')
    t=t.replace("for name in ('ns','pde_compound'):\n        for seed in (36,37,38):","for name in ('ks',):\n        for profile in ('iid','positive','negative'):\n            seed=36")
    t=t.replace("{name}_attribution_seed{seed}","{name}_{profile}_attribution_seed{seed}").replace("{name}_iid_curvature_seed{seed}","{name}_{profile}_curvature_seed{seed}")
    t=t.replace("C.Study(name,seed,'iid','iid')","C.Study(name,seed,profile,'known_rho')").replace('[0.]*y.ndim','[e.rho]*y.ndim')
    t=t.replace("'audit_attribution.json'","'audit_ks_attribution.json'").replace("'failure_attribution_summary.json'","'ks_attribution_summary.json'").replace('records=6','records=3')
    p=ROOT/'audit_ks_attribution.py';assert not p.exists();p.write_text(t,encoding='utf-8')
    paths=[ROOT/n for n in ('failure_attribution.py','audit_attribution.py','ks_attribution.py','audit_ks_attribution.py')]
    (ROOT/'ks_harness_provenance.json').write_text(json.dumps({'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

if __name__=='__main__':main()
