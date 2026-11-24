from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'latent_signal_noise_2026-11-17'

def main():
    files=[]
    for source,dest in [('failure_attribution.py','compound_attribution.py'),('audit_attribution.py','audit_attribution.py')]:
        t=(OLD/source).read_text(encoding='utf-8').replace("('ns','pde_compound')","('pde_compound',)").replace('(36,37,38)','(39,40,41)').replace('_iid_curvature_seed','_iid_ar2_seed')
        t=t.replace("'failure_attribution_protocol.txt'","'compound_attribution_protocol.txt'").replace("'failure_attribution_summary.json'","'compound_attribution_summary.json'")
        if source=='failure_attribution.py':
            t=t.replace("e=C.Study(name,seed,'iid','curvature');","e=C.Study(name,seed,'iid','iid');e.model=d['model'];e.cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),e.model);")
            t=t.replace("ROOT/'correlated_fit.py',ROOT/'latent_fit.py'","ROOT/'score_fit.py',ROOT/'score_core.py',ROOT/'fit.py'")
        else:t=t.replace('records=6','records=3')
        p=ROOT/dest;assert not p.exists();p.write_text(t,encoding='utf-8');files.append((OLD/source,p))
    (ROOT/'attribution_provenance.json').write_text(json.dumps(dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for pair in files for p in pair}),indent=2),encoding='utf-8')

if __name__=='__main__':main()
