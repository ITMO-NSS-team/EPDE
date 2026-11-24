from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'latent_signal_noise_2026-11-17'

def save(name,t):
    p=ROOT/name;assert not p.exists();p.write_text(t,encoding='utf-8')

def main():
    generated=[]
    for name in ('coverage.py','run_case.py','run_suite.py','audit.py','audit_live.py','analyze.py','analyze_pairs.py'):
        t=(OLD/name).read_text(encoding='utf-8').replace('import correlated_fit as F','import score_fit as F')
        t=t.replace("'curvature'","'ar2'").replace('(36,37,38)','(39,40,41)')
        for a,b in [('960101','970101'),('960102','970102'),('960103','970103'),('960104','970104')]:t=t.replace(a,b)
        if name=='coverage.py':
            start=t.index('def sources():');end=t.index('class Study:',start)
            t=t[:start]+"def sources():\n    paths=[ROOT/p for p in ('coverage.py','run_case.py','fit.py','score_fit.py','score_core.py','score_protocol.txt','protocol.txt','coverage_decision.txt')]+[F.PREVIOUS/p for p in ('latent_fit.py','correlated_fit.py')]\n    return {**S.source_hashes(),str(S.ROOT/'intersect_banks.py'):S.E.sha(S.ROOT/'intersect_banks.py'),**{str(p):S.E.sha(p) for p in paths}}\n\n"+t[end:]
        if name=='run_suite.py':t=t.replace("('checks.json','checks_correlated.json')","('checks.json','checks_score.json')")
        if name=='audit_live.py':
            t=t.replace("type=int,default=2);args=parser.parse_args()","type=int,default=7);parser.add_argument('--initial-workers',type=int,default=2);args=parser.parse_args()")
            t=t.replace("            files=sorted", "            limit=args.workers if (ROOT/'processes.json').exists() else args.initial_workers\n            files=sorted")
            t=t.replace('len(pending)<2*args.workers','len(pending)<limit')
        if name=='analyze.py':
            t=t.replace('Latent curvature','AR2 curvature').replace('Latent signal/noise: fresh anchors 36-38','AR2 signal/noise: fresh anchors 39-41').replace('Latent/local','AR2/local').replace('Latent/broad','AR2/broad')
        save(name,t);generated.append(ROOT/name)
    save('harness_provenance.json',json.dumps(dict(prior_source_sha256={str(OLD/p.name):hashlib.sha256((OLD/p.name).read_bytes()).hexdigest() for p in generated},generated_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}),indent=2))

if __name__=='__main__':main()
