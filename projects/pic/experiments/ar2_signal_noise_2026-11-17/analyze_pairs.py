from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
import score_fit as F
np=F.np
import hashlib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==138
    data=[json.loads(p.read_text()) for p in files];lookup={(d['system'],d['seed'],d['profile'],d['model_kind']):d for d in data};summary=[]
    for profile in ('iid','positive','negative'):
        control='iid' if profile=='iid' else 'known_rho';raw=[d for d in data if d['profile']==profile and d['model_kind']=='ar2'];pairs=[];missing=[]
        for d in raw:
            key=(d['system'],d['seed'],profile,control);c=lookup[key];assert d['raw_sha256']==c['raw_sha256']
            if d['status']=='COMPLETE' and c['status']=='COMPLETE':pairs.append((d,c))
            else:missing.append(dict(system=d['system'],seed=d['seed'],candidate_status=d['status'],control_status=c['status']))
        for bank in ('local','broad','both'):
            def rate(d):
                count=d['joint']['operator_coverage_count'] if bank=='both' else next(r['operator_coverage_count'] for r in d['statistics'] if r['bank']==bank)
                return count/256
            aa=np.array([rate(d) for d,c in pairs]);bb=np.array([rate(c) for d,c in pairs])
            summary.append(dict(profile=profile,bank=bank,planned_pairs=len(raw),complete_pairs=len(pairs),missing=missing,
                candidate_coverage=float(aa.mean()),control_coverage=float(bb.mean()),paired_difference=float((aa-bb).mean()),
                candidate_anchor_range=[float(aa.min()),float(aa.max())],control_anchor_range=[float(bb.min()),float(bb.max())]))
    out=dict(status='COMPLETE',summary=summary,source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256={str(p):sha(p) for p in files})
    (ROOT/'paired_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
