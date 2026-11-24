"""Transport finite fitted coefficients to each other independent measurement in its triplet."""
from collections import Counter, defaultdict
import csv
import json
import numpy as np
from study import ROOT, COHORTS


def main():
    records = json.loads((ROOT/'certificates.json').read_text())['records']
    groups=defaultdict(list)
    for r in records:
        groups[r['cohort'],r['system'],r['profile']].append(r)
    rows=[]
    for (cohort,system,profile), group in groups.items():
        assert len(group)==3
        for dest in group:
            path=ROOT.parent/COHORTS[cohort]/'results'/dest['file']
            with np.load(path.with_suffix('.npz')) as archive:
                names=('local','broad') if cohort=='core' else ('local','broad','selected')
                mats={b:archive[b+'_observed'] for b in names}
                suffix='_anchor_covariance' if cohort=='core' else '_covariance'
                cov={b:archive[b+suffix] for b in names}
            lookup={(t['method'],t['candidate'],t['omitted']):t for t in dest['rows']}
            for source in group:
                if source['seed']==dest['seed']:
                    continue
                for t in source['rows']:
                    if t['method'] not in ('original','all_three') or t['status']!='compatible':
                        continue
                    dst=lookup[t['method'],t['candidate'],t['omitted']]
                    assert (t['target'],t['free'],t['term'])==(dst['target'],dst['free'],dst['term'])
                    indices=t['free']+[t['target']]
                    v=np.r_[-np.array(t['coefficients']),1.]
                    ratios=[]
                    for b in dst['banks']:
                        moment=mats[b][:,indices] @ v
                        allowed=dst['cutoff']*float(v @ cov[b][np.ix_(indices,indices)] @ v)
                        assert allowed>0
                        ratios.append(float(moment@moment/allowed))
                    worst=max(ratios)
                    assert abs(worst-1)>1e-10
                    rows.append(dict(cohort=cohort,system=system,profile=profile,
                                     source=source['seed'],destination=dest['seed'],candidate=t['candidate'],
                                     term=t['term'],parent=t['omitted'] is None,method=t['method'],
                                     max_ratio=worst,accepted=worst<1,destination_refit=dst['status']))
    with (ROOT/'heldout_transfer.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,rows[0]); writer.writeheader(); writer.writerows(rows)
    core=[r for r in rows if r['cohort']=='core' and not r['parent']]
    print('Core deletion transfers',sum(r['accepted'] for r in core),'/',len(core))


if __name__=='__main__':
    main()
