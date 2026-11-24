"""Compare first hits and nondominated fronts at an equal number of full scoring calls."""
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import sys
from statistics import median

from operator_trial import ARMS, PARTS, ROOT, SEEDS, SYSTEM, folder, measurements
from validate import make_banks, nondominated
from transfer import LAWS, LOSS_NAN_VAL, labels_of, np, tokens
from scipy.sparse import csr_matrix


def assess(job):
    arm,seed,part,cap,front_prefix=job
    dest=folder(arm,part)
    rec=json.loads((dest/f'{SYSTEM}_{seed}_wape.json').read_text())
    other='pvra' if arm=='native' else 'native'
    paired=json.loads((folder(other,part)/f'{SYSTEM}_{seed}_wape.json').read_text())
    for key in ('settings','raw_sha256','geometry'):
        assert rec[key]==paired[key]
    telemetry=json.loads((dest/f'operator_{seed}.json').read_text())
    with gzip.open(dest/f'archive_{seed}.json.gz','rt') as f:
        archive=json.load(f)
    assert rec['vwsr_calls']==0 and rec['epochs']==10 and rec['calls']==archive['calls']>=cap
    assert sum(r['calls'] for r in archive['rows'])==archive['calls']
    data=measurements(SYSTEM,seed)
    ax,obs=data[part]
    bx,held=data['early' if part=='late' else 'late']
    assert rec['raw_sha256']['u']==hashlib.sha256(obs['u'].tobytes()).hexdigest()
    assert not np.intersect1d(ax[0],bx[0]).size
    names=archive['labels']
    labels=[labels_of('0 = '+s)[0] for s in names]
    truth,_=tokens(LAWS[SYSTEM])
    points=archive['rows']
    exact=np.array([tuple(sorted(labels[j] for j in r['ids']))==truth for r in points])
    first=np.array([r['first_call'] for r in points])
    src=np.array([r['wape'] for r in points])
    dyad=np.array([r['dyad'] for r in points])
    valid=(src<LOSS_NAN_VAL) & (dyad<LOSS_NAN_VAL)
    ids=np.flatnonzero(valid)
    pivots=np.array([points[i]['pivot'] for i in ids])
    sparse_rows=[]
    columns=[]
    weights=[]
    for n,i in enumerate(ids):
        r=points[i]
        sparse_rows.extend([n]*len(r['ids']))
        columns.extend(r['ids'])
        weights.extend(r['weights'])
    matrix=csr_matrix((weights,(sparse_rows,columns)),shape=(len(ids),len(names)))
    def wape(axes,fields):
        m=make_banks(axes,fields['u'],names,rec['geometry'])(fields['u'])[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.mean(np.abs(matrix @ m.T),axis=1)/np.mean(np.abs(m),axis=0)[pivots]
    source=wape(ax,obs)
    np.testing.assert_allclose(source,src[ids],rtol=1e-6,atol=2e-9)
    validation=np.full(len(points),np.inf)
    validation[ids]=wape(bx,held)
    returned=np.array([r['returned'] for r in points])
    assert bool(np.any(exact & returned))==any(r['hit'] for r in rec['front'])
    details={}
    def front(key,members,score):
        selected=nondominated(np.flatnonzero(valid & members & np.isfinite(score) & (score<=1.)),score,dyad)
        details[key]=[dict(archive_id=int(i),wape=float(score[i]),dyad=float(dyad[i]),exact=bool(exact[i])) for i in selected]
        return int(exact[selected].any())
    metric=dict(arm=arm,seed=seed,part=part,budget=cap,first_exact=int(first[exact].min()) if exact.any() else None,
                generated=int(np.any(exact & (first<=cap))),eligible_exact=int(np.any(exact & valid & (first<=cap))),
                source=front('source',first<=cap,src),held=front('held',first<=cap,validation),
                ten_epoch_return=int(np.any(exact & returned)),returned_held=front('returned_held',returned,validation),
                total_calls=rec['calls'],distinct_support_targets=len({(r['pivot'],tuple(r['ids'])) for r in points if r['first_call']<=cap}),
                seconds=rec['seconds'],extra_gain_columns=telemetry['gain_columns'],
                successful_fills=telemetry['counters']['successful_fills'],
                guided_additions=telemetry['counters'].get('guided_additions',0),ordinary_adds=telemetry['counters']['ordinary_add_opportunities'],
                source_size=len(details['source']),held_size=len(details['held']),replay_error=float(np.max(np.abs(source-src[ids]))))
    (dest/f'{front_prefix}_{seed}.json').write_text(json.dumps(details,allow_nan=False))
    return metric


if __name__=='__main__':
    arms=(*ARMS,'pvpa') if sys.argv[1:]==['--pair'] else ARMS
    all_jobs=[(a,s,p) for p in PARTS for s in SEEDS for a in arms]
    budget=min([6000]+[json.loads((folder(a,p)/f'{SYSTEM}_{s}_wape.json').read_text())['calls'] for a,s,p in all_jobs])
    front_prefix='pair_fronts' if 'pvpa' in arms else 'fronts'
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(assess,[(*j,budget,front_prefix) for j in all_jobs]))
    output='pair_results.csv' if 'pvpa' in arms else 'paired_results.csv'
    with (ROOT/output).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for part in PARTS:
        for arm in arms:
            group=[r for r in rows if r['part']==part and r['arm']==arm]
            print(part,arm,[sum(r[k] for r in group) for k in ('generated','source','held','ten_epoch_return')],
                  'median seconds',round(median(r['seconds'] for r in group),1))
