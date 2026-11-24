"""Replay all six prospective wave triples and summarize their whole-front outcomes."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from search import ARMS,ROOT
from transfer import run_pair
from wave_confirmation import SEEDS


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        transfer=list(pool.map(run_pair,[('wave',seed) for seed in SEEDS]))
    (ROOT/'wave_transfer.json').write_text(json.dumps(transfer,allow_nan=False))
    rows=[]
    for tr in transfer:
        seed=tr['seed']
        for arm in ARMS:
            r=json.loads((ROOT/'results'/f'wave_{seed}_{arm}.json').read_text())
            hits=[f for f in r['front'] if f['hit']]
            v=tr['reference_coefficients']
            # Labels are canonicalized. Find the unique spatial derivative.
            j,=[j for j,label in enumerate(v['labels']) if 'dx1' in label[0][0]]
            rows.append(dict(seed=seed,arm=arm,hit=int(bool(hits)),seen=int(bool(r['reference_points'])),
                             pooled_source=int(tr['survival'][arm][0]),pooled_fresh=sum(tr['survival'][arm][1:]),
                             native_standard_target=v['wls'][j],tls_standard_target=v['tls'][j],
                             max_score_replay_error=tr['max_score_replay_error']))
    with (ROOT/'wave_confirmation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print({a:[sum(r[k] for r in rows if r['arm']==a) for k in ('hit','seen','pooled_source','pooled_fresh')] for a in ARMS})
