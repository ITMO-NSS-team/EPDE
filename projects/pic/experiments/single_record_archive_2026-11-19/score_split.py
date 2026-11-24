"""Validate frozen fits on withheld raw observations and locate each failed recovery."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import sys

from run_split import PARTS, ROOT, SEEDS, SPATIAL, SYSTEMS, folder, measurements
from validate import coefficient_matrices, make_banks, nondominated, scores
from transfer import LAWS, LOSS_NAN_VAL, labels_of, np, tokens, wls


def digest(field):
    return hashlib.sha256(field.tobytes()).hexdigest()


def coefficient_check(system, axes, data, other_axes, other_data, geometry):
    if system=='wave':
        names = ['1','d^2u/dx0^2{power: 1.0}','d^2u/dx1^2{power: 1.0}']
        truth = np.array([-1.,.04])
    else:
        names = ['1','du/dx0{power: 1.0}','du/dx1{power: 2.0}','u{power: 1.0} * d^2u/dx1^2{power: 1.0}']
        truth = np.array([-1.,1.,1.])
    def matrix(ax, obs):
        return make_banks(ax,obs,names,geometry)(obs)[0][:,1:]
    train = matrix(axes,data)
    test = matrix(other_axes,other_data)
    fit,status = wls(train,0)
    assert status=='ok'
    fit = -fit/fit[0]
    def wape(m,coef):
        return float(np.mean(abs(m@coef))/np.mean(abs(m[:,0])))
    return dict(terms=names[1:],weights=fit.tolist(),known=truth.tolist(),train=wape(train,fit),
                test=wape(test,fit),test_known=wape(test,truth))


def check(job):
    system,seed,part = job
    dest = folder(system,part)
    with gzip.open(dest/f'archive_{seed}.json.gz','rt') as f:
        archive = json.load(f)
    record = json.loads((dest/f'{system}_{seed}_wape.json').read_text())
    assert record['epochs']==10 and record['vwsr_calls']==0 and archive['calls']==record['calls']
    parts = measurements(system,seed)
    ax,raw = parts[part]
    f = raw['u']
    other = {'early':'late','late':'early','full':'full','left':'right','right':'left'}[part]
    ax2, val = parts[other]
    y = val['u']
    assert digest(f)==record['raw_sha256']['u']
    split_axis = int(part in SPATIAL)
    if part!='full':
        assert not np.shares_memory(f,y) and not np.intersect1d(ax[split_axis],ax2[split_axis]).size and digest(f)!=digest(y)
    names = archive['labels']
    labels = [labels_of('0 = '+n)[0] for n in names]
    assert len(set(labels))==len(labels)
    truth,_ = tokens(LAWS[system])
    all_supports = [tuple(sorted(labels[i] for i in row['ids'])) for row in archive['rows']]
    valid_ids = [i for i,r in enumerate(archive['rows']) if max(r['wape'],r['dyad'])<LOSS_NAN_VAL]
    rows = [archive['rows'][i] for i in valid_ids]
    exact = np.array([all_supports[i]==truth for i in valid_ids])
    pivots = np.array([r['pivot'] for r in rows])
    dyad = np.array([r['dyad'] for r in rows])
    geometry = record['geometry']
    train = make_banks(ax,f,names,geometry)(f)
    coefs = coefficient_matrices(rows,train)
    replay,internal = scores(train,coefs,pivots)
    native = np.array([r['wape'] for r in rows])
    np.testing.assert_allclose(replay,native,rtol=1e-6,atol=2e-9)
    sets = {'returned':np.array([r['returned'] for r in rows]),'terminal':np.array([r['terminal'] for r in rows]),
            'archive':np.ones(len(rows),bool)}
    def combination(i):
        return tuple(rows[i]['ids']),int(pivots[i])
    src_front = nondominated(np.flatnonzero(sets['terminal']),native,dyad)
    assert {combination(i) for i in src_front}=={combination(i) for i in np.flatnonzero(sets['returned'])}
    fields = [('source',native,internal)]
    if other!=part:
        test = make_banks(ax2,y,names,geometry)(y)
        assert all(a.shape==b.shape for a,b in zip(test,train))
        fields.append(('heldout', *scores(test,coefs,pivots)))
    outcomes = []
    observed = any(s==truth for s in all_supports)
    for evaluation, score, block in fields:
        for name,members in sets.items():
            for gate in (False,True):
                eligible = members & np.isfinite(score) & (score<=1)
                if gate:
                    eligible &= np.isfinite(block) & (block<=1)
                front = nondominated(np.flatnonzero(eligible),score,dyad)
                refs = np.flatnonzero(members & exact)
                hit = bool(np.any(exact[front]))
                if hit:
                    fate = 'retained'
                elif not observed:
                    fate = 'never scored'
                elif not np.any(exact):
                    fate = 'source ineligible'
                elif len(refs)==0:
                    fate = 'absent from pool'
                elif not np.any(eligible[refs]):
                    fate = 'validation ineligible'
                else:
                    fate = 'dominated'
                def point(i):
                    return dict(id=valid_ids[i], wape=float(score[i]), dyad=float(dyad[i]),
                                block=float(block[i]) if np.isfinite(block[i]) else None)
                outcomes.append(dict(evaluation=evaluation,pool=name,gated=gate,hit=hit,fate=fate,eligible=int(eligible.sum()),
                                     size=len({combination(i) for i in front}),front=[point(i) for i in front],
                                     reference=[point(i) for i in refs]))
    diag = coefficient_check(system,ax,f,ax2,y,geometry)
    result = dict(system=system,seed=seed,part=part,source_hash=digest(f),validation_hash=digest(y),rows=len(rows),
                  shape=f.shape,geometry=geometry,time_range=[float(ax[0][0]),float(ax[0][-1])],
                  validation_time_range=[float(ax2[0][0]),float(ax2[0][-1])],observed=observed,
                  calls=archive['calls'],max_replay_error=float(np.max(abs(replay-native))),coefficients=diag,outcomes=outcomes)
    out = dest/f'validation_{seed}.json'
    tmp = out.with_suffix('.pending')
    tmp.write_text(json.dumps(result,allow_nan=False),encoding='utf8')
    tmp.replace(out)
    print(system,seed,part,'checked',len(rows),flush=True)


if __name__=='__main__':
    jobs = [(s,n,p) for n in SEEDS for p in PARTS for s in SYSTEMS]
    if '--space' in sys.argv:
        jobs = [('wave',n,p) for n in SEEDS for p in SPATIAL]
    if len(sys.argv)==4:
        jobs = [(sys.argv[1],int(sys.argv[2]),sys.argv[3])]
    else:
        if '--ready' in sys.argv:
            jobs = [j for j in jobs if (folder(j[0],j[2])/f'{j[0]}_{j[1]}_wape.json').is_file()]
        jobs = [j for j in jobs if not (folder(j[0],j[2])/f'validation_{j[1]}.json').exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(check,jobs))
