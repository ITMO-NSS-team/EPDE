"""Independent frozen-coefficient and measurement-noise checks for completed proposals."""
import gzip
import json

from replay import PRIOR, ROOT, SEEDS, live, np, ordered, read_archive, write_csv
from frequency_error import Profile

DRAWS = range(925001,925251)


def validate(seed, audit, event, native, problem):
    pivot = native['points'][event['parent']][1]
    pairs = native['points'][event['parent']][2]
    labels = [native['labels'][pivot],*[native['labels'][i] for i,w in pairs if i!=pivot]]
    entries = event['rows']
    fixed = ordered(entries,'fixed_gain')[:8]
    profiled = ordered(entries,'profiled_gain')[:8]
    peak = entries[fixed[0]]
    assert fixed[0]==profiled[0]
    harmonic = audit['labels'][peak['label']]
    assert len(harmonic)==1 and harmonic[0][0]=='cos' and peak['lower']<=1<=peak['upper']
    plan = {k:tuple(v) for k,v in native['plan'].items()}
    _,train,held = live.split(live.R.axes_of(problem))
    raw = problem.noisy(5,seed)
    assert native['train_hashes']=={k:live.R.digest(v[train[0]]) for k,v in raw.items()}
    def matrix(side,omega):
        moved = [(name,[(k,omega if k=='freq' else v) for k,v in p]) for name,p in harmonic]
        terms,opaque = live.R.symbols(problem,[*labels,moved])
        cuts,axes = side
        bank = live.R.LocalBank(axes,terms,{k:v[cuts] for k,v in opaque.items()},plan)
        return bank.matrix({k:v[cuts] for k,v in raw.items()})
    def coefficients(x,y):
        norms = np.linalg.norm(x,axis=0)
        weights,_,rank,_ = np.linalg.lstsq(x/norms,y,rcond=None)
        assert rank==len(norms)
        return weights/norms
    result = dict(seed=seed,parent=event['parent'],fixed_frequency=peak['initial'],
                  profiled_frequency=peak['frequency'],fixed_fits=event['work']['fixed_fits'],
                  profile_all_fits=event['work']['profiled_fits'],
                  profile_first_ranked_fits=event['work']['fixed_fits']+peak['fits']-1)
    weights = np.array([entries[i]['fixed_gain'] for i in fixed])
    result['expected_fits_profile_one_draw'] = result['fixed_fits']+float(np.average(
        [entries[i]['fits']-1 for i in fixed],weights=weights))
    for arm,w in [('fixed',peak['initial']),('profiled',peak['frequency'])]:
        m = matrix(train,w)
        y,x = m[:,0],m[:,1:]
        coef = coefficients(x,y)
        initial = y-x[:,:-1]@coefficients(x[:,:-1],y)
        residual = y-x@coef
        np.testing.assert_allclose(initial@initial-residual@residual,peak[arm+'_gain'],rtol=1e-7,atol=1e-10)
        m = matrix(held,w)
        result[arm+'_training_wape'] = np.abs(residual).sum()/np.abs(y).sum()
        result[arm+'_held_wape'] = np.abs(m[:,0]-m[:,1:]@coef).sum()/np.abs(m[:,0]).sum()
        result[arm+'_coefficients'] = json.dumps(coef.tolist())
    # The only moving column is the actually shortlisted standalone addition.
    profile = Profile(train[1][0],plan,[*labels,harmonic],peak,problem)
    observed = profile.estimate(raw['u'][train[0]])
    np.testing.assert_allclose(observed,peak['frequency'],atol=3e-6,rtol=0)
    clean = profile.estimate(problem.data['u'][train[0]])
    draws = [profile.estimate(problem.noisy(5,n)['u'][train[0]]) for n in DRAWS]
    low,med,high = np.quantile(draws,[.025,.5,.975])
    result.update(clean_frequency=clean,noise_draws=len(draws),q025=low,median=med,q975=high,
                  noise_in_band=sum(abs(w-1)<=.01 for w in draws))
    return result,[dict(parent_seed=seed,noise_seed=n,frequency=w) for n,w in zip(DRAWS,draws)]


if __name__=='__main__':
    results,draws = [],[]
    problem = live.load_data('duffing')
    for seed in SEEDS:
        with gzip.open(ROOT/'replay'/str(seed)/'rankings.json.gz','rt') as src:
            audit = json.load(src)
        events = [e for e in audit['events'] if e['group']=='pure']
        if events:
            native = read_archive(PRIOR/'results'/'duffing'/str(seed)/'native')
            result,replicates = validate(seed,audit,events[0],native,problem)
            results.append(result)
            draws.extend(replicates)
    write_csv(ROOT/'frozen_proposals.csv',results)
    write_csv(ROOT/'conditional_noise_draws.csv',draws)
    print('Verified',len(results),'frozen proposals and',len(draws),'conditional noise refits.',flush=True)
