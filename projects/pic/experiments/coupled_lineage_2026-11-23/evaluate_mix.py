"""Budget-matched paired assessment of the archived donor-block operator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from statistics import median

from mix_trial import CONFIRMATION, ROOT, folder, live
from assess_swap import assess_run, read_archive, write_csv


def assess(job):
    return assess_run(job, destination=folder(*job), moves=None, timing='mix_seconds')


def publish(seeds, out):
    systems, laws, costs, by_system, verification = [], [], [], [], []
    summary = {}
    screen = list(seeds)==list(live.SEEDS)
    if screen:
        source = json.loads((ROOT.parent/'guided_swap_2026-11-23'/'manifest.json').read_text())['pairs']
    else:
        source = json.loads((ROOT.parent/'candidate_frequency_2026-11-23'/'confirmation'/'manifest.json').read_text())
    hashes = {(r['dataset'],r['seed']):r['archive_sha256']['native'] for r in source}
    for name in live.DATASETS:
        for seed in seeds:
            native = folder(name,seed,'native')
            original = read_archive(native)
            scalar = len(original['variables'])==1
            dest = native if scalar else folder(name,seed,'mix')
            native_sha = hashlib.sha256((native/'archive.json.gz').read_bytes()).hexdigest()
            assert native_sha==hashes[name,seed]
            base, mix = [json.loads((d/'assessment.json').read_text()) for d in (native,dest)]
            for k in ('train_hashes','windows','settings','peaks','trig'):
                assert base[k]==mix[k]
            if scalar:
                events, decisions = [], {}
                digest = native_sha
            else:
                archive = read_archive(dest)
                op = archive['operator']
                assert archive['calls']==live.BUDGET and op['full_cost'][-1]<=live.BUDGET+op['extra_fits']
                events,decisions = op['events'],op['decisions']
                assert sum(decisions.values())==len(events)
                digest = hashlib.sha256((dest/'archive.json.gz').read_bytes()).hexdigest()
            verification.append(dict(dataset=name,seed=seed,native_sha256=native_sha,mix_sha256=digest,
                                     native_scalar_path=scalar))
            costs.append(dict(dataset=name,seed=seed,native_scalar_path=scalar,attempts=len(events),
                              accepted=sum(v for k,v in decisions.items() if k.startswith('accept_')),
                              extra_fits=mix['extra_fits'],replay_error=mix['replay_error'],
                              crossover_seconds=mix.get('mix_seconds',0),decisions=json.dumps(decisions,sort_keys=True),
                              operator_work=json.dumps(mix['work'] if not scalar else {},sort_keys=True)))
            for b in ('full','total'):
                for arm,r in [('native',base),('mix',mix)]:
                    ident = dict(dataset=name,seed=seed,budget=b,arm=arm)
                    s = r[b]['system']
                    systems.append(dict(**ident,**{k:s[k] for k in
                        ('calls','candidates','total_fits','assembled','source','held','source_system_front_size',
                         'held_system_front_size','source_law_front_size','held_law_front_size')}))
                    for law in r[b]['laws']:
                        laws.append(dict(**ident,**{k:law[k] for k in
                            ('law','generated','source','held','held_fate','first','first_total',
                             'first_coefficient_error','held_coefficient_error')}))
    for b in ('full','total'):
        gains, losses = [], []
        for name in live.DATASETS:
            data = [r for r in systems if r['dataset']==name and r['budget']==b]
            pairs = list(zip(data[::2],data[1::2],strict=True))
            change = {k:[r['seed'] for a,r in pairs if r['held']-a['held']==v] for k,v in [('gains',1),('losses',-1)]}
            if name!='ode_simple':
                gains.extend((name,n) for n in change['gains'])
                losses.extend((name,n) for n in change['losses'])
            by_system.append(dict(dataset=name,budget=b,**{arm:sum(r['held'] for r in data if r['arm']==arm)
                                  for arm in ('native','mix')},gains=len(change['gains']),losses=len(change['losses']),
                                  **{arm+'_median_front':median(r['held_system_front_size'] for r in data if r['arm']==arm)
                                     for arm in ('native','mix')}))
        summary[b] = dict(gains=gains,losses=losses,
                          **{arm:sum(r['held'] for r in systems if r['dataset']!='ode_simple' and r['budget']==b and r['arm']==arm)
                             for arm in ('native','mix')})
    summary['positive_balance'] = len(summary['total']['gains'])>len(summary['total']['losses'])
    summary['extra_fits'] = sum(r['extra_fits'] for r in costs)
    summary['control_held'] = {arm:sum(r['held'] for r in systems if r['dataset']=='ode_simple' and r['budget']=='total'
                              and r['arm']==arm) for arm in ('native','mix')}
    out.mkdir(exist_ok=True)
    for name,rows in [('systems',systems),('laws',laws),('work',costs),('by_system',by_system)]:
        write_csv(out/(name+'.csv'),rows)
    (out/'manifest.json').write_text(json.dumps(verification,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['total']),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--confirmation',action='store_true')
    args = p.parse_args()
    seeds = CONFIRMATION if args.confirmation else live.SEEDS
    coupled = [s for s in live.DATASETS if len(live.load_data(s).variables)>1]
    jobs = [(s,n,'mix') for n in seeds for s in coupled]
    todo = [j for j in jobs if (folder(*j)/'archive.json.gz').exists() and not (folder(*j)/'assessment.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in pool.map(assess,todo):
            print(*job,flush=True)
    if all((folder(*j)/'assessment.json').exists() for j in jobs):
        publish(seeds, ROOT/('mix_confirmation' if args.confirmation else 'mix_screen'))
