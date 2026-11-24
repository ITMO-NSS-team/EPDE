"""Replay actual Duffing parents and compare matched fixed and profiled add shortlists."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from ranking import ROOT, Trial, live, np, libraries, ordered, rank
from assess_swap import read_archive, write_csv

PRIOR = ROOT.parent/'candidate_frequency_2026-11-23'
SEEDS = range(923701,923713)


def shape(label):
    return tuple(sorted((n,tuple((k,v) for k,v in p if k!='freq')) for n,p in label))


def cohort(a):
    problem = live.load_data('duffing')
    target,terms,_ = live.R.parse(problem.truth[0])
    rhs = {shape(t) for t in terms}
    tone, = [t for t in rhs if any(n=='cos' for n,_ in t)]
    remaining = rhs-{tone}
    labels = list(map(shape,a['labels']))
    first = {}
    for combo,cost in sorted(zip(a['systems'],a['operator']['system_cost']),key=lambda row:row[1]):
        pid, = combo
        slot,pivot,pairs,wape,_,_ = a['points'][pid]
        if wape>1 or labels[pivot]!=shape(target):
            continue
        ids = tuple(sorted(i for i,w in pairs if i!=pivot))
        support = {labels[i] for i in ids}
        if remaining <= support and tone not in support and len(ids)<=len(remaining)+1:
            first.setdefault((slot,pivot,ids),(pid,cost,'pure' if len(ids)==len(remaining) else 'distractor'))
    return {p:(cost,group) for p,cost,group in first.values()}


class Replay(Trial):
    def __init__(self,variables,required):
        super().__init__(variables,False)
        self.required = required
        self.parents = {}
        self.labels_for_rank = live.Observer(())
        self.events = []

    def hand_over(self,system):
        combo = super().hand_over(system)
        if combo and combo[0] in self.required and combo[0] not in self.parents:
            point = combo[0]
            assert self.calls==self.required[point][0]
            self.parents[point] = deepcopy(system.vals[self.variables[0]])

    def run(self,arm,seed,part,*,experiment):
        experiment()
        assert set(self.parents)==set(self.required)
        lib = None
        for pid,eq in self.parents.items():
            if lib is None:
                lib = libraries(eq)
            started = monotonic()
            rows,_,work = rank(eq,set(),self.ranges,lib,self.labels_for_rank)
            cost,group = self.required[pid]
            self.events.append(dict(parent=pid,first_handover=cost,group=group,rows=rows,
                                    work=work,seconds=monotonic()-started))
        return dict(system_cost=[self.system_cost[c] for c in sorted(self.systems)],full_cost=self.full_cost)


def worker(seed):
    src = PRIOR/'results'/'duffing'/str(seed)/'native'
    original = read_archive(src)
    digest = hashlib.sha256((src/'archive.json.gz').read_bytes()).hexdigest()
    manifest = json.loads((PRIOR/'confirmation'/'manifest.json').read_text())
    entry, = [r for r in manifest if r['dataset']=='duffing' and r['seed']==seed]
    assert digest==entry['archive_sha256']['native']
    trial = Replay(original['variables'],cohort(original))
    with TemporaryDirectory() as directory:
        dest = Path(directory)
        live.run('duffing',seed,'native',observer_factory=lambda _:trial,operator_run=trial.run,
                 destination=dest,trig_factory=trial.family)
        actual = read_archive(dest)
        for key in ('settings','train_hashes','plan','variables','trig','axis','calls','labels','points','systems','system_events'):
            assert actual[key]==original[key],(seed,key)
        for key in ('full_cost','system_cost'):
            assert actual['operator'][key]==original['operator'][key]
    data = dict(seed=seed,native_sha256=digest,labels=trial.labels_for_rank.labels,events=trial.events)
    path = ROOT/'replay'/str(seed)/'rankings.json.gz'
    temp = path.with_suffix('.pending')
    with gzip.open(temp,'wt',encoding='utf8') as out:
        json.dump(data,out,allow_nan=False)
    temp.replace(path)


def launch(seed):
    dest = ROOT/'replay'/str(seed)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'rankings.json.gz').is_file():
        return
    with (dest/'search.log').open('w') as log:
        run = subprocess.run([sys.executable,'-u',__file__,'--worker',str(seed)],env=dict(os.environ,PYTHONHASHSEED='0'),
                             stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3000:])
    print(seed,flush=True)


def publish():
    rows, coverage, arms = [], [], ('ordinary','fixed','profiled')
    for seed in SEEDS:
        with gzip.open(ROOT/'replay'/str(seed)/'rankings.json.gz','rt') as src:
            audit = json.load(src)
        counts = Counter(e['group'] for e in audit['events'])
        coverage.append(dict(seed=seed,pure=counts['pure'],distractor=counts['distractor']))
        for e in audit['events']:
            for arm in arms:
                table = [r for r in e['rows'] if arm!='ordinary' or r['ordinary']]
                field = 'profiled_gain' if arm=='profiled' else 'fixed_gain'
                order = ordered(table,field)
                shortlist = order[:8]
                def bare(r):
                    label = audit['labels'][r['label']]
                    return len(label)==1 and label[0][0]=='cos' and r['axis']==0
                matches = [i for i,r in enumerate(table) if bare(r) and r['lower']<=1<=r['upper']]
                assert len(matches)<=1
                index = matches[0] if matches else None
                r = table[index] if matches else {}
                position = order.index(index)+1 if index in order else None
                inside = index in shortlist
                w = r.get('frequency' if arm=='profiled' else 'initial')
                mass = r[field]/sum(table[j][field] for j in shortlist) if inside else 0.
                rows.append(dict(seed=seed,parent=e['parent'],group=e['group'],first_handover=e['first_handover'],
                                 arm=arm,terms=len(table),trig_terms=sum('frequency' in v for v in table),
                                 matching_available=int(bool(matches)),rank=position,top8=int(inside),probability=mass,
                                 frequency=w,strict_top8=int(inside and abs(w-1)<=.01),
                                 fits=1+sum(v['fits'] if arm=='profiled' else 1 for v in table)))
    write_csv(ROOT/'parent_comparisons.csv',rows)
    write_csv(ROOT/'opportunities.csv',coverage)
    pure = [r for r in rows if r['group']=='pure']
    assert len(pure)==3*sum(x['pure'] for x in coverage) and all(x['pure']<=1 for x in coverage)
    summaries = {}
    for arm in arms:
        sel = [r for r in pure if r['arm']==arm]
        summaries[arm] = dict(seeds=len(sel),in_library=sum(r['matching_available'] for r in sel),
                              top8=sum(r['top8'] for r in sel),strict=sum(r['strict_top8'] for r in sel),
                              mean_probability=float(np.mean([r['probability'] for r in sel])),
                              fits=sum(r['fits'] for r in sel))
    left,right = summaries['fixed'],summaries['profiled']
    advance = right['top8']>left['top8'] or (right['top8']==left['top8'] and
                                             right['mean_probability']>left['mean_probability']+1e-10)
    result = dict(pure=summaries,advance=advance,verified_native_runs=len(coverage),distinct_parents=len(rows)//3)
    (ROOT/'replay_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=int)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=p.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,args.seeds))
        if all((ROOT/'replay'/str(n)/'rankings.json.gz').is_file() for n in SEEDS):
            publish()
