"""Replay scalar structural/frequency misses without changing the primary pilot results."""
from collections import Counter
import ast
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from functools import partial
import csv
import gzip
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pilot
from report import describe_term


def term_name(term):
    label = [(f.label if f.ftype != 'grids' else 'x',
              [(v['name'], float(f.params[i])) for i,v in f.params_description.items()])
             for f in term.structure]
    return describe_term(label)[0]


def support(eq, fitted=False):
    terms = [term_name(t) for i,t in enumerate(eq.structure)
             if not fitted or i == eq.target_idx or eq.weights_final[eq.weight_index(i,eq.target_idx)]]
    if fitted and eq.weights_final[-1]:
        terms.append('1')
    return tuple(sorted(terms))


def read_archive(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return json.load(f)


def replay(name, arm, draw, detailed_fit=False):
    pilot.limit_threads()
    from cpu_budget import await_capacity
    await_capacity()
    import numpy as np
    from noise_suite import live, trial
    from compare import system_front
    from epde.operators.common import fill_prune, plain_fit
    from epde.optimizers.nsga.optimizer import NSGAOptimizer
    from epde.operators.common.objectives import LOSS_NAN_VAL
    source = pilot.ROOT/'results'/name/arm/str(draw)/'archive.json.gz'
    original = read_archive(source)
    target, terms, _ = live.R.parse(live.load_data(name).truth_systems[0][0])
    expected = tuple(sorted(describe_term(t)[0] for t in (target, *terms)))
    omega, = [w for t in (target, *terms) for w in describe_term(t)[1]]
    counts = Counter()
    slots, parents, ineligible, probes, raw_details = [], {}, set(), [], []
    beam_checks = {}
    current = dict(epoch=0, args=None, ranges={})
    native_versions, native_profile = trial.plain_versions, trial.Trial.profile
    native_handover, native_report = plain_fit._hand_over, NSGAOptimizer._report
    native_beam, native_draw = fill_prune._beam_levels, fill_prune.draw_fill
    printed = print

    def log(*args, **kwargs):
        if args and isinstance(args[0], str):
            match = re.search(r'NSGA-II : (\d+)-th epoch', args[0])
            if match:
                current['epoch'] = int(match[1])
        printed(*args, **kwargs)

    def versions(eq, *args):
        if current.get('probing'):
            return native_versions(eq, *args)
        current['args'] = args
        raw = support(eq)
        if detailed_fit and raw == expected:
            from epde.operators.common import multisample
            detail = dict(epoch=current['epoch'],target=term_name(eq.target),
                          tones=[float(f.params[i]) for t in eq.structure for f,i,_ in trial.trig_terms(t)])
            normal_pareto, normal_subsets = plain_fit._pareto_versions, multisample.joint_subset_table
            def describe(variants):
                return [dict(terms=support(v,True),vector=[*v._record_wape,v.coefficients_stability])
                        for v in variants]
            def pareto(variants,host):
                detail['before_pareto'] = describe(variants)
                return normal_pareto(variants,host)
            def subsets(*args,**kwargs):
                result = normal_subsets(*args,**kwargs)
                detail['subsets'] = dict(masks=kwargs['subsets'],discrepancy=result[0].tolist(),
                                         amplification=result[1].tolist(),full_rank=result[4].tolist())
                return result
            with patch.object(plain_fit,'_pareto_versions',pareto), patch.object(multisample,'joint_subset_table',subsets):
                out = native_versions(eq, *args)
            detail['returned'] = describe(out)
            raw_details.append(detail)
        else:
            out = native_versions(eq, *args)
        if raw == expected:
            counts['exact_raw_fit_calls'] += 1
            counts['exact_raw_calls_returning_catalog_terms'] += int(any(support(v, True) == expected for v in out))
        return out

    def profile(instance, item, args):
        slots.append(dict(epoch=current['epoch'], terms=support(item[-1].vals[instance.points[item[1]][0]], True)))
        current['ranges'] = instance.ranges
        return native_profile(instance, item, args)

    def handover(system, siblings):
        for candidate in (system, *siblings):
            eq, = candidate.vals
            if not plain_fit._fitted(eq) or support(eq, True) != expected:
                continue
            tone_set = {(axis, float(f.params[i])) for t in eq.structure for f,i,axis in trial.trig_terms(t)}
            if len(tone_set) != 1 or any(trial.trig_terms(eq.target)):
                ineligible.add('frequency count or target')
                continue
            tone, = tone_set
            key = (term_name(eq.target), tone)
            if tone not in current['ranges']:
                ineligible.add(('unbracketed', tone[1]))
            elif key not in parents:
                parents[key] = (deepcopy(candidate), current['args'], current['epoch'], len(slots))
        return native_handover(system, siblings)

    def beam(gs, var, label, cand, sig, t, k, width, top, mode):
        answer = native_beam(gs, var, label, cand, sig, t, k, width, top, mode)
        if name == 'duffing' and answer[0] == 'ok':
            current['library'], current['target'] = gs['lib'], term_name(gs['lib'][t])
            level = answer[1].get(len(expected)-1, [])
            cache_key = (id(gs), sig, t)
            if level and current['target'] in expected and cache_key not in beam_checks:
                names = [term_name(term) for term in gs['lib']]
                predictors = [wanted for wanted in expected if wanted != current['target']]
                alternatives = [[int(i) for i in cand if names[i] == wanted] for wanted in predictors]
                unavailable = {}
                for wanted, allowed in zip(predictors,alternatives):
                    if not allowed:
                        present = [i for i,term in enumerate(names) if term == wanted]
                        unavailable[wanted] = ('absent from library' if not present else
                                               'invalid column' if not any(gs['ok'][present]) else 'excluded for target')
                rss = []
                for indices in product(*alternatives):
                    b = gs['Gn'][list(indices),t]
                    gram = gs['Gn'][np.ix_(indices,indices)]
                    try:
                        rss.append(float(1 - b @ np.linalg.solve(gram,b)))
                    except np.linalg.LinAlgError:
                        pass
                beam_checks[cache_key] = (gs, dict(target=current['target'],eligible=all(alternatives),unavailable=unavailable,
                    best_catalog_rss=min(rss) if rss else None, cutoff=max(r for _,r in level),
                    options=len(level),offered=any(tuple(sorted([current['target'], *(names[j] for j in ids)]))
                                                   == expected for ids,_ in level)))
        return answer

    def draw_fill(top):
        choice = native_draw(top)
        if name == 'duffing' and len(top[0][0]) == len(expected)-1:
            counts['fill_draws_at_catalog_size'] += 1
            lib, target = current['library'], current['target']
            exact = [i for i,(ids,_) in enumerate(top)
                     if tuple(sorted([target, *(term_name(lib[j]) for j in ids)])) == expected]
            counts['fill_draws_offering_catalog'] += bool(exact)
            counts['fill_draws_picking_catalog'] += choice in exact
        return choice

    def after_search(optimizer):
        native_report(optimizer)
        current['probing'] = True
        # These extra fits never enter the live search. Test every distinct
        # eligible seed/target actually seen for the catalog term set.
        for (target_name,tone),(parent,args,epoch,spent) in parents.items():
            children = []
            class Probe(trial.Trial):
                def hand_over(self, child):
                    values = child.obj_fun
                    if np.isfinite(values).all() and (values < LOSS_NAN_VAL).all():
                        eq, = child.vals
                        frequencies = [float(f.params[i]) for t in eq.structure for f,i,_ in trial.trig_terms(t)]
                        hit = (support(eq, True) == expected and len(frequencies) == 1
                               and abs(frequencies[0]-omega) <= 0.01*omega)
                        children.append(dict(values=values.tolist(), catalog=hit))
            eq, = parent.vals
            probe = Probe([eq.main_var_to_explain], True)
            probe.points = [(eq.main_var_to_explain,)]
            probe.ranges = current['ranges']
            native_profile(probe, (None,0,(0,),tone,parent), args)
            p, = probe.profiles
            probes.append(dict(target=target_name,seed=tone[1],first_epoch=epoch,slots_spent=spent,
                               lower=p['lower'],upper=p['upper'],proposed=p['proposed'],children=children))

    with tempfile.TemporaryDirectory(dir=pilot.ROOT) as temp:
        archive = Path(temp)/'archive.json.gz'
        with ExitStack() as stack:
            for obj,attr,fn in [(trial,'plain_versions',versions),(trial.Trial,'profile',profile),
                                (plain_fit,'_hand_over',handover),(NSGAOptimizer,'_report',after_search),
                                (fill_prune,'_beam_levels',beam),(fill_prune,'draw_fill',draw_fill)]:
                stack.enter_context(patch.object(obj,attr,fn))
            stack.enter_context(patch('builtins.print',log))
            stack.enter_context(redirect_stdout(stack.enter_context(open(os.devnull,'w'))))
            pilot.search(name,arm,draw,original['epochs'],archive,policy='first12')
        result = read_archive(archive)
    for field in ('samples','options','points','labels','systems','front','frequency_profiles'):
        if result[field] != original[field]:
            raise ValueError(f'{name}/{arm}/{draw}: replay differs in {field}')
    new = [child for p in probes for child in p['children']]
    if new:
        points = original['points']
        vectors = [[*points[i][5],points[i][4]] for i, in original['systems']]
        mask = system_front(np.asarray(vectors+[c['values'] for c in new],dtype=float))[-len(new):]
        for child,on_front in zip(new,mask):
            child['on_augmented_front'] = bool(on_front)
    return dict(dataset=name,arm=arm,draw=draw,exact_replay=True,counts=dict(counts),
                beam_checks=[entry for _,entry in beam_checks.values()],profiles=slots,
                eligible_probes=probes,ineligible=sorted(map(str,ineligible)),raw_details=raw_details)


def run(row):
    command = [sys.executable,__file__,'--worker',row['dataset'],row['arm'],row['draw']]
    child = subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONHASHSEED='0'))
    if child.returncode:
        raise RuntimeError(child.stderr or child.stdout)
    result = json.loads(child.stdout.splitlines()[-1])
    print('Verified replay',row['dataset'],row['arm'],row['draw'],flush=True)
    return result


def check_seeds():
    """Run the original spectral estimator on matched clean data, verifying draw zero against the logs."""
    pilot.limit_threads()
    import noise_suite
    import numpy as np
    from epde.interface.token_frequencies import dominant_frequencies
    rows = []
    for name, omega in [('ode',2.), ('duffing',1.)]:
        for arm in pilot.ARMS:
            recorded = []
            for draw in pilot.DRAWS:
                log = (pilot.ROOT/'results'/name/arm/str(draw)/'worker.log').read_text(encoding='utf8')
                recorded.append(ast.literal_eval(re.search(r'Frequencies of trigonometric found in the data: (\[.*?\])',log)[1]))
            for clean in (False, True):
                with ExitStack() as stack:
                    if clean:
                        stack.enter_context(patch.object(noise_suite,'measurement',lambda family: lambda p,*args:(p.data,{})))
                    problem, samples, _ = pilot.records(name,arm,0)
                axes = noise_suite.live.R.axes_of(problem)
                trajectories = [SimpleNamespace(variable_names=list(y),raw_field=y.__getitem__,grids=lambda _:axes)
                                for _,y in samples]
                peaks = dominant_frequencies(trajectories,axes=[0],n=2,unit='angular')
                if not clean:
                    np.testing.assert_allclose(peaks,recorded[0],rtol=1e-12)
            rows.append(dict(dataset=name,arm=arm,catalog=omega,clean_peaks=list(peaks),noisy_peaks=recorded))
    out = pilot.ROOT/'frequency_seed_check.json'
    temp = out.with_suffix('.pending')
    temp.write_text(json.dumps(rows, indent=2)+'\n', encoding='utf8')
    temp.replace(out)


def check_library():
    """Enumerate the same Duffing pool twice, changing only permission to start a term with trig."""
    pilot.limit_threads()
    from cpu_budget import await_capacity
    from noise_suite import trial
    from epde.operators.common import fill_prune
    await_capacity()
    audit = {}
    class Inspected(Exception):
        pass
    def inspect(pool, max_factors=None):
        family, = [f for f in pool.families if f.ftype == 'trigonometric']
        audit['original_meaningful'] = family.status['meaningful']
        for enabled in (False,True):
            with patch.dict(family.status, meaningful=enabled):
                lib = fill_prune._enumerate(pool,fill_prune._factors_cfg(max_factors))
                cosines = sorted({float(f.params[i]) for t in lib.values() if term_name(t)=='cos[0]'
                                  for f,i,_ in trial.trig_terms(t)})
                shapes = {term_name(t) for t in lib.values()}
                audit[str(enabled)] = dict(cosines=cosines,
                    all_duffing_terms={'cos[0]','u','u^3','du/dx0','d^2u/dx0^2'}.issubset(shapes))
        raise Inspected()
    with patch.object(fill_prune,'library_terms',inspect), tempfile.TemporaryDirectory(dir=pilot.ROOT) as temp:
        try:
            pilot.search('duffing','one',0,50,Path(temp)/'unused.json.gz',policy='first12')
        except Inspected:
            pass
        else:
            raise RuntimeError('Search did not request a term library')
    if audit['original_meaningful'] or audit['False']['cosines'] or not audit['True']['all_duffing_terms']:
        raise ValueError(audit)
    out = pilot.ROOT/'duffing_library_check.json'
    out.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')


def ablation_worker(arm):
    """Repeat draw zero with the trigonometric family allowed to start a term."""
    pilot.limit_threads()
    from cpu_budget import await_capacity
    from noise_suite import live
    from report import inspect
    await_capacity()
    baseline = pilot.ROOT/'results'/'duffing'/arm/'0'/'archive.json.gz'
    before = read_archive(baseline)
    with tempfile.TemporaryDirectory(dir=pilot.ROOT) as folder, open(os.devnull,'w') as quiet:
        path = Path(folder)/'archive.json.gz'
        with patch.object(live.U,'TrigonometricTokens',partial(live.U.TrigonometricTokens,meaningful=True)), redirect_stdout(quiet):
            pilot.search('duffing',arm,0,before['epochs'],path,policy='first12')
        after = read_archive(path)
        for key in ('samples','options','epochs','search_seed','working_population'):
            if before[key] != after[key]:
                raise ValueError(f'Mismatched {key} in the Duffing comparison')
        output = {}
        for key,archive in [('original',baseline),('standalone_trig',path)]:
            row, *_ = inspect(archive)
            output[key] = {k: row[k] for k in ('terms_generated','terms_on_front','generated','front_hit',
                                               'catalog_terms_profiled','frequency_profiles','evaluated')}
    return dict(arm=arm,draw=0,**output)


def run_ablation(arm):
    child = subprocess.run([sys.executable,__file__,'--ablation-worker',arm],env=dict(os.environ,PYTHONHASHSEED='0'),
                           text=True,capture_output=True)
    if child.returncode:
        raise RuntimeError(child.stderr or child.stdout)
    return json.loads(child.stdout.splitlines()[-1])


if __name__ == '__main__':
    if sys.argv[1:] == ['--seeds']:
        check_seeds()
    elif sys.argv[1:] == ['--library']:
        check_library()
    elif sys.argv[1:] == ['--raw-fit']:
        row = replay('duffing','diverse5',1,detailed_fit=True)
        (pilot.ROOT/'duffing_raw_fit.json').write_text(json.dumps(row['raw_details'],indent=2,allow_nan=False)+'\n',encoding='utf8')
    elif sys.argv[1:] == ['--ablation']:
        with pilot.FileLock(pilot.ROOT/'campaign.lock',timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(run_ablation,pilot.ARMS))
        target = pilot.ROOT/'duffing_term_start_ablation.json'
        target.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    elif len(sys.argv)==3 and sys.argv[1]=='--ablation-worker':
        print(json.dumps(ablation_worker(sys.argv[2]),allow_nan=False))
    elif len(sys.argv)>1 and sys.argv[1]=='--worker':
        print(json.dumps(replay(sys.argv[2],sys.argv[3],int(sys.argv[4])),allow_nan=False))
    else:
        with (pilot.ROOT/'runs.csv').open() as f:
            misses = [r for r in csv.DictReader(f) if r['dataset'] in ('ode','duffing') and not int(r['generated'])]
        with pilot.FileLock(pilot.ROOT/'campaign.lock',timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            rows = []
            target = pilot.ROOT/'failure_diagnosis.json'
            if sys.argv[1:] == ['--resume'] and target.is_file():
                rows = [r for r in json.loads(target.read_text(encoding='utf8'))
                        if r.get('exact_replay') and all('unavailable' in b for b in r['beam_checks'])]
                done = {(r['dataset'],r['arm'],r['draw']) for r in rows}
                misses = [r for r in misses if (r['dataset'],r['arm'],int(r['draw'])) not in done]
            pending = target.with_suffix('.pending')
            for future in as_completed([pool.submit(run,row) for row in misses]):
                rows.append(future.result())
                rows.sort(key=lambda r:(r['dataset'],r['arm'],r['draw']))
                pending.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf8')
                pending.replace(target)
