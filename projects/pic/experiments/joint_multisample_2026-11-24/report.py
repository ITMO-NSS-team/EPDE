"""Describe structural recurrence and coefficient spread over entire multisample fronts."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import re
from statistics import median

from filelock import FileLock
from pilot import ARMS, DATASETS, DRAWS, POLICIES, ROOT, limit_threads, output_root


def describe_term(label):
    factors, tones = [], []
    for name, params in label:
        params = dict(params)
        if 'freq' in params:
            tones.append(params.pop('freq'))
        power = params.get('power', 1)
        dim = f"[{params['dim']:g}]" if 'dim' in params else ''
        factors.append(name + dim + (f'^{power:g}' if power != 1 else ''))
    return '*'.join(sorted(factors)) or '1', tuple(tones)


def aligned(pairs):
    """Give matching supports a common sign, scale and frequency-independent parameter names."""
    terms = sorted((describe_term(label), float(w)) for label,w in pairs)
    counts = Counter(t for (t,_),_ in terms)
    seen = Counter()
    coefficients, frequencies = {}, {}
    for (term, tones), weight in terms:
        seen[term] += 1
        key = term if counts[term] == 1 else f'{term}#{seen[term]}'
        coefficients[key] = weight
        for i,tone in enumerate(tones):
            frequencies[f'{key}:omega{i+1}'] = tone
    # A system can express a catalog equation in either variable slot.
    # Pick its highest-order standalone derivative, independent of that slot.
    derivatives = [(int(match[1] or 1), key) for key in coefficients
                   if (match := re.fullmatch(r'd(?:\^(\d+))?\w+/dx\d+(?:\^\d+)?', key))]
    pivot = min(derivatives, key=lambda item: (-item[0], item[1]))[1] if derivatives else min(coefficients)
    scale = -coefficients[pivot]
    values = {k:v/scale for k,v in coefficients.items() if k!=pivot}
    return tuple(sorted(coefficients)), pivot, dict(values, **frequencies)


def verify_receipt(path):
    with path.open('rb') as raw:
        actual = hashlib.file_digest(raw, 'sha256').hexdigest()
    if actual != path.with_name('sha256.txt').read_text().strip():
        raise ValueError(f'Archive checksum mismatch at {path}')


def objective_vectors(data, combos):
    import numpy as np
    points = data['points']
    errors = np.array([pt[5] for pt in points])
    assert errors.shape == (len(points), ARMS[data['arm']])
    return np.column_stack([row[combos] for row in errors.T] + [np.array([pt[4] for pt in points])[combos]])


def shape_matches(data, combos, catalog_shapes):
    import numpy as np
    names = [describe_term(label)[0] for label in data['labels']]
    supports = [tuple(sorted(names[j] for j, _ in pt[2])) for pt in data['points']]
    return np.array([tuple(sorted(supports[i] for i in combo)) in catalog_shapes for combo in combos])


def combined_front(baseline, data, vector, matches, problem, catalog_shapes):
    import numpy as np
    from compare import structural_matches, system_front
    old_systems = np.array(baseline['systems'], int)
    old_vectors = objective_vectors(baseline, old_systems)
    old_front = np.flatnonzero(system_front(old_vectors))
    np.testing.assert_array_equal(old_front, baseline['front'])
    old_combos = old_systems[old_front]
    old_matches, _, _ = structural_matches(data['dataset'], problem, baseline, old_combos)
    old_matches &= shape_matches(baseline, old_combos, catalog_shapes)

    front = data['front']
    values = np.vstack((old_vectors[old_front], vector[front]))
    flags = np.r_[old_matches, matches[front]]
    references = [(baseline['frequency_policy'], int(i)) for i in old_front]
    references.extend((data['frequency_policy'], i) for i in front)
    # By transitivity, a point dominated in either run is dominated in the union.
    return [dict(dataset=data['dataset'], arm=data['arm'], draw=data['draw'], policy=references[i][0],
                 system_index=references[i][1], objectives=json.dumps(values[i].tolist()), catalog_match=int(flags[i]))
            for i in np.flatnonzero(system_front(values))]


def inspect(path, baseline_path=None, paired_members=None):
    import numpy as np
    from noise_suite import live
    from compare import structural_matches,system_front
    with gzip.open(path,'rt',encoding='utf8') as f:
        data=json.load(f)
    if baseline_path is not None:
        verify_receipt(baseline_path)
        with gzip.open(baseline_path, 'rt', encoding='utf8') as f:
            baseline = json.load(f)
        if data.get('frequency_policy') == 'systematic':
            if baseline.get('frequency_policy') != 'per_support' or 'fill_probability' in baseline['options']['system_rps_params']:
                raise ValueError(f'Unexpected proposal policy at {baseline_path}')
            # The sole option change disables the residual-ranked fill beam.
            baseline['options']['system_rps_params']['fill_probability'] = 0.
        for field in ('dataset', 'arm', 'draw', 'samples', 'search_seed', 'options', 'epochs', 'working_population'):
            if baseline[field] != data[field]:
                raise ValueError(f'Unmatched {field} between {path} and {baseline_path}')
    name, arm, draw = data['dataset'], data['arm'], data['draw']
    p=live.load_data(name)
    points, labels = data['points'], data['labels']
    combos = np.array(data['systems'],int)
    front=np.array(data['front'],int)
    vector = objective_vectors(data, combos)
    np.testing.assert_array_equal(front,np.flatnonzero(system_front(vector)))
    matches, _, _ = structural_matches(name,p,data,combos)
    profiles=data['frequency_profiles']
    policy = data.get('frequency_policy','first12')
    if data['epochs'] > 1 and ((policy=='first12' and not profiles) or data.get('working_population') != 8):
        raise ValueError(f'Incomplete frequency or population budget at {path}')
    evaluated={tuple(c) for c in data['systems']}
    if policy != 'first12':
        fitted = {i for p in profiles for i in p['fitted_points']}
        source_points = {p['point'] for p in profiles}
        if not all(0 <= i < len(points) for i in fitted | source_points):
            raise ValueError(f'Untracked frequency fit at {path}')
        children = {c for c in evaluated if any(i in fitted for i in c)}
    else:
        children={tuple(child) for profile in profiles for child in profile['children']}
        if not children.issubset(evaluated) or any(tuple(p['parent']) not in evaluated for p in profiles):
            raise ValueError(f'Untracked frequency proposal at {path}')
    reference={}
    catalog_shapes=set()
    for catalog_system in p.truth_systems:
        catalog_supports=[]
        for law in catalog_system:
            target,terms,coefs=live.R.parse(law)
            key,pivot,params=aligned([(target,-1.),*zip(terms,coefs)])
            reference[(key,pivot)]=params
            catalog_supports.append(tuple(sorted(describe_term(t)[0] for t in (target, *terms))))
        catalog_shapes.add(tuple(sorted(catalog_supports)))
    shapes = shape_matches(data, combos, catalog_shapes)
    # The catalog matcher bins nearby frequencies. Two separate terms in
    # the same frequency bin must still have the catalog's multiplicity.
    matches &= shapes
    catalog_term_systems={tuple(combo) for combo in combos[shapes]}
    if policy != 'first12':
        catalog_terms_profiled=any(i in source_points or i in fitted for c in catalog_term_systems for i in c)
    else:
        catalog_terms_profiled=any(tuple(p['parent']) in catalog_term_systems for p in profiles)
    forms, parameters, physical = set(), defaultdict(list), defaultdict(list)
    for idx in front:
        system = []
        for pt in combos[idx]:
            variable,_,pairs=points[pt][:3]
            support,pivot,params=aligned([(labels[j],w) for j,w in pairs])
            system.append((support,pivot,variable,params))
        system.sort(key=lambda eq:eq[:3])
        structure=json.dumps([eq[0] for eq in system],separators=(',',':'))
        forms.add(structure)
        for slot,(support,pivot,var,params) in enumerate(system):
            for term,value in params.items():
                key=(structure,slot,pivot,term)
                parameters[key].append(value)
                if matches[idx]:
                    exact=reference[(support,pivot)][term]
                    physical[(structure,slot,pivot,term,exact)].append(value)
    row=dict(dataset=name,arm=arm,draw=draw,generated=int(matches.any()),front_hit=int(matches[front].any()),
             terms_generated=int(shapes.any()),terms_on_front=int(shapes[front].any()),
             catalog_terms_profiled=int(catalog_terms_profiled),
             front_size=len(front),front_structures=len(forms),evaluated=len(combos),epochs=data['epochs'],
             frequency_profiles=len(profiles),profiled_children=len(children),
             profiled_children_on_front=sum(tuple(combos[i]) in children for i in front),
             seconds=round(data['seconds'],2),generations=None,admission_cpu_pct=None)
    if policy != 'first12':
        row['last_profile_epoch'] = max((p['epoch'] for p in profiles), default=None)
        row['profile_cache_hits'] = data['profile_cache_hits']
    if policy == 'systematic':
        row.update(data['structure_proposals'])
        row['multimodal_bands'] = sum(len(p['proposals']) > 1 for p in profiles)
        row['proposed_frequencies'] = sum(len(p['proposals']) for p in profiles)
        row['combined_front_hit'] = None
        if baseline_path is not None:
            combined = combined_front(baseline, data, vector, matches, p, catalog_shapes)
            row['combined_front_hit'] = int(any(member['catalog_match'] for member in combined))
            if paired_members is not None:
                paired_members.extend(combined)
    log=path.with_name('worker.log')
    if log.is_file():
        trace=log.read_text(encoding='utf8',errors='replace')
        epochs_run=re.findall(r'NSGA-II : (\d+)-th epoch \((\d+) generations\)',trace)[-row['epochs']:]
        if [int(index) for index,_ in epochs_run] != list(range(1,row['epochs']+1)):
            raise ValueError(f'Incomplete epoch log at {log}')
        row['generations']=sum(int(count) for _,count in epochs_run)
        admissions=re.findall(r'Admitted at ([\d.]+)% host CPU',trace)
        if admissions:
            row['admission_cpu_pct']=float(admissions[-1])
    return row,forms,parameters,physical,data['samples']


def write_table(root, name, rows, fields):
    dest=root/name
    pending=dest.with_suffix('.pending')
    open_file=gzip.open if name.endswith('.gz') else open
    with open_file(pending,'wt',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(dest)


def render(partial=False, policy='systematic'):
    from noise_suite import epochs
    root=output_root(policy)
    rows,struct_rows,param_rows,truth_rows,groups=[],[],[],[],[]
    hashes={}
    baseline = {}
    paired_members = []
    previous = 'per_support' if policy == 'systematic' else 'first12'
    prior = output_root(previous)
    if policy != 'first12' and (prior/'runs.csv').is_file():
        with (prior/'runs.csv').open(encoding='utf8') as stream:
            baseline = {(r['dataset'], r['arm'], int(r['draw'])): r for r in csv.DictReader(stream)}
    for name in DATASETS:
        for arm in ARMS:
            runs, counts, ranges, physical = [],Counter(),defaultdict(dict),defaultdict(dict)
            for draw in DRAWS:
                path=root/'results'/name/arm/str(draw)/'archive.json.gz'
                receipt=path.with_name('sha256.txt')
                if partial and not receipt.is_file():
                    continue
                verify_receipt(path)
                baseline_path = prior/path.relative_to(root) if (name, arm, draw) in baseline else None
                r,forms,parameters,observed,metadata=inspect(path, baseline_path, paired_members)
                assert (r['dataset'],r['arm'],r['draw'],r['epochs']) == (name,arm,draw,epochs(name))
                assert len(metadata)==ARMS[arm]
                for i,rec in enumerate(metadata):
                    state=i if arm.startswith('diverse') else 0
                    assert (rec['state'],rec['noise_seed'])==(state,948000+10*draw+i)
                    k=(name,draw,i,state)
                    assert hashes.setdefault(k,rec)==rec
                rows.append(r)
                runs.append(r)
                counts.update(forms)
                for key,values in parameters.items():
                    ranges[key][draw]=(min(values),max(values))
                for key,values in observed.items():
                    exact=key[-1]
                    errors=[abs(v-exact) for v in values]
                    physical[key][draw]=(min(values),max(values),min(errors),max(errors))
            if not runs:
                continue
            base=dict(dataset=name,arm=arm)
            groups.append(dict(**base,draws=len(runs),generated=sum(r['generated'] for r in runs),
                               front_hits=sum(r['front_hit'] for r in runs),
                               terms_generated=sum(r['terms_generated'] for r in runs),
                               terms_on_front=sum(r['terms_on_front'] for r in runs),
                               median_front_size=median(r['front_size'] for r in runs),
                               recurrent_structures=sum(n>=2 for n in counts.values())))
            struct_rows.extend(dict(**base,structure=s,draws=count) for s,count in sorted(counts.items()))
            for (structure,slot,pivot,term),by_draw in sorted(ranges.items()):
                if len(by_draw)<2:
                    continue
                joint_min=min(lo for lo,hi in by_draw.values())
                joint_max=max(hi for lo,hi in by_draw.values())
                for draw,(lo,hi) in sorted(by_draw.items()):
                    param_rows.append(dict(**base,structure=structure,slot=slot,pivot=pivot,term=term,
                                           draw=draw,draws=len(by_draw),within_min=lo,within_max=hi,
                                           across_min=joint_min,across_max=joint_max))
            for (structure,slot,pivot,term,exact),by_draw in sorted(physical.items()):
                for draw,(lo,hi,min_abs,max_abs) in sorted(by_draw.items()):
                    truth_rows.append(dict(**base,structure=structure,slot=slot,pivot=pivot,term=term,exact=exact,draw=draw,
                                           draws=len(by_draw),within_min=lo,within_max=hi,min_abs_error=min_abs,
                                           max_abs_error=max_abs,across_min=min(v[0] for v in by_draw.values()),
                                           across_max=max(v[1] for v in by_draw.values())))
    if not rows:
        return
    write_table(root,'runs.csv',rows,rows[0].keys())
    write_table(root,'by_configuration.csv',groups,groups[0].keys())
    write_table(root,'structure_recurrence.csv.gz',struct_rows,['dataset','arm','structure','draws'])
    write_table(root,'coefficient_ranges.csv.gz',param_rows,
                'dataset arm structure slot pivot term draw draws within_min within_max across_min across_max'.split())
    write_table(root,'benchmark_parameter_errors.csv',truth_rows,
                'dataset arm structure slot pivot term exact draw draws within_min within_max min_abs_error max_abs_error across_min across_max'.split())
    lines=['# Joint multisample Pareto discovery','',f'{len(rows)} of 100 searches verified. Each front contains '
           'every nondominated system among the eligible fits evaluated in its run. All supplied recordings contributed to discovery.','',
           'The design uses 5% correlated measurement noise (AR(1), lag-one correlation 0.8), up to five matched noise draws per arm, '
           f'and one optimizer seed per benchmark. The [protocol]({"../" if root != ROOT else ""}protocol.md) describes the shared frequency fit and Pareto comparison.', '']
    if baseline:
        lines += comparison_lines(root, rows, baseline, previous, policy)
    if paired_members:
        write_table(root, 'combined_fronts.csv.gz', paired_members, paired_members[0].keys())
    lines += [
           '| System | Recordings | Law generated | Law on front | Median front size | Structures on at least two fronts |',
           '|---|---|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['dataset']} | {g['arm']} | {g['generated']}/{g['draws']} | "
                     f"{g['front_hits']}/{g['draws']} | {g['median_front_size']:g} | {g['recurrent_structures']} |")
    lines+=['','The arm names give the recording count. Repeat arms use independent measurement noise on the same initial condition. '
            'Diverse arms also vary the initial conditions. Generation counts a catalog system anywhere in the evaluated pool. '
            'Recovery counts a catalog system on the complete front. A structure contributes at most once per draw to recurrence.',
            '', 'Coefficient and frequency ranges use all aligned fits of each recurring structure. '
            'Physical parameter errors are conditional on recovering a catalog structure on the full front. '
            'The five draws describe measurement sensitivity at one fixed optimizer seed per system.',
            '', '[Structure recurrence](structure_recurrence.csv.gz) · [Parameter ranges](coefficient_ranges.csv.gz) · '
            '[Benchmark parameter errors](benchmark_parameter_errors.csv) · [Individual searches](runs.csv)']
    if truth_rows:
        lines+=['','## Catalog parameter ranges','',
                'Each interval spans all front fits that recovered the indicated catalog system, over the draws where it appeared. '
                'The pivot coefficient is fixed at −1 in the zero-form equation. Angular frequencies use the original time units.','',
                '| System | Arm | Pivot | Term | Catalog value | Fitted range | Draws |',
                '|---|---|---|---|---:|---:|---:|']
        seen=set()
        for row in truth_rows:
            key=tuple(row[k] for k in ('dataset','arm','structure','slot','term'))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {row['dataset']} | {row['arm']} | `{row['pivot']}` | `{row['term']}` | "
                         f"{row['exact']:.5g} | {row['across_min']:.5g} to {row['across_max']:.5g} | {row['draws']} |")
    # Pair recovery by the same system and noise draw. These are counts of
    # whole-front events, independent of the number of fitted systems on a front.
    recoveries={(r['dataset'],r['draw'],r['arm']):r['front_hit'] for r in rows}
    contrasts=[('one','repeat3'),('one','repeat5'),('repeat3','diverse3'),('repeat5','diverse5'),
               ('repeat3','repeat5'),('diverse3','diverse5')]
    lines+=['','## Paired changes in recovery','',
            'Each comparison uses only completed pairs. A gain means the catalog system appeared on the second '
            'front alone. A loss means it appeared on the first front alone. Counts pool the four systems.','',
            '| First arm | Second arm | Paired searches | Gain | Loss | On both | On neither |',
            '|---|---|---:|---:|---:|---:|---:|']
    for first,second in contrasts:
        paired=[(recoveries[n,d,first], recoveries[n,d,second]) for n in DATASETS for d in DRAWS
                if (n,d,first) in recoveries and (n,d,second) in recoveries]
        lines.append(f'| {first} | {second} | {len(paired)} | {paired.count((0,1))} | {paired.count((1,0))} | '
                     f'{paired.count((1,1))} | {paired.count((0,0))} |')
    lines+=['','## Where recovery was lost','',
            'The counts below include all arms. A frequency miss means the search generated the full system of '
            'catalog terms, but never generated those terms with every frequency within the catalog tolerance. '
            'A front exclusion means the search generated a catalog match and every matching fit was dominated.','',
            '| System | Searches | Catalog terms never generated | Frequency miss | Front exclusion | Recovered |',
            '|---|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        runs=[r for r in rows if r['dataset']==name]
        lines.append(f"| {name} | {len(runs)} | {sum(1-r['terms_generated'] for r in runs)} | "
                     f"{sum(r['terms_generated']-r['generated'] for r in runs)} | "
                     f"{sum(r['generated']-r['front_hit'] for r in runs)} | {sum(r['front_hit'] for r in runs)} |")
    for name in ('ode', 'duffing'):
        misses=[r for r in rows if r['dataset']==name and r['terms_generated'] and not r['generated']]
        if misses:
            count=sum(r['catalog_terms_profiled'] for r in misses)
            lines+=['',f'For `{name}`, the frequency profiler reached the catalog term set in '
                    f'{count} of {len(misses)} frequency misses.']
    admissions=[r['admission_cpu_pct'] for r in rows if r['admission_cpu_pct'] is not None]
    if admissions:
        lines+=['','## Compute budget','',
                f'The worker logs record {sum(r["epochs"] for r in rows):,} epochs and '
                f'{sum(r["generations"] or 0 for r in rows):,} generations. Recorded host CPU load at admission '
                f'ranged from {min(admissions):.1f}% to {max(admissions):.1f}% across {len(admissions)} starts. '
                'The two-worker controller uses a six-core affinity mask and single-threaded numerical libraries.']
    bands = sum(r['frequency_profiles'] for r in rows)
    if policy == 'systematic':
        profile_summary = (f"The profiler searched {bands:,} equation-specific frequency bands and proposed "
                           f"{sum(r['proposed_frequencies'] for r in rows):,} frequencies. "
                           f"{sum(r['multimodal_bands'] for r in rows):,} bands yielded multiple minima. ")
    else:
        profile_summary = f"The profiler fitted a shared frequency {bands:,} times. "
    lines+=['',profile_summary +
            f"The search evaluated {sum(r['profiled_children'] for r in rows):,} distinct systems containing a profiled fit, "
            f"of which {sum(r['profiled_children_on_front'] for r in rows):,} appeared on the final fronts."]
    pending=root/'findings.pending'
    pending.write_text('\n'.join(lines)+'\n',encoding='utf8')
    pending.replace(root/'findings.md')


def comparison_lines(root, rows, baseline, previous, current):
    pairs = []
    for r in rows:
        key = r['dataset'], r['arm'], r['draw']
        if key not in baseline:
            continue
        old = baseline[key]
        pair = dict(zip(('dataset', 'arm', 'draw'), key))
        for field in ('terms_generated', 'generated', 'front_hit', 'seconds'):
            parse = float if field == 'seconds' else int
            pair[previous + '_' + field] = parse(old[field])
            pair[current + '_' + field] = r[field]
        if current == 'systematic':
            pair['combined_front_hit'] = r['combined_front_hit']
        pairs.append(pair)
    if not pairs:
        return []
    write_table(root, 'paired_policies.csv', pairs, pairs[0].keys())
    description = ('The revised policy fits every resolved frequency minimum and generates structural alternatives independently of residual rank.'
                   if current == 'systematic' else 'The revised policy also allows standalone sine and cosine terms.')
    lines = ['## Paired search policies', '',
             f'Each pair uses identical recording hashes, optimizer seed, epoch count and working population. {description} '
             'Recovery requires a catalog match on the complete Pareto front.', '']
    if current == 'systematic' and all(not p[previous+'_front_hit'] for p in pairs):
        lines += [f'These {len(pairs)} pairs were selected from searches that missed the catalog law on the previous front.', '']
    if current == 'per_support' and len(pairs) < 100 and all(p['dataset'] in ('ode', 'duffing') and not p['first12_generated'] for p in pairs):
        lines += [f'These {len(pairs)} completed pairs come from the 34 original forced-oscillator and Duffing generation failures. '
                  'This selection measures recovery of those misses. An overall recovery rate requires the other benchmark runs.', '']
    lines += ['| System | Pairs | Catalog terms generated, old → revised | Front recoveries, old → revised | Gained | Lost | Median search seconds, old → revised |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        group = [p for p in pairs if p['dataset'] == name]
        if group:
            def counts(field):
                return ' → '.join(str(sum(p[policy+'_'+field] for p in group)) for policy in (previous, current))
            hits = [(p[previous+'_front_hit'], p[current+'_front_hit']) for p in group]
            times = ' → '.join(f"{median(p[policy+'_seconds'] for p in group):.1f}" for policy in (previous, current))
            lines.append(f"| {name} | {len(group)} | {counts('terms_generated')} | {counts('front_hit')} | "
                         f"{hits.count((0,1))} | {hits.count((1,0))} | {times} |")
    late = sum(r['last_profile_epoch'] == r['epochs'] for r in rows)
    if current == 'systematic':
        hits = sum(p['combined_front_hit'] for p in pairs)
        lines += ['', f'The catalog law appears on {hits}/{len(pairs)} combined fronts after pooling all eligible systems '
                  'from both policies on each matched draw. The [complete combined fronts](combined_fronts.csv.gz) list '
                  'every surviving archive reference, including ties and membership in both archives.']
    lines += ['', f'New structures received frequency fits in the final epoch in {late}/{len(rows)} searches. '
              f"The profiler reused previously fitted structures {sum(r['profile_cache_hits'] for r in rows):,} times. "
              'Search times include fitting and optimization. [Matched runs](paired_policies.csv)', '',
              '## Recovery by recording configuration', '']
    return lines


if __name__=='__main__':
    limit_threads()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--frequency-policy',choices=POLICIES,default='systematic')
    args = parser.parse_args()
    root = output_root(args.frequency_policy)
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(root/'report.lock'):
        render(args.partial,args.frequency_policy)
