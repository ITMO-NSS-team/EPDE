"""Paired, row-matched test of one complementary weak geometry on the existing 15-system panel."""
import json
from pathlib import Path
import sys
from time import process_time


from cpu_limit import cap_worker


if __name__ == '__main__':
    cap_worker()

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'general_weak_holdout_2026-11-23'
sys.path.insert(0, str(ARCHIVE))
from screen import R, SYSTEMS, np, score, split
from epde.interface.search_config import load_search_config, set_active_config
from selector import select

WF = R.WF
SEEDS = range(923211, 923217)
CONFIRMATION = range(923217, 923223)
ARM_CPU_SECONDS = 30


def fronts(rows):
    slots = []
    for slot in sorted({r['slot'] for r in rows}):
        valid = [r for r in rows if r['slot'] == slot and r['status'] == 'ok']
        front = [r for r in valid if not any(v['held_wape'] <= r['held_wape'] and v['dyad'] <= r['dyad']
                 and (v['held_wape'] < r['held_wape'] or v['dyad'] < r['dyad']) for v in valid)]
        ref = next((r for r in valid if r['label'] == 'reference'), None)
        slots.append(dict(hit=any(r['registered'] for r in front),
                          additions=sum(r['label'].startswith('add_') and not r['registered'] for r in front),
                          exact=any(r['label'] == 'reference' for r in front),
                          front=[r['id'] for r in front], reference=ref))
    return slots


def evaluate(a, b, candidates, shape, include_truth=False):
    out = []
    for c in candidates:
        value = score(a, b, c, shape)
        if c['label'] == 'reference' and value['status'] == 'ok':
            yscale = np.linalg.norm(b[:, c['yi']])
            value['scales'] = (np.linalg.norm(b[:, c['xi']], axis=0)/yscale).tolist()
            if include_truth:
                assert len(c['exact']) == len(c['xi'])
                value['truth'] = list(c['exact'])
                value['terms'] = [str(term) for term in c['terms']]
        out.append(dict(id=c['id'], label=c['label'], registered=c['registered'], slot=c['slot'], **value))
    return out


def run(system, seed, *, select_geometry=select, panel_seed=None, choose_plan=None, include_truth=False):
    path = ARCHIVE/'confirmation'/'results'/f'{system}_{panel_seed if panel_seed is not None else seed}.json'
    archive = json.loads(path.read_text())
    problem = R.datasets.load(system)
    raw = problem.noisy(5, seed)
    _, (sa, aa), (sb, ab) = split(R.axes_of(problem))
    train = {k: v[sa] for k, v in raw.items()}
    hashes = {k: R.digest(v) for k, v in train.items()}
    if choose_plan is None:
        assert hashes == archive['train_hashes']
        plan = {k: tuple(v) for k, v in archive['plan'].items()}
    else:
        picked = choose_plan(train, aa)
        plan = {k: tuple(getattr(picked, k)) for k in ('q', 'ms', 'ss', 'cut', 'shape')}
    fields = WF.Fields(train, aa)
    t0 = process_time()
    primary, selected, audit = select_geometry(fields, plan, (2,)+(4,)*(fields.D-1))
    selection_seconds = process_time()-t0
    # The scoring panel and validation observations enter only once the choice is frozen.
    current, labels = R.panel(problem)
    by_id = {c['id']: c for c in current}
    candidates = [by_id[r['id']] for r in archive['rows']]
    # The external registry can acquire new equivalents while experiments run. Replay the frozen panel.
    used = sorted({j for c in candidates for j in [c['yi'], *c['xi']]})
    index = {old: new for new, old in enumerate(used)}
    labels = [labels[j] for j in used]
    candidates = [dict(c, yi=index[c['yi']], xi=[index[j] for j in c['xi']]) for c in candidates]
    terms, opaque = R.symbols(problem, labels)
    fields.opaque = {k: v[sa] for k, v in opaque.items()}
    held = WF.Fields({k: v[sb] for k, v in raw.items()}, ab, {k: v[sb] for k, v in opaque.items()})
    t0 = process_time()
    a = primary.matrix(fields, terms)
    b = primary.matrix(held, terms)
    const = labels.index(()) if () in labels else None
    if const is not None:
        for mat in (a, b):
            np.testing.assert_allclose(mat[:, const], 1., atol=1e-12)
            mat[:, const] = 1.
    common_seconds = process_time()-t0
    t0 = process_time()
    baseline = evaluate(a, b, candidates, plan['shape'], include_truth)
    base_seconds = process_time()-t0
    for got, expected in zip(baseline, archive['rows'], strict=True):
        assert (got['id'], got['registered']) == (expected['id'], expected['registered'])
        if choose_plan is None:
            assert got['status'] == expected['status'], (system, seed, got['id'])
        if choose_plan is None and got['status'] == 'ok':
            np.testing.assert_allclose([got['held_wape'], got['dyad']], [expected['held_wape'], expected['dyad']],
                                       rtol=1e-7, atol=1e-10, err_msg=f"{system}/{seed}/{got['id']}")
    if selected is None:
        alternative, extra_seconds = baseline, base_seconds
    else:
        t0 = process_time()
        mixed = selected.matrix(fields, terms, a)
        if const is not None:
            np.testing.assert_allclose(mixed[:, const], 1., atol=1e-12)
            mixed[:, const] = 1.
        assert mixed.shape == a.shape and np.array_equal(mixed[~selected.replace], a[~selected.replace])
        alternative = evaluate(mixed, b, candidates, plan['shape'], include_truth)
        extra_seconds = process_time()-t0
    times = dict(native=common_seconds+base_seconds, complementary=common_seconds+selection_seconds+extra_seconds)
    assert max(times.values()) <= ARM_CPU_SECONDS, (system, seed, times)
    return dict(system=system, seed=seed, train_hashes=hashes, plan=plan, rows=len(a), candidate_count=len(candidates),
                changed_rows=int(selected.replace.sum()) if selected else 0, selector=audit, cpu_seconds=times,
                second=dict(ms=selected.ms, shift=selected.shift) if selected else None,
                native=fronts(baseline), complementary=fronts(alternative))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true', help='run the second six noise draws per trajectory')
    parser.add_argument('--both', action='store_true', help='run both cohorts in one worker')
    args = parser.parse_args()
    set_active_config(load_search_config(overrides=dict(evaluation_form='weak', system_rps_mode='plain',
                                                       instability_metric='dyad', equation_terms_max_number=7)))
    for system in ['kdv_cossin', *(s for s in SYSTEMS if s != 'kdv_cossin')]:
        seeds = (*SEEDS, *CONFIRMATION) if args.both else CONFIRMATION if args.confirm else SEEDS
        for seed in seeds:
            case = run(system, seed)
            path = ROOT/'results'/f'{system}_{seed}.json'
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix('.pending')
            tmp.write_text(json.dumps(case, indent=1, allow_nan=False), encoding='utf8')
            tmp.replace(path)
        print(f'{system}: {len(seeds)} paired recordings complete', flush=True)
