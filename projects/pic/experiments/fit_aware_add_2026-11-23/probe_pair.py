"""Replay the archived joint ranking on the native weak windows after discovery."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import combinations
import json
from types import SimpleNamespace

from probe_add import check, target_gene
from operator_trial import PARTS, ROOT, SEEDS, folder
from search import U, np


def rankings(seed, part, search, guide, library, wanted, factors):
    import arm_pvpa as pair

    with (ROOT/'pair_results.csv').open() as stream:
        budgets = {int(r['budget']) for r in csv.DictReader(stream)}
    budget, = budgets
    events = json.loads((folder('pvpa', part)/f'operator_{seed}.json').read_text())['pair_events']
    parents = Counter(e['parents'][0] for e in events if e['after_call'] < budget
                      and e['target'] == wanted[1] and len(e['added']) == 2
                      and len(e['parents']) == 1 and e['parents'][0] not in wanted[2])
    # Resolve a tie by label and select the most visited one-predictor, off-law parent.
    frequent = min(parents, key=lambda label: (-parents[label], label))
    by_label = {U.term_key(t): t for t in library}
    target = by_label[wanted[1]]
    rows = []
    for predictors in ([], [frequent]):
        equation = SimpleNamespace(structure=[target, *(by_label[k] for k in predictors)],
                                   pool=search.pool, main_var_to_explain='u', _plain_genes=target_gene(target),
                                   metaparameters={'max_factors_in_term': {'value': factors}})
        design, reason = guide.child_design(equation, frozenset())
        assert reason is None and design['target_term'] is target
        candidates, gains, top, _ = guide.stage1(equation, design, frozenset(), m=pair.TOP_SINGLES)
        indices = {U.term_key(t): i for i, (t, _) in enumerate(candidates)}
        physical = [indices[k] for k in wanted[2]]
        single_order = list(np.argsort(-gains, kind='stable'))
        rows.append(dict(seed=seed, part=part, parent=predictors[0] if predictors else '(target only)',
                         observed_pair_opportunities=parents[frequent] if predictors else
                         sum(e['after_call'] < budget and e['target'] == wanted[1] and len(e['added']) == 2
                             and not e['parents'] for e in events),
                         u_uxx_single_rank=single_order.index(indices['d^2u/dx1^2*u'])+1,
                         ux_squared_single_rank=single_order.index(indices['du/dx1^2'])+1,
                         both_in_top64=int(all(i in top for i in physical)), joint_rank=None))
        if not rows[-1]['both_in_top64']:
            continue
        L = np.column_stack([candidates[i][1] for i in top])
        pairs = list(combinations(range(len(top)), 2))
        slot = pairs.index(tuple(sorted(top.index(i) for i in physical)))
        X, y, sw = (design[k] for k in ('X', 'y', 'sw'))
        states = [False] + ([True] if X.shape[1]+3 <= guide.SC._config()['cap'] else [])
        state_gains = []
        for intercept in states:
            basis = guide.SC.gain_basis(y, X, sw, intercept)
            values, degenerate = pair.pair_gains_on_basis(basis, L, pairs)
            assert not degenerate[slot]
            # Independent check of the physical pair against two direct weighted fits.
            A = np.column_stack([X, np.ones(len(y))]) if intercept else X
            B = np.column_stack([A, L[:, pairs[slot]]])
            yw = y*basis['s']
            def rss(matrix):
                weighted = matrix*basis['s'][:, None]
                residual = yw - weighted @ np.linalg.lstsq(weighted, yw, rcond=None)[0]
                return residual @ residual
            np.testing.assert_allclose(values[slot], rss(A)-rss(B), rtol=1e-8, atol=1e-12)
            state_gains.append(np.where(degenerate, -np.inf, values))
        joint = np.max(state_gains, axis=0)
        assert joint[slot] > 0
        rank = 0
        for k in np.argsort(-joint, kind='stable'):
            a, b = (candidates[top[i]][0] for i in pairs[k])
            if pair.pair_keeps_target(a, b, equation, target, frozenset()):
                rank += 1
                if k == slot:
                    rows[-1]['joint_rank'] = rank
                    break
        assert rows[-1]['joint_rank'] is not None
    return rows


def run(job):
    return check(job, pair_probe=rankings)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = [r for group in pool.map(run, [(s, p) for s in SEEDS for p in PARTS]) for r in group]
    with (ROOT/'pair_shortlist.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
