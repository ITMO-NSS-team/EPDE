"""Compare full-resolution WLS and homogeneous TLS on identical weak measurements."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'core15_implicit_correction_2026-11-18'
sys.path[:0] = [str(ROOT.parent / 'implicit_objective_2026-11-18'),
                str(ROOT.parent / 'core15_residual_products_2026-11-16')]
import run_screen as R
from records import SYSTEMS, sha
from implicit import fit, projective_error
import numpy as np
from scipy import linalg

METHODS = ('wls', 'tls')
SPLITS = ('blocked_energy', 'record_cross', 'blocked_cross')


def matrix_for(bank, fields, segment, const):
    matrix = bank.matrix({k: v[segment] for k, v in fields.items()})
    if const is not None:
        np.testing.assert_allclose(matrix[:, const], 1., atol=1e-12)
        matrix[:, const] = 1.
    return matrix


class Panel:
    def __init__(self, system):
        base = 'ac' if system == 'ac_zero' else system
        problem = R.datasets.load(base)
        if system == 'ac_zero':
            t = problem.grids[0]
            initial = problem.data['u'][:1]
            denom = 1 + initial**2 * np.expm1(10*t)
            field = initial * np.exp(5*t) / np.sqrt(denom)
            deriv = 5*initial*np.exp(5*t)*(1-initial**2)/denom**1.5
            np.testing.assert_allclose(deriv, 5*field*(1-field**2), atol=2.e-14)
            problem = replace(problem, data={'u': field})
        self.problem = problem
        self.cs, labels = R.panel(problem)
        self.const = labels.index(()) if () in labels else None
        axes = R.axes_of(problem)
        n = len(axes[0])
        self.slices = (slice(None), slice(0, n//2), slice(n//2, n))
        terms, opaque = R.symbols(problem, labels)
        self.width = max(len(c['xi']) for c in self.cs)
        meta = json.loads((ARCHIVE/'records'/f'{base}_73001.json').read_text())
        assert json.loads(json.dumps(labels)) == meta['labels']
        plans = [Path(p) for p in meta['inputs'] if 'core15_residual_products' in p]
        plan_path, = plans
        assert sha(plan_path) == meta['inputs'][str(plan_path)]
        old = {k: tuple(v) for k, v in json.loads(plan_path.read_text())['base_plan'].items()}
        self.dyad_bank = R.LocalBank(axes, terms, opaque, old)
        time_cut = old['cut'][0]
        radius = min(old['ms'][0], max(old['q'][0], (n//2 - 2*time_cut - 1)//4))
        if 2*radius + 1 > n//2 - 2*time_cut:
            raise ValueError(f'Insufficient raw time samples on {system}')
        self.banks = []
        for seg in self.slices:
            ax = [axes[0][seg], *axes[1:]]
            ms = (radius, *old['ms'][1:])
            ss = (min(old['ss'][0], radius), *old['ss'][1:])
            shape = tuple(1+(len(a)-1-2*c-2*m)//s for a,c,m,s in zip(ax, old['cut'], ms, ss))
            plan = dict(old, ms=ms, ss=ss, shape=shape)
            bank = R.LocalBank(ax, terms, {k: v[seg] for k,v in opaque.items()}, plan)
            for mono in bank.monos:
                R.WF.check_cut(mono, len(ax), old['cut'])
            self.banks.append(bank)
        self.clean_scale = np.sqrt(np.mean(matrix_for(self.banks[0], problem.data, slice(None), self.const)**2, axis=0))
        self.source_plan = str(plan_path.relative_to(ROOT.parent))
        self.code = SYSTEMS.index(base)
        # Same dyadic slab floor as the archived comparator, applied to each arm.
        cfg = R.active_config()
        R.set_active_config(replace(cfg, search_space=replace(cfg.search_space,
                            system_rps_mode='plain', equation_terms_max_number=self.width)))
        assert R.SV._reference_width(1) == self.width
        if system != 'ac_zero':
            previous = problem.noisy(5, 73001)
            assert all(R.digest(arr) == meta['raw_hash'][k] for k, arr in previous.items())
            replay = matrix_for(self.dyad_bank, previous, slice(None), self.const)
            with np.load(ARCHIVE/'records'/f'{base}_73001.npz') as z:
                np.testing.assert_allclose(replay, z['dyad_matrix'], rtol=1e-12, atol=2e-13)
            ref = next(c for c in self.cs if c['label'] == 'reference')
            archived = json.loads((ARCHIVE/'results'/f'{base}_73001.json').read_text())['rows']
            expected, = [r['legacy_dyad'] for r in archived if r['id'] == ref['id']]
            np.testing.assert_allclose(self.dyad(replay, ref), expected, rtol=1e-8, atol=1e-10)

    def dyad(self, matrix, c):
        if not c['xi']:
            return 0.
        return float(np.sum(R.SV.dyad_scores(matrix[:, c['xi']], matrix[:, c['yi']], np.ones(len(matrix)),
                                             self.dyad_bank.plan['shape'], fit_intercept=False)))

    def recording(self, anchor, repeat):
        rng = np.random.default_rng([886701, self.code, anchor, repeat])
        return {k: v+.05*v.std()*rng.standard_normal(v.shape) for k,v in self.problem.data.items()}

    def matrices(self, fields):
        return [matrix_for(b, fields, s, self.const) for b,s in zip(self.banks,self.slices)]


def wls(matrix, pivot):
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        return None, 'zero training column'
    normalized = matrix / norms
    predictors = [j for j in range(len(norms)) if j != pivot]
    vector = np.ones(len(norms))
    if predictors:
        x = normalized[:, predictors]
        tol = 64*np.finfo(float).eps*max(x.shape)
        b, _, rank, _ = np.linalg.lstsq(x, normalized[:, pivot], rcond=tol)
        if rank != len(predictors):
            return None, 'rank-deficient WLS predictors'
        vector[predictors] = -b
    return vector / norms, 'ok'


def evaluate(matrix, a, b, ids, vector):
    train = matrix[:, ids]
    rms = np.linalg.norm(train, axis=0)/np.sqrt(len(train))
    unit = vector / np.linalg.norm(rms*vector)
    return float(np.mean((a[:, ids] @ unit)*(b[:, ids] @ unit)))


def scores(c, trains, test_a, test_b, clean_scale):
    ids = sorted([c['yi'], *c['xi']])
    pivot = ids.index(c['yi'])
    results = {}
    for name in METHODS:
        vectors, reasons = [], []
        for training in trains:
            m = training[:, ids]
            vector, reason = wls(m, pivot) if name == 'wls' else fit(m)
            vectors.append(vector)
            reasons.append(reason)
        d = dict.fromkeys(SPLITS)
        if vectors[0] is not None:
            d['record_cross'] = evaluate(trains[0], test_a[0], test_b[0], ids, vectors[0])
        if all(v is not None for v in vectors[1:]):
            d['blocked_energy'] = float(np.mean([evaluate(trains[g], trains[3-g], trains[3-g], ids, vectors[g])
                                                 for g in (1,2)]))
            d['blocked_cross'] = float(np.mean([evaluate(trains[g], test_a[3-g], test_b[3-g], ids, vectors[g])
                                                for g in (1,2)]))
        angle = None
        if c['label'] == 'reference' and vectors[0] is not None:
            truth = np.ones(len(ids))
            for j, coefficient in zip(c['xi'], c['exact']):
                truth[ids.index(j)] = -coefficient
            angle = float(np.degrees(projective_error(vectors[0], truth, clean_scale[ids])))
        results[name] = dict(scores=d, status=reasons, vectors=[v.tolist() if v is not None else None for v in vectors],
                             angle=angle)
    return ids, results


def checks(matrix, c, results):
    """Independent LAPACK fit and a signed rescaling/permutation of each TLS relation."""
    ids = sorted([c['yi'],*c['xi']])
    m = matrix[:, ids]
    scale = np.linalg.norm(m, axis=0)
    if np.any(scale == 0):
        return
    unit = np.geomspace(.03, 29, len(ids)) * (-1)**np.arange(len(ids))
    v = results['tls']['vectors'][0]
    shuffled, status = fit((m*unit)[:, ::-1])
    assert status == results['tls']['status'][0]
    if v is not None:
        alt = linalg.svd(m/scale, full_matrices=False, lapack_driver='gesvd')[2][-1]/scale
        assert projective_error(np.array(v), alt, scale) < 1e-7
        assert projective_error(np.array(v), shuffled[::-1]*unit, scale) < 1e-7
    v = results['wls']['vectors'][0]
    if v is not None:
        pivot = ids.index(c['yi'])
        keep = np.arange(len(ids)) != pivot
        if np.any(keep):
            a = m[:, keep] / scale[keep]
            coef = linalg.lstsq(a, m[:, pivot]/scale[pivot], lapack_driver='gelsy')[0]
            expected = np.ones(len(ids))
            expected[keep] = -coef
            assert projective_error(np.array(v), expected/scale, scale) < 1e-7


def study(system):
    panel = Panel(system)
    root = ROOT/'results'
    clean = panel.problem.data
    shape = next(iter(clean.values())).shape
    rows = []
    manifests = []
    for anchor in range(12):
        fields = [panel.recording(anchor, r) for r in range(3)]
        manifests.append([{k: R.digest(v) for k,v in f.items()} for f in fields])
        train, a, b = [panel.matrices(f) for f in fields]
        dyad_matrix = matrix_for(panel.dyad_bank, fields[0], slice(None), panel.const)
        for c in panel.cs:
            diagnostic = dict(c, exact=[0., *c['exact'][1:]]) if system == 'ac_zero' and c['label']=='reference' else c
            if diagnostic is not c:
                assert c['exact'][0] == .0001
            ids, result = scores(diagnostic, train, a, b, panel.clean_scale)
            if anchor in (0,6):
                checks(train[0], c, result)
            rows.append(dict(anchor=anchor, id=c['id'], slot=c['slot'], registered=c['registered'],
                             label=c['label'], ids=ids, dyad=panel.dyad(dyad_matrix, c), methods=result))
    R.dump(root/f'{system}.json', dict(system=system, shape=shape, raw_hashes=manifests,
                                      base_plan=panel.source_plan, plans=[b.plan for b in panel.banks], rows=rows))
    print(system, 'complete', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('systems', nargs='*', default=[*SYSTEMS, 'ac_zero'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(study, args.systems))
