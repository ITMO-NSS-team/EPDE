"""Frozen-coefficient probe of oscillating weak tests, with no equation solves.

Select a frequency bandwidth from raw u by GCV for nested orthonormal DCT-II
projections. Use every grid frequency from zero through that bandwidth, for
every candidate. Modulate the production polynomial bump by exp(i*nu*t),
differentiate it analytically, and retain production integration-by-parts
semantics for all library terms. Compare original-size and full-record bumps.
Complex magnitudes pool sine/cosine pairs without dependence on phase origin.
"""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import window_profiles as W
import numpy as np
from scipy.fft import dct


REPRESENTATIVES = ['reference', 'archive_0', 'archive_1', 'archive_2', 'archive_6']


def select_bandwidth(t, observed):
    """GCV = N*RSS/(N-df)^2, df=k+1; no equation enters selection."""
    n = len(observed)
    transformed = dct(observed, type=2, norm='ortho')
    rss = np.cumsum(transformed[::-1] ** 2)[::-1][1:]
    df = np.arange(1, n)
    gcv = n * rss / (n - df) ** 2
    k = int(np.argmin(gcv))
    spacing = np.pi / (n * float(t[1] - t[0]))
    return np.arange(k + 1) * spacing, dict(
        method='DCT-II projection GCV on raw observed u', selected_max_mode=k,
        selected_df=k+1, frequency_spacing=spacing, maximum_frequency=k*spacing,
        gcv=gcv.tolist(), dct_power=(transformed ** 2).tolist())


def kernels(m, h, q, frequencies, max_order):
    """Product-rule kernels, with exact discrete polynomial IBP identities.

The zero-frequency kernels reproduce production. For each frequency, higher
derivative weights receive the minimum-norm moment correction so integration
by parts agrees on Legendre polynomials through degree 2q. The zeroth kernel
is the existing quadrature times the complex modulation. Never divide by the
oscillating test's near-zero signed mass; keep the bump's original scale.
"""
    base = W.WF.window_kernels((m,), (h,), (q,), (max_order,))[0]
    s = np.arange(-m, m+1) / m
    length = m*h
    phase = np.exp(1j * frequencies[:, None] * length * s[None, :])
    result = [phase * base[0]]
    degree = min(2*m, 2*q)
    vander = np.polynomial.legendre.legvander(s, degree)
    right_inverse = np.linalg.solve(vander.T @ vander, vander.T)
    maximum_moment_error = 0.0
    for order in range(1, max_order+1):
        raw = phase * sum(math.comb(order, j) * (1j*frequencies[:, None]) ** (order-j)
                          * base[j] for j in range(order+1))
        derivatives = np.column_stack([
            np.polynomial.legendre.legval(s, np.polynomial.legendre.legder(
                np.eye(degree+1)[j], m=order)) / length ** order
            for j in range(degree+1)])
        wanted = (-1) ** order * (result[0] @ derivatives)
        corrected = raw + (wanted - raw @ vander) @ right_inverse
        maximum_moment_error = max(maximum_moment_error, float(np.max(np.abs(corrected @ vander-wanted))))
        result.append(corrected)
    zero_error = max(float(np.max(np.abs(result[g][0]-base[g]))) for g in range(max_order+1))
    assert zero_error < 1e-10
    assert maximum_moment_error < 1e-9
    return result, dict(zero_frequency_kernel_max_error=zero_error,
                        polynomial_IBP_max_error=maximum_moment_error)


class Bank:
    def __init__(self, t, case, terms, frequencies, support, reach):
        self.terms = terms
        self.monos = [term.monomials(psi_cap=case['base_plan']['q']) for term in terms]
        self.max_order = max(key[2][0] for mono in self.monos for key in mono)
        n, h = len(t), float(t[1]-t[0])
        q = case['base_plan']['q'][0]
        if support == 'local':
            self.m = case['base_plan']['ms'][0]
            centres = case['base_plan']['cut'][0]+self.m+np.arange(case['base_plan']['shape'][0])*case['base_plan']['ss'][0]
            self.centres = centres[(centres-self.m-reach >= 1) & (centres+self.m+reach <= n-2)]
        else:
            first, last = 1+reach, n-2-reach
            self.m = (last-first)//2
            self.centres = np.array([first+self.m])
        self.kernels, self.validation = kernels(self.m, h, q, frequencies, self.max_order)
        self.offsets = np.arange(-self.m, self.m+1)
        self.mass = np.tile(self.kernels[0].sum(axis=1)[None, :], (len(self.centres), 1))
        self.frequencies = frequencies
        self.metadata = dict(support=support, half_width=self.m, width_time=2*self.m*h,
                             centres=self.centres.tolist(), test_count=len(frequencies)*len(self.centres),
                             real_test_count=(2*len(frequencies)-1)*len(self.centres), **self.validation)

    def matrix(self, fields):
        cache, integrals, columns = {}, {}, []
        positions = self.centres[:, None]+self.offsets[None, :]
        for term, monos in zip(self.terms, self.monos):
            column = np.zeros((len(self.centres), len(self.frequencies)), dtype=complex)
            for (data, coords, orders), multiplier in monos.items():
                key = (data, coords, term.opaque)
                if key not in cache:
                    cache[key] = fields.point(data, coords, term.opaque)[positions]
                ikey = (key, orders)
                if ikey not in integrals:
                    integrals[ikey] = cache[key] @ self.kernels[orders[0]].T
                column += multiplier * integrals[ikey]
            columns.append(column)
        matrix = np.stack(columns, axis=-1)
        assert np.isfinite(matrix).all()
        return matrix


def score(item, a, b, bank, index, metric, crossed=False, dc_only=False):
    coef = np.asarray(item['coefficients'])
    yi, xi = index[item['target']], [index[s] for s in item['terms']]
    groups = np.asarray(item['derivative_group'], dtype=bool)
    pairs = ((a, b), (b, a)) if crossed else ((a, a),) if b is None else ((a, a), (b, b))
    r_mass = np.zeros(len(bank.frequencies))
    y_mass, net_mass = r_mass.copy(), r_mass.copy()
    for target, features in pairs:
        y, x = target[:, :, yi], features[:, :, xi]
        residual = y - x @ coef - item['intercept'] * bank.mass
        net = y - x[:, :, groups] @ coef[groups]
        r_mass += np.abs(residual).sum(axis=0)
        y_mass += np.abs(y).sum(axis=0)
        net_mass += np.abs(net).sum(axis=0)
    # Include both signs of each nonzero frequency; the negative is conjugate.
    weight = np.full(len(r_mass), 2.0)
    weight[0] = 1.0
    numerator = np.cumsum(weight*r_mass)
    target_denominator = np.cumsum(weight*y_mass)
    denominator = np.minimum(target_denominator, np.cumsum(weight*net_mass)) if metric == 'wape_net_cap' else target_denominator
    if np.any(denominator <= 0):
        raise ValueError('Zero spectral discrepancy denominator')
    profile = numerator / denominator
    pick = 0 if dc_only else -1
    return dict(discrepancy=float(profile[pick]), wape=float((numerator/target_denominator)[pick]),
                prefix_discrepancy=profile.tolist(), frequency_residual_mass=r_mass.tolist(),
                frequency_target_mass=y_mass.tolist())


def evaluate(case, fits, a, b, bank, index, crossed=False, dc_only=False):
    out = dict(models=[])
    fitted = {m['id']:m for m in fits}
    for template in case['models']:
        saved = fitted[template['id']]
        item = dict(template, coefficients=saved['coefficients'], intercept=saved['intercept'])
        result = score(item, a, b, bank, index, case['metric'], crossed, dc_only)
        out['models'].append(dict(id=item['id'], coefficients=item['coefficients'], intercept=item['intercept'],
                                  objectives=[result['discrepancy'],saved['objectives'][1]], **result))
    reference = out['models'][0]['objectives']
    out['reference_objectives'] = reference
    out['dominators'] = [m['id'] for m in out['models'][1:] if W.dominates(m['objectives'],reference)]
    out['representative_dominators'] = [s for s in out['dominators'] if s in REPRESENTATIVES[1:]]
    return out


def field_error_diagnostic(case, fits, a, b, clean, bank, index):
    """Oracle field swap with the SAME noisy-fit coefficients, for explanation."""
    saved = {m['id']:m for m in fits}
    weights = np.full(len(bank.frequencies),2.)
    weights[0] = 1.
    mass = lambda x: float(np.sum(weights[None,:]*np.abs(x)))
    out=[]
    for template in case['models']:
        if template['id'] not in REPRESENTATIVES:
            continue
        fit=saved[template['id']]
        coef=np.asarray(fit['coefficients'])
        yi,xi=index[template['target']],[index[s] for s in template['terms']]
        cy,cx=clean[:,:,yi],clean[:,:,xi]@coef+fit['intercept']*bank.mass
        cr=cy-cx
        sums=np.zeros(3)
        for left,right in ((a,b),(b,a)):
            dy=left[:,:,yi]-cy
            dx=right[:,:,xi]@coef+fit['intercept']*bank.mass-cx
            sums += [mass(dy),mass(dx),mass(dy-dx)]
        denominator=2*mass(cy)
        out.append(dict(id=template['id'],clean_field_discrepancy_same_coefficients=mass(cr)/mass(cy),
                        target_field_error=sums[0]/denominator,rhs_field_error=sums[1]/denominator,
                        residual_field_error=sums[2]/denominator))
    return out


def numerical_checks():
    """Check IBP independently against analytic derivatives of a smooth signal."""
    t = np.linspace(-3, 3, 1501)
    frequencies = np.array([0., .7, 2.1, 4.4])
    ks, diagnostics = kernels(750, float(t[1]-t[0]), 3, frequencies, 2)
    u = np.sin(1.7*t)+.2*np.cos(.4*t)
    derivatives = [u, 1.7*np.cos(1.7*t)-.08*np.sin(.4*t),
                   -1.7**2*np.sin(1.7*t)-.032*np.cos(.4*t)]
    errors = [float(np.max(np.abs((-1)**g*ks[g]@u-ks[0]@derivatives[g]))) for g in (1,2)]
    assert max(errors) < 2e-7, errors
    # Constant intercept is transformed by the test, including its signed mass.
    mass = ks[0].sum(axis=1)
    np.testing.assert_allclose(ks[0]@np.full(len(t),2.3)-2.3*mass,0,atol=1e-13)
    return dict(**diagnostics, analytic_sinusoid_IBP_absolute_errors=errors)


def run_case(case, corrected, crossed_source, root):
    print(case['run'], flush=True)
    labels = sorted({s for m in case['models'] for s in [m['target'],*m['terms']]})
    terms = [W.parse_term(s) for s in labels]
    index = {s:i for i,s in enumerate(labels)}
    reach = W.WF.FD_REACH * max(W.WF.leftover_orders(term.monomials(psi_cap=case['base_plan']['q']),1)[0] for term in terms)
    result = dict(run=case['run'],system=case['system'],metric=case['metric'],trials=[])
    for trial, correction, cross_old in zip(case['trials'],corrected['trials'],crossed_source['trials']):
        seed = trial['seed']
        path = root/'signal_results'/f"{case['run']}_seed{seed}_signal.npz"
        with np.load(path) as z:
            t, noisy, smoothed, clean = [z[k].copy() for k in ('t','noisy','smoothed','clean')]
        assert W.digest(noisy) == trial['noise_digest']
        split_path = root/'cross_results'/f"{case['system']}_seed{seed}_splits.npz"
        with np.load(split_path) as z:
            splits = {k:z[k].copy() for k in ('a','b','clean_a','clean_b')}
        frequencies, selection = select_bandwidth(t,noisy)
        row = dict(seed=seed,selection=selection,source_hashes={str(p):W.source_hash(p) for p in (path,split_path)},
                   frequencies=trial['frequencies'],banks={},stages={})
        banks = {support:Bank(t,case,terms,frequencies,support,reach) for support in ('local','global')}
        row['banks'] = {name:b.metadata for name,b in banks.items()}
        fields = {name:W.fields(t,values,trial['frequencies'],terms) for name,values in
                  dict(raw=noisy,smoothed=smoothed,clean=clean,**splits).items()}
        matrices = {support:{name:bank.matrix(f) for name,f in fields.items()} for support,bank in banks.items()}
        # At nu=0, local tests must equal the existing production matrix.
        baseline_error = 0.
        for name in fields:
            old, centres = W.matrix(terms,fields[name],case['base_plan'])
            positions = np.searchsorted(centres,banks['local'].centres)
            diff = np.max(np.abs(old[positions]-matrices['local'][name][:,0,:]))
            baseline_error=max(baseline_error,float(diff))
            np.testing.assert_allclose(old[positions],matrices['local'][name][:,0,:].real,rtol=1e-9,atol=1e-9)
        row['zero_frequency_matrix_max_error'] = baseline_error
        for stage in ('raw','smoothed','clean'):
            fits = (correction['stages'][stage]['methods']['clsm']['full_baseline']['models'] if stage != 'clean'
                    else cross_old['clean_control']['full_baseline']['models'])
            row['stages'][stage] = {}
            for support,bank in banks.items():
                mm = matrices[support]
                aa,bb = (mm['clean_a'],mm['clean_b']) if stage == 'clean' else (mm['a'],mm['b'])
                for variant,dc in (('dc',True),('spectral',False)):
                    arm = f'{support}_{variant}'
                    row['stages'][stage][arm] = dict(
                        full=evaluate(case,fits,mm[stage],None,bank,index,dc_only=dc),
                        within=evaluate(case,fits,aa,bb,bank,index,dc_only=dc),
                        crossed=evaluate(case,fits,aa,bb,bank,index,crossed=True,dc_only=dc))
                    if stage != 'clean':
                        row['stages'][stage][arm]['clean_fields_only'] = evaluate(
                            case,fits,mm['clean'],None,bank,index,dc_only=dc)
                if support=='global' and stage!='clean':
                    row['stages'][stage]['global_spectral']['field_error_diagnostic'] = field_error_diagnostic(
                        case,fits,aa,bb,mm['clean'],bank,index)
                    # A common phase rotation must not change the complex score.
                    phase=np.exp(1j*np.random.default_rng(935).normal(size=bank.mass.shape))
                    rotated=SimpleNamespace(frequencies=bank.frequencies,mass=bank.mass*phase)
                    template=case['models'][0]
                    item=dict(template,coefficients=fits[0]['coefficients'],intercept=fits[0]['intercept'])
                    original=score(item,aa,bb,bank,index,case['metric'],True)['discrepancy']
                    changed=score(item,aa*phase[:,:,None],bb*phase[:,:,None],rotated,index,case['metric'],True)['discrepancy']
                    np.testing.assert_allclose(original,changed,rtol=1e-12,atol=1e-14)
                    equal=score(item,aa,aa,bank,index,case['metric'],True)['discrepancy']
                    single=score(item,aa,None,bank,index,case['metric'])['discrepancy']
                    np.testing.assert_allclose(equal,single,rtol=1e-12,atol=1e-14)
            expected = cross_old['clean_control']['crossed'] if stage == 'clean' else correction['stages'][stage]['methods']['clsm']['crossed']
            actual = row['stages'][stage]['local_dc']['crossed']
            np.testing.assert_allclose([m['objectives'] for m in actual['models']],
                                       [m['objectives'] for m in expected['models']],rtol=1e-8,atol=1e-9)
            assert actual['dominators'] == expected['dominators']
            counts = {arm:len(v['crossed']['dominators']) for arm,v in row['stages'][stage].items()}
            print(f"  seed={seed} {stage} k={selection['selected_max_mode']} crossed={counts}",flush=True)
        result['trials'].append(row)
    return result


def write(report, output):
    (output/'spectral_weak_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['Oscillating weak tests: frozen coefficients and dyad','',report['protocol'],'',
             'Primary arm: global_spectral, data-only GCV bandwidth, crossed discrepancy.',
             'All other arms are controls; bandwidth profiles are diagnostics, never selected by rank.','']
    for case in report['cases']:
        lines.append(case['run'])
        for trial in case['trials']:
            lines.append(f"seed {trial['seed']}: max mode={trial['selection']['selected_max_mode']}, omega={trial['selection']['maximum_frequency']:.6g}")
            for stage,arms in trial['stages'].items():
                for arm,variants in arms.items():
                    for variant in ('full','crossed'):
                        v=variants[variant]
                        lines.append(f"  {stage} {arm} {variant}: reference={v['reference_objectives']}; dominators={v['dominators']}")
                if case['system']=='kepler':
                    v=arms['global_spectral']['crossed']
                    lines.append('  Kepler representatives, primary arm:')
                    lines.extend(f"    {m['id']}: {m['objectives']}" for m in v['models'] if m['id'] in REPRESENTATIVES)
    lines += ['', 'Limitations:', *report['limitations']]
    (output/'spectral_weak_probe.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case=next(c for c in report['cases'] if c['system']=='kepler')
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    colors=['#b12a31','#1976b9','#d78c21','#588d42','#7954aa']
    labels=['Reference','A','B','C','D']
    for trial,ax in zip(case['trials'],axes):
        freq=np.arange(trial['selection']['selected_max_mode']+1)*trial['selection']['frequency_spacing']
        records=trial['stages']['raw']['global_spectral']['crossed']['models']
        for ident,label,color in zip(REPRESENTATIVES,labels,colors):
            row=next(m for m in records if m['id']==ident)
            ax.plot(freq[1:],row['prefix_discrepancy'][1:],label=label,color=color)
            ax.scatter(freq[-1],row['discrepancy'],color=color,s=22)
        ax.set(xlabel='Maximum angular frequency included',ylabel='Crossed relative weak residual',
               yscale='log',title=f"Seed {trial['seed']}: GCV selects {freq[-1]:.2f}")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Oscillating full-record weak tests: Kepler\nCoefficients and dyad frozen; bandwidth selected from raw observations')
    fig.savefig(output/'spectral_weak_profiles.png',dpi=180)
    fig.savefig(output/'spectral_weak_profiles.svg')
    plt.close(fig)


def main():
    if os.name=='nt':
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--all-cases',action='store_true')
    args=parser.parse_args()
    root=args.root
    output=root/('spectral_results_all' if args.all_cases else 'spectral_results')
    output.mkdir(parents=True,exist_ok=True)
    paths=[root/'signal_results/signal_refits.json',root/'estimator_results/corrected_estimator.json',
           root/'cross_results/cross_reconstruction.json']
    sources=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    report=dict(protocol=__doc__,sources={str(p):W.source_hash(p) for p in [*paths,Path(__file__),Path(W.__file__),
                                                                         W.REPO/'epde/preprocessing/weak_form.py']},
                validation=numerical_checks(),cases=[],limitations=[
                    'Archived coefficients, frequencies and supports used the full record: not independent held-out validation.',
                    'Raw-u GCV is a field-prediction criterion; it need not be optimal for derivatives or model ranking.',
                    'Complex-modulus pooling differs from separate sine/cosine L1 rows; at zero frequency it equals existing WAPE.',
                    'No coefficients or dyad were refitted on the new test bank; clean control fits are saved clean fits.',
                    'The same noisy-data bandwidth and bank are used in clean controls; clean data never select the bank.',
                    'Crossed reconstructions assume iid raw noise; full-record tests do not create independent observations.',
                    'No candidate equation integration, complexity penalty, support restriction, or production change.'])
    start=time.perf_counter()
    for case,corrected,crossed_source in zip(*(s['cases'] for s in sources)):
        if case['system']!='kepler' and not args.all_cases:
            continue
        report['cases'].append(run_case(case,corrected,crossed_source,root))
        write(report,output)
    report['runtime_seconds']=time.perf_counter()-start
    write(report,output)
    plot(report,output)
    print(f"Completed in {report['runtime_seconds']:.1f}s: {output}",flush=True)


if __name__=='__main__':
    main()
