"""Isolate dispersion-column error on the blind forced KdV support at 5% noise."""
import contextlib
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'fitted_frequencies_2026-11-23'))
from check_support import fit_terms
from frequency_study import blind, directory, live, np, read_archive
from epde.operators.common.frequency_fit import fit_support, fitted_factors
from epde.preprocessing.weak_form import Fields, window_kernels
from epde.structure.weak_rows import WeakRows


def smooth(field, axes, plan):
    """Compare selection for field error and for error on the weak dispersion measurements."""
    # The four-point difference has noise standard deviation 2*sigma for iid observations.
    from scipy.special import ndtri
    diff = np.diff(np.diff(field, axis=0), axis=1)
    sigma = np.median(abs(diff - np.median(diff))) / (2 * ndtri(.75))
    bases = []
    for axis in axes:
        coord = 2 * (axis-axis[0]) / np.ptp(axis) - 1
        bases.append(np.linalg.qr(np.polynomial.legendre.legvander(coord, len(coord)-1))[0])
    t, x = bases
    coeff = t.T @ field @ x
    energy = np.cumsum(np.cumsum(coeff**2, axis=0), axis=1)
    count = np.outer(np.arange(1, len(t)+1), np.arange(1, len(x)+1))
    bic = count * np.log(field.size) * sigma**2 - energy

    # L integrates against the third spatial derivative of this window. For a
    # fixed orthogonal projection P, SURE for L P u is
    # ||L(I-P)y||² + 2 sigma² tr(L P L.T), up to a constant shared by all P.
    kernels = window_kernels(plan['ms'], [a[1]-a[0] for a in axes], plan['q'], (0, 3))
    measured = []
    for a, order in enumerate((0, 3)):
        cut = plan['cut'][a]
        basis = bases[a][cut:len(axes[a])-cut]
        kernel = kernels[a][order]
        windows = np.lib.stride_tricks.sliding_window_view(basis.T, len(kernel), axis=1)
        measured.append(windows[:, ::plan['ss'][a]] @ kernel)
    bt, bx = measured
    lt = np.cumsum(np.sum(bt**2, axis=1))
    lx = np.cumsum(np.sum(bx**2, axis=1))
    penalty = 2 * sigma**2 * np.outer(lt, lx)
    observed = bt.T @ coeff @ bx
    partial = np.zeros((len(x), *observed.shape))
    risk = np.empty_like(coeff)
    for i in range(len(t)):
        by_space = np.cumsum(coeff[i, :, None] * bx, axis=0)
        partial += bt[i, None, :, None] * by_space[:, None, :]
        risk[i] = np.sum((observed-partial)**2, axis=(1, 2)) + penalty[i]

    def reconstruct(score):
        i, j = (int(d) for d in np.unravel_index(np.argmin(score), score.shape))
        return t[:, :i+1] @ coeff[:i+1, :j+1] @ x[:, :j+1].T, [i, j]
    return reconstruct(bic), reconstruct(risk)


def measure(problem, seed, plan, raw):
    axes = live.R.axes_of(problem)
    target, features, _ = live.R.parse(problem.truth[0])
    fitted, rows = fit_terms(problem, raw, plan, target, features)
    assert fitted is not None
    dispersion, = [t for t in fitted.structure if any(f.label == 'd^3u/dx1^3' for f in t.structure)]
    (filtered, degrees), (derivative_filtered, derivative_degrees) = smooth(raw['u'], axes, plan)

    def reference_rows(data):
        return WeakRows({0: Fields(data, axes)}, rows.plans, rows.keys)

    smooth_rows = reference_rows({'u': filtered})
    derivative_rows = reference_rows({'u': derivative_filtered})
    clean_rows = reference_rows(problem.data)
    clean_column = clean_rows.columns(dispersion, 'u')[0]
    keep = [j for j in range(len(fitted.structure)) if j != fitted.target_idx]
    fits = {}
    for name, replacement in [('raw', None), ('smooth_dispersion', smooth_rows),
                              ('smooth_all', smooth_rows), ('derivative_sure', derivative_rows),
                              ('clean_dispersion_control', clean_rows)]:
        work = reference_rows(raw)
        work.fit_frequencies = True
        original = work.columns

        def columns(term, slot):
            use = replacement is not None and (term.factors_labels == dispersion.factors_labels or
                                                (name == 'smooth_all' and not fitted_factors(term)))
            return replacement.columns(term, slot) if use else original(term, slot)

        with patch.object(work, 'columns', columns):
            eq = fitted if name == 'raw' else fit_support(fitted, keep, False, work)
            assert eq is not None
            design = np.column_stack([columns(eq.structure[j], 'u')[0] for j in keep])
            y = columns(eq.target, 'u')[0]
            coeff = np.linalg.lstsq(design, y, rcond=None)[0]
            idx, = [i for i, j in enumerate(keep) if eq.structure[j].factors_labels == dispersion.factors_labels]
            tones = {f.label: float(f.param('freq')) for term in eq.structure for f in fitted_factors(term)}
            err = np.linalg.norm(design[:, idx]-clean_column) / np.linalg.norm(clean_column)
            fits[name] = dict(dispersion_column_error=float(err), dispersion_coefficient=float(coeff[idx]),
                              omega_t=tones['cos'], omega_x=tones['sin'])
            periods = [2*np.pi / ((a[1]-a[0])*s) for a, s in zip(axes, plan['ss'])]
            if any(tones[tone] > p/2 for tone, p in zip(('cos', 'sin'), periods)):
                folded = deepcopy(eq)
                for term in folded.structure:
                    for f in fitted_factors(term):
                        p = periods[int(f.param('dim'))]
                        f.set_param(float(abs((f.param('freq') + p/2) % p - p/2)), name='freq')
                    term.resetSavedState()
                alt = np.column_stack([columns(folded.structure[j], 'u')[0] for j in keep])
                low_coef = np.linalg.lstsq(alt, y, rcond=None)[0]
                fits[name]['alias_check'] = dict(
                    lower_frequencies={f.label: float(f.param('freq')) for t in folded.structure
                                       for f in fitted_factors(t)},
                    residual_change_over_target=float((np.linalg.norm(y-alt@low_coef) -
                                                       np.linalg.norm(y-design@coeff))/np.linalg.norm(y)))
    return dict(seed=seed, plan=plan, smoothing_degrees=degrees, derivative_degrees=derivative_degrees,
                fits=fits)


def main():
    problem = blind(live.load_data('kdv_cossin'))
    cases = []
    for seed in (923701, 923702, 923703, *range(924701, 924713)):
        raw = problem.noisy(5, seed)
        if seed < 924701:
            archive = read_archive(directory('kdv_cossin', seed, 5))
            assert archive['train_hashes'] == {k: live.R.digest(v) for k, v in raw.items()}
            plan = archive['plan']
        else:
            # Repeat the first failing design on fresh independent measurement noise.
            plan = cases[0]['plan']
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            cases.append(measure(problem, seed, plan, raw))
        print(seed, cases[-1]['derivative_degrees'], cases[-1]['fits']['derivative_sure'], flush=True)
    (ROOT / 'results.json').write_text(json.dumps(cases, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
