"""Paired blind-frequency KdV window and two-trajectory replay on the fixed physical support.

Every arm fits u_t = a u u_x + b u_xxx + c cos(w_t t) sin(w_x x). The
single-record arms fit all five parameters. The staged arms estimate a and b
on an independent noisy unforced record and hold both fixed on the forced record.

The original failing plan sets the time window, stride and smoothness. The
width arm pools two wider spatial half-widths. The stagger arm shifts the
windows independently by one grid point along each axis. The validation
residual is ||y-X beta|| / ||y|| on a common bank with other centers. A new
noise realization and the clean benchmark are each evaluated there using the
frozen training fit. The second-center refit uses the original noisy field.
All noise draws come from the same two underlying trajectories.
"""
import ctypes
from ctypes import wintypes
from itertools import product
import json
import os
from pathlib import Path
import sys


def limit_cpu():
    import psutil
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel.CreateJobObjectW(None, None)
        rate = (wintypes.DWORD * 2)(5, 200)  # CPU rate control and hard cap, in hundredths of one percent
        if not (job and kernel.SetInformationJobObject(job, 15, ctypes.byref(rate), ctypes.sizeof(rate))
                and kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
            raise ctypes.WinError(ctypes.get_last_error())
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[-1]])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'


if __name__ == '__main__':
    limit_cpu()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'fitted_frequencies_2026-11-23'))
from frequency_study import blind, directory, live, np, read_archive
from epde.operators.common.frequency_fit import project_frequencies, residual_modes
from epde.preprocessing import weak_form as wf
from epde.structure.weak_rows import WeakRows

ZERO = (0, 0)
TERMS = [wf.WeakTerm([('u', (1, 0))], ZERO),
         wf.WeakTerm([('u', ZERO), ('u', (0, 1))], ZERO),
         wf.WeakTerm([('u', (0, 3))], ZERO)]
STAGGER = tuple(product(range(2), repeat=2))
DESIGNS = {'native': ((16,), (ZERO,)), 'stagger': ((16,), STAGGER),
           'wide': ((24, 32), (ZERO,)), 'wide_stagger': ((24, 32), STAGGER)}


def intrinsic(fields, ms, ss, q):
    cols, shape = wf.weak_matrix([term.monomials(psi_cap=q) for term in TERMS], [()] * len(TERMS),
                                 fields, ZERO, ms, ss, q)
    return cols.reshape(*shape, len(TERMS))


class Measurement:
    """Integrate every term on the same centers, using native weak-form quadrature."""
    def __init__(self, fields, spatial_widths, plan):
        self.axes = fields.axes
        self.steps = np.asarray(fields.steps)
        self.stride = np.asarray(plan['ss'])
        self.values = {}
        self.kernels = {}
        assert list(plan['cut']) == [0, 0]
        for width in spatial_widths:
            ms = (plan['ms'][0], width)
            self.values[width] = intrinsic(fields, ms, (1, 1), plan['q'])
            self.kernels[width] = [v[0] for v in wf.window_kernels(ms, self.steps, plan['q'], ZERO)]

    def bank(self, widths, offsets):
        return Bank(self, widths, offsets)


class Bank:
    """Stack geometries with equal total squared weight and one common coefficient vector."""
    def __init__(self, measurement, widths, offsets):
        self.m = measurement
        self.widths = widths
        self.blocks = []
        arrays = []
        for width, offset in product(widths, offsets):
            take = tuple(slice(o, None, s) for o, s in zip(offset, measurement.stride))
            block = measurement.values[width][take]
            scale = 1 / np.sqrt(block.shape[0] * block.shape[1])
            self.blocks.append((width, take, block.shape[:2], scale))
            arrays.append(block.reshape(-1, 3) * scale)
        self.columns = np.vstack(arrays)

    def tone(self, omega):
        t = np.cos(self.m.axes[0] * omega[0])
        x = np.sin(self.m.axes[1] * omega[1])
        t = np.correlate(t, self.m.kernels[self.widths[0]][0], mode='valid')
        spatial = {w: np.correlate(x, self.m.kernels[w][1], mode='valid') for w in self.widths}
        return np.concatenate([(t[st, None] * spatial[w][sx] * s).ravel()
                               for w, (st, sx), _, s in self.blocks])

    def fit(self, peaks, fixed=None):
        y, x = self.columns[:, 0], self.columns[:, 1:]
        if fixed is None:
            residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            cols = [(c, None) for c in x.T]
        else:
            y = y - x @ fixed
            residual, cols = y, []
        upper = np.pi / self.m.steps
        spans = np.array([np.ptp(a) for a in self.m.axes])
        spacing = self.m.steps * self.m.stride
        starts = []
        pos = 0
        for _, _, shape, _ in self.blocks:
            n = int(np.prod(shape))
            starts.append([residual_modes(residual[pos:pos+n], shape, d, spacing[d], spans[d], upper[d])[0]
                           for d in range(2)])
            pos += n
        samples = [(y, np.ones_like(y), [*cols, (self.tone, slice(2))])]
        omega = project_frequencies(samples, (0, 1), upper, np.pi/(2*spans),
                                    np.median(starts, axis=0), False, lambda: None,
                                    [2*np.pi/spacing], peaks)
        if omega is None:
            raise ValueError('No finite full-rank frequency fit')
        tone = self.tone(omega)
        if fixed is None:
            beta = np.linalg.lstsq(np.column_stack((x, tone)), y, rcond=None)[0]
        else:
            beta = np.r_[fixed, np.linalg.lstsq(tone[:, None], y, rcond=None)[0]]
        return dict(coefficients=beta.tolist(), frequencies=omega.tolist())

    def discrepancy(self, fit):
        y = self.columns[:, 0]
        a, b, c = fit['coefficients']
        r = y - self.columns[:, 1:] @ [a, b] - c * self.tone(fit['frequencies'])
        return float(np.linalg.norm(r) / np.linalg.norm(y))


def reference_scores(observed, clean, held, held_clean):
    """Use benchmark truth only after fitting to measure the noise at physical coefficient scale."""
    physical = dict(coefficients=[-6., -1., 1.], frequencies=[1., 1.])
    scores = dict(clean_residual=held_clean.discrepancy(physical), noisy_residual=held.discrepancy(physical),
                  dispersion_noise_over_forcing={})
    for name, (widths, offsets) in DESIGNS.items():
        noisy_bank = observed.bank(widths, offsets)
        clean_bank = clean.bank(widths, offsets)
        noise = noisy_bank.columns[:, 2] - clean_bank.columns[:, 2]
        source = clean_bank.tone(physical['frequencies'])
        scores['dispersion_noise_over_forcing'][name] = float(np.linalg.norm(noise) / np.linalg.norm(source))
    return scores


def run():
    original = json.loads((ROOT / 'results.json').read_text())[0]
    plan = original['plan']
    assert original['seed'] == 923701 and plan['ms'] == [16, 16] and plan['ss'] == [4, 4]
    unforced_plan = read_archive(directory('kdv', 923701, 5))['plan']
    assert unforced_plan['cut'] == [0, 0]
    forced = blind(live.load_data('kdv_cossin'))
    unforced = live.load_data('kdv')
    widths = (16, 24, 32)

    def fields(problem, seed, noise):
        return wf.Fields(problem.noisy(noise, seed), live.R.axes_of(problem))

    clean = Measurement(fields(forced, 0, 0), widths, plan)
    def shifted(offsets):
        return tuple((i+2, j+2) for i, j in offsets)
    held_offsets = shifted(STAGGER)
    assert set(held_offsets).isdisjoint(STAGGER)
    held_clean = clean.bank(widths, held_offsets)
    output = dict(forced_plan=plan, unforced_plan=unforced_plan, designs=DESIGNS,
                  common_validation_widths=widths, common_validation_offsets=held_offsets, cases=[])
    for seed in (0, 923701, *range(925701, 925713)):
        noise = 0 if seed == 0 else 5
        raw = fields(forced, seed, noise)
        observed = Measurement(raw, widths, plan)
        held = Measurement(fields(forced, seed+20000, noise), widths, plan).bank(widths, held_offsets)
        unforced_field = fields(unforced, seed+10000, noise)
        ucols = intrinsic(unforced_field, unforced_plan['ms'], unforced_plan['ss'], unforced_plan['q']).reshape(-1, 3)
        fixed = np.linalg.lstsq(ucols[:, 1:], ucols[:, 0], rcond=None)[0]
        rows = WeakRows({0: raw}, {}, [0])
        peaks = {d: rows.observed_frequencies(d) for d in range(2)}
        case = dict(seed=seed, unforced_seed=seed+10000, validation_seed=seed+20000, noise_percent=noise,
                    unforced_coefficients=fixed.tolist(), arms={})
        for name, (ws, offsets) in DESIGNS.items():
            train = observed.bank(ws, offsets)
            comparison = clean.bank(ws, offsets).columns[:, 2]
            err = np.linalg.norm(train.columns[:, 2]-comparison) / np.linalg.norm(comparison)
            center_refit = observed.bank(ws, shifted(offsets)).fit(peaks)
            for mode, coefficients in [('free', None), ('staged', fixed)]:
                fit = train.fit(peaks, fixed=coefficients)
                fit.update(clean_held_residual=held_clean.discrepancy(fit), noisy_held_residual=held.discrepancy(fit),
                           dispersion_column_error=float(err))
                if mode == 'free':
                    fit['shifted_center_refit'] = center_refit
                case['arms'][name+'_'+mode] = fit
        if seed == 923701:
            baseline = case['arms']['native_free']
            old = original['fits']['raw']
            np.testing.assert_allclose(baseline['frequencies'], [old['omega_t'], old['omega_x']], rtol=2e-4)
            np.testing.assert_allclose(baseline['dispersion_column_error'], old['dispersion_column_error'])
            np.testing.assert_allclose(baseline['coefficients'][1], old['dispersion_coefficient'], atol=1e-5)
        case['physical_reference'] = reference_scores(observed, clean, held, held_clean)
        output['cases'].append(case)
        print(f'Completed {seed}: {len(case["arms"])} fits and four second-center refits', flush=True)
    path = ROOT / 'window_results.json'
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    pending.replace(path)


if __name__ == '__main__':
    run()
