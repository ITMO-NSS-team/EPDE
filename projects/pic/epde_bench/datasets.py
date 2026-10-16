"""Every data set in ``projects/pic/data`` behind one interface.

``load(name, **kwargs)`` returns a :class:`Problem`; ``REGISTRY`` describes
them all (``python bench.py list``). Loaders read files relative to this
package, so they work from any working directory and on any OS.

Ground truths are taken, in order of preference, from
``projects/pinn/gate.py`` (curated per system, with the data windows it
justifies), the group's former benchmark configs (``projects/thesis/configs``
on ``main``) and the docstrings of the per-system scripts. Where a choice
differs from an older script it is stated in the loader's ``notes``.

Suites:

* ``core``     -- synthetic, known truth: the 14 systems of the group's
  benchmark plus forced Duffing. This is what the benchmark tables report.
* ``extended`` -- synthetic, known truth, but slower or less standard
  (alternative KdV records, JHTDB turbulence slice).
* ``real``     -- measured data. Truth is known only for the single
  pendulum; the others are reported without structural metrics.
* ``other``    -- loadable, but not part of any benchmark (no truth, data
  incomplete, or too large for a routine run).
"""

import gzip
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from . import tokens
from .paths import data_path
from .problem import Problem, mesh


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    kind: str
    source: str
    suite: str
    files: str
    loader: Callable[..., Problem]


REGISTRY: Dict[str, DatasetSpec] = {}


def dataset(name, title, kind, source, suite, files):
    def register(loader):
        REGISTRY[name] = DatasetSpec(name, title, kind, source, suite, files, loader)
        return loader
    return register


def load(name: str, **kwargs) -> Problem:
    if name not in REGISTRY:
        raise KeyError(f'Unknown data set {name!r}; known: {", ".join(sorted(REGISTRY))}')
    return REGISTRY[name].loader(**kwargs)


def names(suite: str = None) -> List[str]:
    """Registered names, optionally of one suite (``'all'`` = every suite)."""
    if suite in (None, 'all'):
        return list(REGISTRY)
    suites = suite.split('+')
    return [n for n, spec in REGISTRY.items() if spec.suite in suites]


def _problem(name, grids, data, axes, **kwargs) -> Problem:
    spec = REGISTRY[name]
    return Problem(name=name, title=spec.title, kind=spec.kind, source=spec.source,
                   grids=grids, data=data, axis_names=axes, **kwargs)


def _loadmat(*parts, **kwargs):
    from scipy.io import loadmat
    return loadmat(data_path(*parts), **kwargs)


# =========================================================================
# ODEs
# =========================================================================

@dataset('ode', 'Forced oscillator with time-dependent damping', 'ode', 'synthetic', 'core',
         'ode/ode_data.npy')
def load_ode():
    t = np.arange(320) * 0.05
    return _problem(
        'ode', mesh(t), {'u': np.load(data_path('ode', 'ode_data.npy'))}, ('t',),
        truth=['-4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'
               ' + 1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}'],
        notes="u'' + sin(2t) u' + 4u = 1.5t on t in [0, 16), dt = 0.05.")


@dataset('vdp', 'Van der Pol oscillator (mu = 0.2)', 'ode', 'synthetic', 'core',
         'vdp/vdp_data.npy')
def load_vdp():
    t = np.arange(320) * 0.05
    return _problem(
        'vdp', mesh(t), {'u': np.load(data_path('vdp', 'vdp_data.npy'))}, ('t',),
        truth=['-0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0}'
               ' + -1.0 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}'],
        notes="u'' = 0.2 (1 - u^2) u' - u on t in [0, 16), dt = 0.05.")


@dataset('duffing', 'Forced Duffing oscillator', 'ode', 'synthetic', 'core',
         'duffing/duffing.npz')
def load_duffing():
    d = np.load(data_path('duffing', 'duffing.npz'))
    p = {k: float(d[k]) for k in ('delta', 'alpha', 'beta', 'gamma', 'omega')}
    truth = (f"{-p['delta']} * du/dx0{{power: 1.0}} + {-p['alpha']} * u{{power: 1.0}}"
             f" + {-p['beta']} * u{{power: 3.0}}"
             f" + {p['gamma']} * cos{{power: 1.0, freq: {p['omega']}, dim: 0.0}}"
             f" = d^2u/dx0^2{{power: 1.0}}")
    return _problem(
        'duffing', mesh(d['t']), {'u': d['x'].astype(np.float64)}, ('t',),
        truth=[truth], meta=p,
        notes="u'' + delta u' + alpha u + beta u^3 = gamma cos(omega t); the parameters are stored "
              'in the data file.')


# =========================================================================
# Systems of ODEs
# =========================================================================

@dataset('lv', 'Lotka-Volterra predator-prey', 'ode_system', 'synthetic', 'core',
         'lv/t_20.npy, lv/data_20.npy')
def load_lv(n_points: int = None):
    t = np.load(data_path('lv', 't_20.npy')).astype(np.float64)
    x = np.load(data_path('lv', 'data_20.npy')).astype(np.float64)
    if n_points:
        t, x = t[:n_points], x[:n_points]
    return _problem(
        'lv', mesh(t), {'u': x[:, 0], 'v': x[:, 1]}, ('t',),
        truth=['20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} = du/dx0{power: 1.0}',
               '20.0 * u{power: 1.0} * v{power: 1.0} + -20.0 * v{power: 1.0} = dv/dx0{power: 1.0}'],
        truth_alternatives=[[
            '-1.0 * dv/dx0{power: 1.0} + 20.0 * u{power: 1.0} + -20.0 * v{power: 1.0} = du/dx0{power: 1.0}',
            '20.0 * du/dx0{power: 1.0} + -20.0 * dv/dx0{power: 1.0} + -1.0 * d^2u/dx0^2{power: 1.0}'
            ' = d^2v/dx0^2{power: 1.0}']],
        notes='alpha = beta = gamma = delta = 20, all 301 samples: with only the first half of the '
              'record the system is not identifiable.')


@dataset('lorenz', 'Lorenz-63 system', 'ode_system', 'synthetic', 'core',
         'lorenz/t.npy, lorenz/lorenz.npy')
def load_lorenz(i0: int = 20000, n: int = 1041, step: int = 5):
    t = np.load(data_path('lorenz', 't.npy'))
    x = np.load(data_path('lorenz', 'lorenz.npy'))
    idx = i0 + step * np.arange(n)
    return _problem(
        'lorenz', mesh(t[idx] - t[idx[0]]),
        {'u': x[idx, 0], 'v': x[idx, 1], 'w': x[idx, 2]}, ('t',),
        truth=['10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} = du/dx0{power: 1.0}',
               '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0}'
               ' = dv/dx0{power: 1.0}',
               '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0}'
               ' = dw/dx0{power: 1.0}'],
        meta={'i0': i0, 'n': n, 'step': step},
        notes='sigma = 10, rho = 28, beta = 8/3. Window t in [20.0, 25.2] of the stored run, every '
              '5th sample: on the attractor, not the initial transient.')


# =========================================================================
# PDEs, one space dimension
# =========================================================================

@dataset('ac', 'Allen-Cahn equation', 'pde_1d', 'synthetic', 'core', 'ac/ac_data.npy')
def load_ac():
    grids = mesh(np.linspace(0.0, 1.0, 51), np.linspace(-1.0, 0.984375, 128))
    return _problem(
        'ac', grids, {'u': np.load(data_path('ac', 'ac_data.npy'))}, ('t', 'x'),
        truth=['0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        notes='u_t = 1e-4 u_xx + 5u - 5u^3. The diffusion term is tiny, which makes it hard to '
              'separate from noise.')


@dataset('burgers', 'Viscous Burgers equation (PDE-FIND data)', 'pde_1d', 'synthetic', 'core',
         'burgers/burgers.mat')
def load_burgers():
    m = _loadmat('burgers', 'burgers.mat')
    grids = mesh(np.ravel(m['t']), np.ravel(m['x']))
    return _problem(
        'burgers', grids, {'u': np.transpose(np.real(m['usol']))}, ('t', 'x'),
        truth=['-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.1 * d^2u/dx1^2{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        notes='u_t = -u u_x + 0.1 u_xx, periodic in x.')


@dataset('burgers_inviscid', 'Inviscid Burgers equation', 'pde_1d', 'synthetic', 'core',
         'burgers/burgers_sln_100.csv')
def load_burgers_inviscid():
    data = np.loadtxt(data_path('burgers', 'burgers_sln_100.csv'), delimiter=',').T
    grids = mesh(np.linspace(0, 1, 101), np.linspace(-1000, 0, 101))
    return _problem(
        'burgers_inviscid', grids, {'u': data}, ('t', 'x'),
        truth=['-1.0 * u{power: 1.0} * du/dx1{power: 1.0} = du/dx0{power: 1.0}'],
        truth_alternatives=[
            ['1.0 * u{power: 1.0} = x{power: 1.0, dim: 1.0} * du/dx1{power: 1.0}'],
            ['0.5 * x{power: 1.0, dim: 0.0} * u{power: 1.0} + -0.5 * x{power: 1.0, dim: 1.0}'
             ' = du/dx1{power: 1.0} * x{power: 1.0, dim: 1.0}']],
        notes='The record is the similarity solution u = x / (t + c), so two identities hold as '
              'well as the PDE and count as correct.')


@dataset('wave', 'Wave equation', 'pde_1d', 'synthetic', 'core', 'wave/wave_sln_80.csv')
def load_wave():
    data = np.loadtxt(data_path('wave', 'wave_sln_80.csv'), delimiter=',').T
    grids = mesh(np.linspace(0, 1, 81), np.linspace(0, 1, 81))
    return _problem(
        'wave', grids, {'u': data}, ('t', 'x'),
        truth=['0.04 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}'],
        truth_alternatives=[
            ['48.67869238111131 * d^2u/dx1^2{power: 1.0} * d^2u/dx0^2{power: 1.0}'
             ' + -591.9797844706457 * d^2u/dx0^2{power: 2.0} = d^2u/dx1^2{power: 2.0}'],
            ['0.04012 * d^2u/dx1^2{power: 1.0} + 1.08338 * d^2u/dx0^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'
             ' + -0.04347 * d^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'
             ' = d^2u/dx0^2{power: 1.0}'],
            ['0.04012 * d^2u/dx1^2{power: 1.0} + 0.68737 * du/dx0{power: 1.0} * d^2u/dx0^2{power: 1.0}'
             ' + -0.02751 * d^2u/dx1^2{power: 1.0} * du/dx0{power: 1.0} = d^2u/dx0^2{power: 1.0}'],
            ['0.46769 * u{power: 1.0} * d^2u/dx0^2{power: 1.0} + -0.01878 * u{power: 1.0} * d^2u/dx1^2{power: 1.0}'
             ' + 0.04029 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}']],
        notes='u_tt = 0.04 u_xx. The alternatives are the wave equation multiplied by another '
              'factor; they are accepted as in the group\'s former benchmark.')


@dataset('kdv', 'Korteweg-de Vries equation (PDE-FIND data)', 'pde_1d', 'synthetic', 'core',
         'kdv/kdv_sindy.mat')
def load_kdv():
    m = _loadmat('kdv', 'kdv_sindy.mat')
    grids = mesh(np.ravel(m['t']), np.ravel(m['x']))
    return _problem(
        'kdv', grids, {'u': np.transpose(np.real(m['usol']))}, ('t', 'x'),
        truth=['-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        truth_alternatives=[
            ['-1.0 * u{power: 3.0} + 1.0 * du/dx1{power: 2.0} = u{power: 1.0} * d^2u/dx1^2{power: 1.0}'],
            ['-3.0 * u{power: 2.0} * du/dx1{power: 1.0} + 1.0 * du/dx1{power: 1.0} * d^2u/dx1^2{power: 1.0}'
             ' = d^3u/dx1^3{power: 1.0} * u{power: 1.0}'],
            ['-0.3333333333 * u{power: 1.0} * du/dx0{power: 1.0} + -0.3333333333 * du/dx1{power: 1.0}'
             ' * d^2u/dx1^2{power: 1.0} = u{power: 2.0} * du/dx1{power: 1.0}']],
        notes='u_t = -6 u u_x - u_xxx. The record is a soliton family, so three identities of it '
              'are also exact and accepted.')


@dataset('kdv_cossin', 'KdV with a cos(t)sin(x) source', 'pde_1d', 'synthetic', 'core',
         'kdv/data.csv')
def load_kdv_cossin():
    data = np.loadtxt(data_path('kdv', 'data.csv'), delimiter=',').T
    grids = mesh(np.linspace(0, 1, 81), np.linspace(0, 1, 81))
    return _problem(
        'kdv_cossin', grids, {'u': data}, ('t', 'x'),
        truth=['-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0}'
               ' + 1.0 * cos(t)sin(x){power: 1.0} = du/dx0{power: 1.0}'],
        extra_tokens=tokens.cos_t_sin_x,
        extra_arrays={'cos(t)sin(x)': (np.cos(grids[0]) * np.sin(grids[1]), True)},
        notes='The source enters as one product token cos(t)sin(x).')


@dataset('kdv_homogen', 'KdV, homogeneous, x in [-3, 3]', 'pde_1d', 'synthetic', 'extended',
         'kdv/data_kdv_homogen.npy')
def load_kdv_homogen():
    grids = mesh(np.linspace(0, 1, 120), np.linspace(-3, 3, 480))
    return _problem(
        'kdv_homogen', grids, {'u': np.load(data_path('kdv', 'data_kdv_homogen.npy'))}, ('t', 'x'),
        truth=['-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        notes="Same law on a short interval, as in the group's original experiments.")


@dataset('kdv_sga', 'KdV, SGA-PDE record (u_t = -u u_x - 0.0025 u_xxx)', 'pde_1d', 'synthetic',
         'extended', 'kdv/Kdv.mat')
def load_kdv_sga():
    m = _loadmat('kdv', 'Kdv.mat')
    grids = mesh(np.squeeze(m['tt']), np.squeeze(m['x']))
    return _problem(
        'kdv_sga', grids, {'u': m['uu'].T}, ('t', 'x'),
        truth=['-1.0 * du/dx1{power: 1.0} * u{power: 1.0} + -0.0025 * d^3u/dx1^3{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        notes='The record used in the symbolic genetic algorithm study.')


@dataset('ks', 'Kuramoto-Sivashinsky equation', 'pde_1d', 'synthetic', 'core',
         'ks/kuramoto_sivishinky.mat')
def load_ks(t_stride: int = 1, x_stride: int = 1):
    m = _loadmat('ks', 'kuramoto_sivishinky.mat')
    t, x, u = np.ravel(m['tt']), np.ravel(m['x']), m['uu'].T
    t, x, u = t[::t_stride], x[::x_stride], u[::t_stride, ::x_stride]
    return _problem(
        'ks', mesh(t, x), {'u': u}, ('t', 'x'),
        truth=['-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + -1.0 * d^2u/dx1^2{power: 1.0}'
               ' + -1.0 * d^4u/dx1^4{power: 1.0} = du/dx0{power: 1.0}'],
        meta={'t_stride': t_stride, 'x_stride': x_stride},
        notes='u_t = -u u_x - u_xx - u_xxxx; chaotic, needs a 4th derivative.')


def _pde_unit_grid():
    return mesh(np.linspace(0, 0.5, 251), np.linspace(1, 2, 100))


@dataset('pde_compound', 'Nonlinear diffusion u_t = (u u_x)_x', 'pde_1d', 'synthetic', 'core',
         'pde_compound/PDE_compound.npy')
def load_pde_compound():
    return _problem(
        'pde_compound', _pde_unit_grid(),
        {'u': np.load(data_path('pde_compound', 'PDE_compound.npy'))}, ('t', 'x'),
        truth=['1.0 * du/dx1{power: 2.0} + 1.0 * d^2u/dx1^2{power: 1.0} * u{power: 1.0}'
               ' = du/dx0{power: 1.0}'],
        notes='u_t = u_x^2 + u u_xx on t in [0, 0.5], x in [1, 2].')


@dataset('pde_divide', 'PDE with a 1/x coefficient', 'pde_1d', 'synthetic', 'core',
         'pde_divide/PDE_divide.npy')
def load_pde_divide():
    return _problem(
        'pde_divide', _pde_unit_grid(),
        {'u': np.load(data_path('pde_divide', 'PDE_divide.npy'))}, ('t', 'x'),
        truth=['-2.0 * du/dx1{power: 1.0} + 0.5 * d^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0}'
               ' = du/dx0{power: 1.0} * x{power: 1.0, dim: 1.0}'],
        notes='x u_t = -2 u_x + 0.5 x u_xx, i.e. u_t = -2 u_x / x + 0.5 u_xx (needs the x token).')


# =========================================================================
# PDEs in two or three space dimensions
# =========================================================================

@dataset('ns', 'Navier-Stokes, cylinder wake (Re = 100)', 'pde_2d', 'synthetic', 'core',
         'ns/cylinder_nektar_wake.mat')
def load_ns(subset: str = 'gate'):
    m = _loadmat('ns', 'cylinder_nektar_wake.mat')
    t = np.ravel(m['t'])
    x, y = np.unique(m['X_star'][:, 0]), np.unique(m['X_star'][:, 1])

    def field(a):                                   # (N, T) -> (t, y, x)
        return a.T.reshape(len(t), len(y), len(x))

    full = {'u': field(m['U_star'][:, 0, :]), 'v': field(m['U_star'][:, 1, :]),
            'p': field(m['p_star'])}
    if subset == 'gate':
        # gate.py's subset: first 50 time levels, every 2nd y and x node
        ts, ys, xs = slice(0, 50), slice(5, 45, 2), slice(0, 72, 2)
    elif subset == 'full50':
        ts, ys, xs = slice(0, 50), slice(None), slice(None)
    else:
        raise ValueError("subset must be 'gate' or 'full50'")
    data = {k: np.ascontiguousarray(f[ts][:, ys][:, :, xs]) for k, f in full.items()}
    return _problem(
        'ns', mesh(t[ts], y[ys], x[xs]), data, ('t', 'y', 'x'),
        truth=['-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0}'
               ' + -1.0 * dp/dx2{power: 1.0} + 0.01 * d^2u/dx2^2{power: 1.0}'
               ' + 0.01 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}',
               '-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0}'
               ' + -1.0 * dp/dx1{power: 1.0} + 0.01 * d^2v/dx2^2{power: 1.0}'
               ' + 0.01 * d^2v/dx1^2{power: 1.0} = dv/dx0{power: 1.0}',
               '-1.0 * dv/dx1{power: 1.0} = du/dx2{power: 1.0}'],
        meta={'subset': subset},
        notes='Axes (t, y, x): dx1 = d/dy, dx2 = d/dx. Two momentum equations (nu = 0.01) and '
              'continuity. By default a subset of about 36 thousand points; the full window has 250'
              ' thousand points per variable.')


@dataset('heat_solar_1d', 'Heat in soil under solar forcing, 1-D', 'pde_1d', 'synthetic', 'other',
         'heat_solar/heat_soil_uniform_1d_p1.npz')
def load_heat_solar_1d():
    d = np.load(data_path('heat_solar', 'heat_soil_uniform_1d_p1.npz'))
    return _problem(
        'heat_solar_1d', mesh(d['t'], d['x']), {'u': d['u'].squeeze().T}, ('t', 'x'),
        notes='Simulated soil temperature with a periodic surface flux. Expected law: the heat '
              'equation u_t = a u_xx in the interior (coefficient not stored), so no truth is '
              'scored. The data also hold the time derivative.')


@dataset('heat_solar_2d', 'Heat in soil under solar forcing, 2-D', 'pde_2d', 'synthetic', 'other',
         'heat_solar/heat_soil_uniform_2d_p1.npz')
def load_heat_solar_2d(t_stride: int = 4):
    d = np.load(data_path('heat_solar', 'heat_soil_uniform_2d_p1.npz'))
    u = np.transpose(d['u'].squeeze(), axes=(2, 0, 1))[::t_stride]
    return _problem(
        'heat_solar_2d', mesh(d['t'][::t_stride], d['x'], d['y']), {'u': u}, ('t', 'x', 'y'),
        meta={'t_stride': t_stride},
        notes='2-D version of heat_solar_1d (576 x 51 x 51 before striding in t).')


@dataset('heat_laser', 'Heat equation with a moving laser source, 3-D', 'pde_3d', 'synthetic',
         'other', 'heat_laser/heat_laser.npz')
def load_heat_laser(xy_stride: int = 4):
    d = np.load(data_path('heat_laser', 'heat_laser.npz'))
    u = np.transpose(d['u'].squeeze(), axes=(3, 0, 1, 2))[:, ::xy_stride, ::xy_stride, :]
    laser = np.load(data_path('heat_laser', 'laser.npy'))[:, ::xy_stride, ::xy_stride]   # (t, x, y)
    laser = np.ascontiguousarray(np.broadcast_to(laser[..., None], u.shape))
    return _problem(
        'heat_laser', mesh(d['t'], d['x'][::xy_stride], d['y'][::xy_stride], d['z']), {'u': u},
        ('t', 'x', 'y', 'z'), token_groups=[('laser', {'L': laser}, True)],
        meta={'xy_stride': xy_stride},
        notes='Only 3 points along z and 20 in t, so z- and t-derivatives are crude. The laser '
              'source is supplied as a field in (t, x, y), assumed uniform in z. No truth is '
              'scored.')


@dataset('jhtdb_plane', 'Isotropic turbulence, 2-D slice of JHTDB', 'pde_2d', 'synthetic',
         'extended', 'jhtdb/jhtdb_pilot_plane.npz')
def load_jhtdb_plane():
    d = np.load(data_path('jhtdb', 'jhtdb_pilot_plane.npz'))
    vel = d['velocity_field'].astype(np.float64)        # (t, y, x, 3)
    gu = d['velocity_gradient'].astype(np.float64)      # (t, y, x, 9)
    gp = d['pressure_gradient'].astype(np.float64)
    lap = d['velocity_laplacian'].astype(np.float64)
    nu, t = float(d['nu']), d['t'].astype(np.float64)
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    dt = float(t[1] - t[0])
    exact = {'p_x': gp[..., 0], 'p_y': gp[..., 1],
             'nu_lap_u': nu * lap[..., 0], 'nu_lap_v': nu * lap[..., 1]}
    out_of_plane = {'w': w, 'u_z': gu[..., 2], 'v_z': gu[..., 5]}
    # Column order of the derivative stacks is (t, y, x): spatial from the
    # DNS gradients, temporal by frame differencing.
    derivs = [np.column_stack([np.gradient(u, dt, axis=0).ravel(), gu[..., 1].ravel(), gu[..., 0].ravel()]),
              np.column_stack([np.gradient(v, dt, axis=0).ravel(), gu[..., 4].ravel(), gu[..., 3].ravel()])]
    return _problem(
        'jhtdb_plane', mesh(t, d['y'], d['x']), {'u': u, 'v': v}, ('t', 'y', 'x'),
        truth=['-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0}'
               ' + -1.0 * w{power: 1.0} * u_z{power: 1.0} + -1.0 * p_x{power: 1.0}'
               ' + 1.0 * nu_lap_u{power: 1.0} = du/dx0{power: 1.0}',
               '-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0}'
               ' + -1.0 * w{power: 1.0} * v_z{power: 1.0} + -1.0 * p_y{power: 1.0}'
               ' + 1.0 * nu_lap_v{power: 1.0} = dv/dx0{power: 1.0}'],
        token_groups=[('ns_exact', exact, True),
                      ('ns_oop_velocity', {'w': out_of_plane['w']}, True),
                      ('ns_oop_u_gradient', {'u_z': out_of_plane['u_z']}, False),
                      ('ns_oop_v_gradient', {'v_z': out_of_plane['v_z']}, False)],
        derivs=derivs, deriv_orders=[1, 1, 1],
        notes='48 x 48 slice (stride 8 of the DNS grid), 40 frames. Too coarse for numerical '
              'derivatives (aliased), so the exact server-side gradients are passed instead; '
              'out-of-plane and pressure terms enter as exact tokens. Artificial noise is unsupported '
              'with these supplied derivatives; use noise=0.')


# =========================================================================
# Real measurements
# =========================================================================

@dataset('pend_single', 'Real single pendulum (encoder, HardwareX rig)', 'ode', 'real', 'real',
         'dp/MultiArm_Pendulum/Single_FreeSwing_1.mat')
def load_pend_single(start: int = 4000, n: int = 5000, step: int = 40):
    m = _loadmat('dp', 'MultiArm_Pendulum', 'Single_FreeSwing_1.mat', squeeze_me=True,
                 struct_as_record=False)
    s = m['Theta1']
    t = np.asarray(s.time, float).ravel()
    th = np.asarray(s.signals.values, float).ravel()
    sl = slice(start, start + n * step, step)
    t, th = t[sl] - t[sl][0], th[sl]
    return _problem(
        'pend_single', mesh(t), {'theta': th - np.pi}, ('t',),
        truth=['-64.8 * theta{power: 1.0} + -0.65 * dtheta/dx0{power: 1.0} = d^2theta/dx0^2{power: 1.0}'],
        truth_alternatives=[['-64.8 * theta{power: 1.0} = d^2theta/dx0^2{power: 1.0}']],
        meta={'start': start, 'n': n, 'step': step},
        notes='Small swing about the hanging position, angle centred (theta - pi): a damped linear '
              'oscillator, -64.8 = -g/l. The undamped form also counts (damping is weak).')


@dataset('dp_encoder', 'Real double pendulum (encoder, HardwareX rig)', 'ode_system', 'real',
         'real', 'dp/MultiArm_Pendulum/Double_FreeSwing_1.mat')
def load_dp_encoder(start: int = 4000, n: int = 6000, step: int = 4):
    m = _loadmat('dp', 'MultiArm_Pendulum', 'Double_FreeSwing_1.mat')
    th1, th2 = m['Theta1'].ravel().astype(float), m['Theta2'].ravel().astype(float)
    t = m['Time'].ravel().astype(float)
    sl = slice(start, start + n * step, step)
    t, th1, th2 = t[sl] - t[sl][0], th1[sl], th2[sl]
    delta = th1 - th2
    return _problem(
        'dp_encoder', mesh(t), {'theta1': th1, 'theta2': th2}, ('t',),
        token_groups=[('trig_state', {'sin_th1': np.sin(th1), 'sin_th2': np.sin(th2)}, True),
                      ('coupling', {'cos_delta': np.cos(delta), 'sin_delta': np.sin(delta)}, False)],
        meta={'start': start, 'n': n, 'step': step},
        notes='Coupled Lagrangian equations with cos/sin of the angle difference; their exact '
              'coefficients are not known, so the run is not scored. sin(theta) and the coupling '
              'factors enter as tokens.')


def _daisy(name):
    with gzip.open(data_path('daisy', f'{name}.dat.gz'), 'rt') as f:
        return np.loadtxt(f)


@dataset('robot_arm', 'DaISy flexible robot arm (input torque -> acceleration)', 'ode', 'real',
         'real', 'daisy/robot_arm.dat.gz')
def load_robot_arm(dt: float = 0.01):
    arr = _daisy('robot_arm')
    u_in, y = arr[:, 0], arr[:, 1]
    return _problem(
        'robot_arm', mesh(np.arange(y.size) * dt), {'y': y}, ('t',),
        token_groups=[('forcing', {'u_in': u_in}, True)],
        meta={'dt': dt},
        notes='Unknown truth (a ~5th-order flexible structure). dt is not distributed with the '
              'file; 0.01 s is assumed (coefficients scale with it, structure does not).')


@dataset('ballbeam', 'DaISy ball and beam (beam angle -> ball position)', 'ode', 'real', 'real',
         'daisy/ballbeam.dat.gz')
def load_ballbeam(dt: float = 0.1):
    arr = _daisy('ballbeam')
    u_in, y = arr[:, 0], arr[:, 1]
    return _problem(
        'ballbeam', mesh(np.arange(y.size) * dt), {'y': y}, ('t',),
        token_groups=[('forcing', {'u_in': u_in}, True)],
        meta={'dt': dt},
        notes='Unknown truth; idealised physics is y\'\' proportional to the beam angle. '
              'Sampling period 0.1 s per the DaISy description.')


# =========================================================================
# Loadable, not benchmarked
# =========================================================================

@dataset('dp_sim', 'Simulated double pendulum', 'ode_system', 'synthetic', 'other', 'dp/dp.npz')
def load_dp_sim():
    d = np.load(data_path('dp', 'dp.npz'))
    t = d['t'] if 't' in d.files else np.arange(d['theta1'].size, dtype=float)
    return _problem(
        'dp_sim', mesh(t), {'theta1': d['theta1'].astype(float), 'theta2': d['theta2'].astype(float)},
        ('t',), meta={k: float(d[k]) for k in ('g', 'L1', 'L2', 'm1', 'm2')},
        notes='Used by the PINN studies in dp/. The equations of motion need coupling tokens and '
              'are not written in token form here.')


@dataset('dp_video', 'Real double pendulum (video tracking)', 'ode_system', 'real', 'other',
         'dp/Video_Tracking_Data/Video_Tracking_Data/Trial*/DPmean_data_RB*.npy')
def load_dp_video(trial: int = 1):
    folder = ('dp', 'Video_Tracking_Data', 'Video_Tracking_Data', f'Trial{trial}')
    upper = np.load(data_path(*folder, 'DPmean_data_RB0.npy'))
    lower = np.load(data_path(*folder, 'DPmean_data_RB1.npy'))
    return _problem(
        'dp_video', mesh(upper[0]), {'theta1': upper[1], 'theta2': lower[1]}, ('t',),
        meta={'trial': trial},
        notes='Mean link angles from video markers. Noisier than the encoder record, which is the '
              'preferred source.')


def _finite_rectangle(mask):
    """Largest axis-aligned rectangle that is finite at every time level."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, None)
    for row, valid in enumerate(mask):
        heights = np.where(valid, heights + 1, 0)
        stack = []
        for col in range(len(heights) + 1):
            height = int(heights[col]) if col < len(heights) else 0
            start = col
            while stack and stack[-1][1] > height:
                left, h = stack.pop()
                area = h * (col - left)
                if area > best[0]:
                    best = (area, (slice(row - h + 1, row + 1), slice(left, col)))
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if best[1] is None:
        raise ValueError('SST has no finite ocean rectangle')
    return best[1]


@dataset('sst', 'Sea surface temperature, ESA CCI L4 (Jan-Mar 2025)', 'pde_2d', 'real', 'other',
         'sst/sst_l4_files/*.nc (90 daily files)')
def load_sst(variable: str = 'analysed_sst', crop_ocean: bool = True):
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError('sst needs the optional netCDF4 package: pip install netCDF4') from exc
    files = sorted(data_path('sst', 'sst_l4_files').glob('*.nc'))
    if not files:
        raise FileNotFoundError("sst: no daily .nc files were found")
    frames, times = [], []
    for path in files:
        with netCDF4.Dataset(path) as ds:
            frames.append(np.ma.filled(ds.variables[variable][0].astype(np.float64), np.nan))
            times.append(float(ds.variables['time'][0]))
            lat = np.asarray(ds.variables['lat'][:], dtype=np.float64)
            lon = np.asarray(ds.variables['lon'][:], dtype=np.float64)
    time = (np.asarray(times) - times[0]) / 86400.0                     # days
    field = np.stack(frames)
    crop = (slice(0, len(lat)), slice(0, len(lon)))
    if crop_ocean:
        crop = _finite_rectangle(np.isfinite(field).all(axis=0))
        field = np.ascontiguousarray(field[:, crop[0], crop[1]])
        lat, lon = lat[crop[0]], lon[crop[1]]
    return _problem(
        'sst', mesh(time, lat, lon), {'T': field}, ('t', 'lat', 'lon'),
        meta={'crop_ocean': crop_ocean, 'lat_indices': [crop[0].start, crop[0].stop],
              'lon_indices': [crop[1].start, crop[1].stop]},
        notes='Daily analysed sea surface temperature (K), 90 days. By default the largest '
              'rectangular ocean region finite on every day is used; the original box with land '
              'masked can be loaded for plotting only.')


@dataset('darcy', 'Darcy flow -div(nu grad u) = 1 (data files missing)', 'pde_2d', 'synthetic',
         'other', 'darcy/darcy_1.0.npy, darcy/darcy_nu_1.0.npy (not in the repository)')
def load_darcy(sample: int = 0):
    """Port of the old darcy.py. UNTESTED: its data files are not in the
    repository (and the old script read the coefficient file relative to the
    working directory)."""
    u = np.load(data_path('darcy', 'darcy_1.0.npy'))[sample]
    nu = np.load(data_path('darcy', 'darcy_nu_1.0.npy'))[sample]
    x = y = np.linspace(0.0, 1.0, u.shape[0])
    step = x[1] - x[0]
    # stationary problem: a dummy time axis of two identical levels, as before
    data = np.stack([u, u], axis=0)

    def lift(field):
        return np.ascontiguousarray(np.broadcast_to(field, data.shape))

    coefficient = {'nu': lift(nu), 'dnu/dx': lift(np.gradient(nu, step, axis=0, edge_order=2)),
                   'dnu/dy': lift(np.gradient(nu, step, axis=1, edge_order=2))}
    mixed = {'d^2u/dxdy': np.gradient(np.gradient(data, step, axis=1, edge_order=2),
                                      step, axis=2, edge_order=2)}
    return _problem(
        'darcy', mesh([0.0, 1.0], x, y), {'u': data}, ('t', 'x', 'y'),
        truth=['-1.0 * du/dx2{power: 1.0} * dnu/dy{power: 1.0} + -1.0 * nu{power: 1.0} * d^2u/dx1^2{power: 1.0}'
               ' + -1.0 * nu{power: 1.0} * d^2u/dx2^2{power: 1.0} = du/dx1{power: 1.0} * dnu/dx{power: 1.0}'],
        token_groups=[('nu-tensors', coefficient, False), ('xy-tensor', mixed, True)],
        notes='Stationary: -nu (u_xx + u_yy) - nu_x u_x - nu_y u_y = 1 (the constant is not '
              'scored). The coefficient field and its gradients enter as factor tokens.')
