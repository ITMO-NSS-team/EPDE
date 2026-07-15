"""Data adapter for the JHTDB isotropic-turbulence pilot plane
(KNOWN-TRUTH config: 3D incompressible Navier-Stokes momentum on a 2D
slice, out-of-plane and pressure terms injected as exact tokens).

Data: projects/pic/data/jhtdb/jhtdb_pilot_plane.npz (jhtdb_pilot.py) —
isotropic1024coarse, 48x48 x-y plane at DNS grid nodes (stride 8),
40 frames at dt = 0.002. Velocity/pressure via lag8, all spatial
derivative fields server-side (fd8noint at grid nodes, DNS-exact).

Signal check (jhtdb_signal_check.py): momentum balance closes at
R^2 = 0.991-0.992 per component, OLS coefficients [1.002, 1.03] vs the
theoretical [1, 1]; residual = du/dt frame-differencing error.

State variables: u, v (in-plane velocity). Exact injected tokens:
    w                 out-of-plane velocity (factor for w*u_z)
    u_z, v_z          out-of-plane derivatives (factor-only)
    p_x, p_y          pressure gradient (standalone forcing terms)
    nu_lap_u, nu_lap_v  full viscous term nu*lap(u_i) (standalone)

Axis order is (t, y, x) [ij meshgrid], so du/dx1 = du/dy, du/dx2 = du/dx.
"""

import os

import numpy as np

_NPZ = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'jhtdb',
    'jhtdb_pilot_plane.npz'
))

_LAST = {}

TOKENS_NEED_SEARCH = True


def load_data():
    d = np.load(_NPZ)
    vel = d['velocity_field'].astype(np.float64)       # (nt, ny, nx, 3)
    gu = d['velocity_gradient'].astype(np.float64)     # (nt, ny, nx, 9)
    gp = d['pressure_gradient'].astype(np.float64)     # (nt, ny, nx, 3)
    lu = d['velocity_laplacian'].astype(np.float64)    # (nt, ny, nx, 3)
    t, x, y, nu = d['t'], d['x'], d['y'], float(d['nu'])

    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    dt = float(t[1] - t[0])
    _LAST.update(
        w=w,
        u_z=gu[..., 2], v_z=gu[..., 5],
        p_x=gp[..., 0], p_y=gp[..., 1],
        nu_lap_u=nu * lu[..., 0], nu_lap_v=nu * lu[..., 1],
        # exact derivative stacks for load_derivs, column order (t, y, x):
        # spatial from the server-side DNS gradients, temporal by frame
        # differencing (np.gradient: central interior, one-sided ends).
        derivs_u=np.column_stack([np.gradient(u, dt, axis=0).ravel(),
                                  gu[..., 1].ravel(), gu[..., 0].ravel()]),
        derivs_v=np.column_stack([np.gradient(v, dt, axis=0).ravel(),
                                  gu[..., 4].ravel(), gu[..., 3].ravel()]),
    )
    grids = np.meshgrid(t, y, x, indexing='ij')
    return tuple(grids), [u, v], ['u', 'v'], 2


def load_derivs():
    """Exact per-variable derivative stacks (see load_data): the stride-8
    sampled turbulent field is aliased, so numerically re-derived
    gradients are ~90% wrong (corr 0.43 vs DNS truth) -- the preprocessor
    must be bypassed."""
    return [_LAST['derivs_u'], _LAST['derivs_v']]


def build_extra_tokens(coords, dim):
    from epde import CacheStoredTokens

    # standalone-capable exact terms of the momentum balance
    standalone = CacheStoredTokens(
        token_type='ns_exact',
        token_labels=['p_x', 'p_y', 'nu_lap_u', 'nu_lap_v'],
        token_tensors={k: _LAST[k] for k in
                       ('p_x', 'p_y', 'nu_lap_u', 'nu_lap_v')},
        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
        dimensionality=dim, meaningful=True)
    # out-of-plane quantities: appear multiplied (w * u_z), factor-only
    factors = CacheStoredTokens(
        token_type='ns_oop',
        token_labels=['w', 'u_z', 'v_z'],
        token_tensors={k: _LAST[k] for k in ('w', 'u_z', 'v_z')},
        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
        dimensionality=dim, meaningful=False)
    return [standalone, factors]
