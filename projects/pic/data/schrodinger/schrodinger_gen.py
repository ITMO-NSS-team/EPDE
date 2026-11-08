"""Exact data for the linear Schrodinger equation with a harmonic potential (hbar = m = omega = 1):

    i psi_t = -1/2 psi_xx + 1/2 x^2 psi

EPDE works on real fields, so psi = u + i v is split into a two-equation real system:

    u_t = -1/2 v_xx + 1/2 x^2 v
    v_t =  1/2 u_xx - 1/2 x^2 u

The solution is a finite sum over the oscillator eigenstates (Hermite functions phi_n, energies n + 1/2):

    psi(x, t) = sum_n c_n phi_n(x) exp(-i (n + 1/2) t),    c_n = <phi_n, psi_0>

so the field and its derivatives are exact up to the truncation of the sum (below 1e-15 of the norm), with no
time-stepping error. phi_n'' = (x^2 - 2 E_n) phi_n gives psi_xx exactly too.

The initial state is a displaced, squeezed Gaussian with momentum. It is a superposition of many eigenstates on
purpose: a single eigenstate psi = phi_n exp(-i E_n t) would also satisfy u_t = E_n v, a one-term identity that
the search would take instead of the Schrodinger equation.

Writes schrodinger_qho.npz: t, x, u, v (shape (nt, nx), axes (t, x)) and the exact u_t, v_t, u_xx, v_xx.
"""
import os

import numpy as np

X0, SIGMA, K0 = 1.0, 0.7, 0.5          # centre, width (ground state has 1), momentum of psi_0
X_RANGE, NX = (-7.5, 7.5), 256
T_RANGE, NT = (0.0, 2 * np.pi), 301    # one classical period
N_MAX = 120                            # eigenstates kept before truncation by norm


def hermite_functions(x: np.ndarray, n_max: int) -> np.ndarray:
    """phi_0 .. phi_{n_max - 1} on x, by the normalised (stable) three-term recurrence."""
    phi = np.empty((n_max, x.size))
    phi[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if n_max > 1:
        phi[1] = np.sqrt(2.0) * x * phi[0]
    for n in range(1, n_max - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * x * phi[n] - np.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def initial_state(x: np.ndarray) -> np.ndarray:
    psi = np.exp(-(x - X0) ** 2 / (2 * SIGMA ** 2) + 1j * K0 * x)
    return psi / (np.pi * SIGMA ** 2) ** 0.25          # unit L2 norm


def expansion_coefficients(n_max: int) -> np.ndarray:
    xq = np.linspace(-25.0, 25.0, 20001)               # psi_0 and phi_n both ~0 at the ends
    phi = hermite_functions(xq, n_max)
    c = np.trapezoid(phi * initial_state(xq)[None, :], xq, axis=1)
    kept = np.cumsum(np.abs(c) ** 2)
    n_keep = int(np.searchsorted(kept, kept[-1] * (1 - 1e-15))) + 1
    print(f'eigenstates: norm {kept[-1]:.15f} from {n_max}, keeping {n_keep}; '
          f'mean energy {np.sum(np.abs(c[:n_keep]) ** 2 * (np.arange(n_keep) + 0.5)):.4f}')
    return c[:n_keep]


def solve(t: np.ndarray, x: np.ndarray, c: np.ndarray):
    """psi, psi_t, psi_xx on the (t, x) grid."""
    energy = np.arange(c.size) + 0.5
    phi = hermite_functions(x, c.size)                          # (n, nx)
    phase = c[None, :] * np.exp(-1j * np.outer(t, energy))      # (nt, n)
    psi = phase @ phi
    psi_t = (-1j * energy[None, :] * phase) @ phi
    psi_xx = phase @ (phi * (x[None, :] ** 2 - 2 * energy[:, None]))
    return psi, psi_t, psi_xx


def fd4(f: np.ndarray, h: float, axis: int, order: int) -> np.ndarray:
    """Five-point central differences on the interior (edges left NaN): a check, not EPDE's own scheme."""
    f = np.moveaxis(f, axis, -1)
    out = np.full_like(f, np.nan)
    if order == 1:
        out[..., 2:-2] = (-f[..., 4:] + 8 * f[..., 3:-1] - 8 * f[..., 1:-3] + f[..., :-4]) / (12 * h)
    else:
        out[..., 2:-2] = (-f[..., 4:] + 16 * f[..., 3:-1] - 30 * f[..., 2:-2] + 16 * f[..., 1:-3]
                          - f[..., :-4]) / (12 * h ** 2)
    return np.moveaxis(out, -1, axis)


def check(t, x, u, v, u_t, v_t, u_xx, v_xx):
    """The true right-hand sides against the exact and the FD4 derivatives, and the OLS refit on FD4."""
    X2 = (x ** 2)[None, :]
    rhs_u, rhs_v = -0.5 * v_xx + 0.5 * X2 * v, 0.5 * u_xx - 0.5 * X2 * u
    print(f'exact residual  u: {np.abs(u_t - rhs_u).max():.2e}   v: {np.abs(v_t - rhs_v).max():.2e}   '
          f'(|u_t| max {np.abs(u_t).max():.2f})')
    dt, dx = t[1] - t[0], x[1] - x[0]
    inner = (slice(2, -2), slice(2, -2))
    for name, f, g in (('u', u, v), ('v', v, u)):
        target = fd4(f, dt, 0, 1)[inner].ravel()
        feats = np.stack([fd4(g, dx, 1, 2)[inner].ravel(), (X2 * g)[inner].ravel()], axis=1)
        coef, *_ = np.linalg.lstsq(feats, target, rcond=None)
        rel = np.linalg.norm(target - feats @ coef) / np.linalg.norm(target)
        other = 'v' if name == 'u' else 'u'
        print(f'FD4 OLS  {name}_t = {coef[0]:+.6f} {other}_xx {coef[1]:+.6f} x^2 {other}   rel. residual {rel:.2e}')


def main():
    t = np.linspace(*T_RANGE, NT)
    x = np.linspace(*X_RANGE, NX)
    psi, psi_t, psi_xx = solve(t, x, expansion_coefficients(N_MAX))
    u, v = psi.real, psi.imag
    check(t, x, u, v, psi_t.real, psi_t.imag, psi_xx.real, psi_xx.imag)
    edge = np.abs(psi[:, [0, -1]]).max()
    print(f'grid t {t.size} x {x.size} (dt {t[1] - t[0]:.4f}, dx {x[1] - x[0]:.4f}); |psi| max {np.abs(psi).max():.3f}, '
          f'at the x edges {edge:.1e}')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schrodinger_qho.npz')
    np.savez(out, t=t, x=x, u=u, v=v, u_t=psi_t.real, v_t=psi_t.imag, u_xx=psi_xx.real, v_xx=psi_xx.imag)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
