"""Clean identity preservation with noise-derived IRLS weighting; not a search."""
import estimator as E
M, R, Q, O, np = E.M, E.R, E.Q, E.O, E.np
import ode_simple_constants as D
from types import SimpleNamespace
import hashlib
from pathlib import Path


def main():
    t = np.linspace(0, 4*np.pi, 200); u = np.sin(t)+1.3*np.cos(t)
    sigma = .05*np.std(u); rng = np.random.default_rng(3)
    raw = u+sigma*rng.standard_normal(len(u))
    p = SimpleNamespace(shape=u.shape, data={'u': u}, grids=[t], named_arrays={})
    cs, labels = D.panel(); terms, opaque = R.symbols(p, labels); ci = labels.index(())
    idx = np.arange(0, len(t), 3); segment = slice(idx[0], idx[-1]+1)
    ax = [t[segment]]; opq = {k: v[segment] for k, v in opaque.items()}
    pilot, _ = R.reconstruct({'u': raw[idx]}, [t], idx)
    bank, meta = M.B.identify_bank(ax, terms, opq, {'u': pilot['u'][segment]}, cs, 2, 0)
    assert bank is not None
    basis = M.RealBasis(meta['k'], 1); tm = O.interpolation(t, idx)[segment]
    cov = O.Covariance(ax, tm, {'u': (sigma, [0.], [])}); draws = []
    for _ in range(512):
        data = {'u': pilot['u'][idx]+sigma*rng.standard_normal(len(idx))}
        draws.append(basis.convert(O.matrix(bank, O.map_values(tm, data), cov)))
    draws = np.stack(draws)
    clean = basis.convert(bank.matrix({'u': u[segment]}))
    rows = []
    for c in cs:
        if c['id'] not in ('energy', 'phase_sin', 'phase_cos'):
            continue
        coefficient, info = E.irls(clean, draws, c)
        assert info['status'] == 'ok'
        residual = clean[:, c['yi']]-clean[:, c['xi']]@coefficient
        energy = float(residual@residual/(clean[:, c['yi']]@clean[:, c['yi']]))
        missing = dict(c, xi=[i for i in c['xi'] if i != ci])
        no_constant, bad_info = E.irls(clean, draws, missing)
        if no_constant is not None:
            bad_residual = clean[:, c['yi']]-clean[:, missing['xi']]@no_constant
            bad_energy = float(bad_residual@bad_residual/(clean[:, c['yi']]@clean[:, c['yi']]))
        else:
            bad_energy = None
        assert energy < 1e-8
        np.testing.assert_allclose(coefficient, c['exact'], atol=1e-5, rtol=1e-5)
        assert bad_energy is None or bad_energy > energy
        rows.append(dict(identity=c['id'], coefficients=coefficient, clean_energy=energy,
                         constant=float(coefficient[c['xi'].index(ci)]), fit=info,
                         without_constant_energy=bad_energy, without_constant_fit=bad_info))
    R.dump(E.ROOT/'invariant_checks.json', dict(control_only=True, maximum_derivative_order=1,
           energy_derivative_power=2, phase_derivative_power=1, rows=rows,
           all_constants_preserved=True,
           source_sha256={str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (__file__, E.__file__, D.__file__)}))
    print([(r['identity'], r['constant'], r['clean_energy']) for r in rows])


if __name__ == '__main__':
    main()
