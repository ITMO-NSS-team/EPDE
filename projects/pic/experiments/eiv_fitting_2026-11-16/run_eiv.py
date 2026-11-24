"""Fresh-seed coefficient experiment on frozen local/broad weak geometries."""
import estimator as E
M, np, R, Q, O, J, F = E.M, E.np, E.R, E.Q, E.O, E.J, E.F
from pathlib import Path
import argparse, hashlib, json, time
ROOT = E.ROOT
DRAWS = 512


def source_hashes():
    paths = [Path(__file__), Path(E.__file__), Path(M.__file__), Path(O.__file__),
             Path(Q.__file__), Path(R.__file__), Path(J.__file__),
             Path(R.WF.__file__), Path(R.WF._central4.__code__.co_filename)]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_broad(axes, terms, opaque, plan, k):
    signed = np.r_[np.arange(k+1), -np.arange(1, k+1)]
    frequencies = [signed*np.pi/(m*(a[1]-a[0])) for m, a in zip(plan['ms'], axes)]
    return Q.Bank(axes, terms, opaque, plan=plan, frequencies=frequencies)


def natural_rows(plan):
    ids = np.arange(np.prod(plan['shape'])).reshape(plan['shape'])
    sl = tuple(slice(None, None, int(np.ceil(m/s))) for m, s in zip(plan['ms'], plan['ss']))
    return ids[sl].reshape(-1)


def fit_methods(point, uncorrected, pilot, draws, reconstructed, candidates, correction_count):
    prepared = J.prepare(np.concatenate([pilot[None], reconstructed]), correction_count)
    out = []
    for c in candidates:
        for method, mat in [('raw_ols', uncorrected), ('corrected_raw_ols', point), ('recon_ols', pilot)]:
            coef, info = E.ordinary(mat, c)
            out.append(dict(candidate=c['id'], method=method, coefficients=coef, fit=info))
        coef, info = R.signed_fit(prepared['point_clsm'], pilot, c['xi'], c['yi'])
        out.append(dict(candidate=c['id'], method='recon_clsm', coefficients=coef, fit=dict(status='ok', **info)))
        coef, info = E.irls(point, draws, c)
        out.append(dict(candidate=c['id'], method='raw_irls', coefficients=coef, fit=info))
    return out


def run(name, seed):
    started = time.monotonic()
    base_path, old = F.baseline(name, 0)
    paths = dict(base=base_path,
                 local=F.ROOT/'fit_results'/f'{name}_seed0.json',
                 broad=J.ROOT/'results'/f'{name}_seed0.json',
                 validation=M.ROOT/'results'/f'{name}_seed0.json')
    saved = {k: json.loads(p.read_text()) for k, p in paths.items() if k != 'base'}
    p = R.datasets.load(name); axes = R.axes_of(p)
    all_candidates, labels = R.panel(p); candidates = [c for c in all_candidates if c['registered']]
    terms, opaque = R.symbols(p, labels); ci = labels.index(())
    raw = p.noisy(5, seed)
    indices = [np.array(g['indices']) for g in old['groups']]
    assert all(not np.intersect1d(indices[i], indices[j]).size for i in range(3) for j in range(i))
    vsegment = slice(*saved['validation']['segment'])
    vaxes = [axes[0][vsegment], *axes[1:]]; vopaque = {k: v[vsegment] for k, v in opaque.items()}
    vmeta = next(meta for key, meta in saved['validation']['banks'].items() if key.startswith('q0_'))
    assert len({f['selection']['q0']['key'] for f in saved['validation']['folds']}) == 1
    validation = make_broad(vaxes, terms, vopaque, vmeta['plan'], vmeta['k'])
    vbasis = M.RealBasis(vmeta['k'], len(axes))
    groups, fit_folds, bank_controls = [], [], []
    print(name, seed, 'fresh-seed IRLS, frozen geometry', flush=True)
    for g, idx in enumerate(indices):
        observed = {v: y[idx] for v, y in raw.items()}
        model = {v: R.WF.noise_model(y) for v, y in observed.items()}
        pilot, reconstruction = R.reconstruct(observed, axes, idx)
        segment, ax, opq, plan = Q.mapped_plan(old, g, saved['local']['plans'][g]['plan'], axes, opaque)
        local = Q.Bank(ax, terms, opq, plan=plan)
        bmeta = saved['broad']['folds'][g]['banks']['broad0']
        broad = make_broad(ax, terms, opq, bmeta['plan'], bmeta['k'])
        basis = M.RealBasis(bmeta['k'], len(axes)); selected = natural_rows(plan)
        tm = O.interpolation(axes[0], idx)[segment]
        vm = O.interpolation(axes[0], idx)[vsegment]
        assert ax[0][0] >= axes[0][idx[0]] and ax[0][-1] <= axes[0][idx[-1]]
        assert vaxes[0][0] >= axes[0][idx[0]] and vaxes[0][-1] <= axes[0][idx[-1]]
        cov, vcov = O.Covariance(ax, tm, model), O.Covariance(vaxes, vm, model)
        mapped, mappedv = O.map_values(tm, observed), O.map_values(vm, observed)
        pp = {v: y[segment] for v, y in pilot.items()}
        st = {}
        for tag, bank, convert in [('local_dense', local, lambda a: a), ('broad', broad, basis.convert)]:
            st[tag] = dict(point=convert(O.matrix(bank, mapped, cov)),
                           uncorrected=convert(bank.matrix(mapped)), pilot=convert(bank.matrix(pp)),
                           draws=[], reconstructed=[])
        vpoint = vbasis.convert(O.matrix(validation, mappedv, vcov))
        vpilot = vbasis.convert(validation.matrix({v: y[vsegment] for v, y in pilot.items()}))
        vdraws = []
        rng = np.random.default_rng(np.random.SeedSequence([175103, seed, g]))
        for b in range(DRAWS):
            draw = {v: pilot[v][idx]+R.WF.ar_noise(rng, y.shape, *model[v][:2]) for v, y in observed.items()}
            fields = O.map_values(tm, draw)
            st['local_dense']['draws'].append(O.matrix(local, fields, cov))
            st['broad']['draws'].append(basis.convert(O.matrix(broad, fields, cov)))
            if b < old['draws']:
                reconstructed, _ = R.reconstruct(draw, axes, idx)
                rr = {v: y[segment] for v, y in reconstructed.items()}
                st['local_dense']['reconstructed'].append(local.matrix(rr))
                st['broad']['reconstructed'].append(basis.convert(broad.matrix(rr)))
            if b < 128:
                vdraws.append(vbasis.convert(O.matrix(validation, O.map_values(vm, draw), vcov)))
        for bank in st.values():
            for key in ('draws', 'reconstructed'):
                bank[key] = np.stack(bank[key])
        st['local'] = {key: (value[:, selected] if key in ('draws', 'reconstructed') else value[selected])
                       for key, value in st['local_dense'].items()}
        fits = []
        for tag, bank in st.items():
            rows = fit_methods(**bank, candidates=candidates, correction_count=old['draws'])
            for row in rows:
                row['bank'] = tag
            fits.extend(rows)
        w, wmeta = M.whitening(np.stack(vdraws), vpilot)
        assert w is not None, wmeta
        # Finite deterministic constant column is unchanged by mean correction.
        for tag, bank in st.items():
            np.testing.assert_allclose(bank['point'][:, ci], bank['uncorrected'][:, ci], atol=3e-13, rtol=3e-13)
        groups.append(dict(point=vpoint, pilot=vpilot, weight=w, weight_meta=wmeta,
                           noise=model, observed_sha256={v: R.digest(y) for v, y in observed.items()},
                           reconstruction=reconstruction))
        fit_folds.append(dict(training=g, rows=fits, local_plan=plan, local_natural_indices=selected,
                              broad_plan=bmeta, local_dense_rows=len(st['local_dense']['point']),
                              local_rows=len(selected), broad_rows=len(st['broad']['point'])))
        # Freeze these coefficients before opening clean fields below.
        bank_controls.append((segment, local, broad, basis, selected, tm))
        print(' group', g, 'fits', [(r['bank'], r['fit']['status'], r['fit'].get('iterations'))
                                    for r in fits if r['method'] == 'raw_irls'],
              round(time.monotonic()-started, 1), 's', flush=True)
    # Only evaluation uses the saved clean trajectories and exact coefficients.
    frozen = F.evaluation_grams(p, old, terms, opaque, labels)
    vclean = vbasis.convert(validation.matrix({v: y[vsegment] for v, y in p.data.items()}))
    by_id = {c['id']: c for c in candidates}
    for g, fold in enumerate(fit_folds):
        left, right = [h for h in range(3) if h != g]
        weight = groups[g]['weight']; wpilot = weight @ groups[g]['pilot']
        aa, bb, clean = weight @ groups[left]['point'], weight @ groups[right]['point'], weight @ vclean
        scale = np.sum(wpilot**2, axis=0)
        extra = [dict(bank='control', method='exact', candidate=c['id'], coefficients=c['exact'], fit=dict(status='ok'))
                 for c in candidates]
        segment, local, broad, basis, selected, tm = bank_controls[g]
        raw_clean = {v: y[indices[g]] for v, y in p.data.items()}
        for kind, values in [('clean_fit', {v: y[segment] for v, y in p.data.items()}),
                             ('interpolated_clean_fit', O.map_values(tm, raw_clean))]:
            dense = local.matrix(values); fullb = basis.convert(broad.matrix(values))
            for tag, mat in [('local_dense', dense), ('local', dense[selected]), ('broad', fullb)]:
                for c in candidates:
                    coef, info = E.ordinary(mat, c)
                    extra.append(dict(bank=tag, method=kind, candidate=c['id'], coefficients=coef, fit=info))
        for row in fold['rows']+extra:
            if row['coefficients'] is None:
                continue
            c = by_id[row['candidate']]; coef = np.asarray(row['coefficients'])
            vector = np.zeros(len(labels)); vector[c['yi']] = 1.; vector[c['xi']] -= coef
            rl, rr, rc = aa@vector, bb@vector, clean@vector
            pl, pr = groups[left]['point']@vector, groups[right]['point']@vector
            row.update(cross=float(rl@rr/scale[c['yi']]),
                       ordinary=float((rl@rl+rr@rr)/(2*scale[c['yi']])),
                       plain_cross=float(pl@pr/(groups[g]['pilot'][:, c['yi']]@groups[g]['pilot'][:, c['yi']])),
                       clean_validation=float(rc@rc/scale[c['yi']]),
                       frozen_clean_energy=float(vector@frozen[f'{g}_loo_oracle']@vector /
                                                 frozen[f'{g}_loo_clean_scale'][c['yi']]),
                       relative_coefficient_error=float(np.linalg.norm(coef-c['exact'])/np.linalg.norm(c['exact'])),
                       coefficient_error=coef-c['exact'])
        fold['rows'].extend(extra)
    aggregates = []
    keys = sorted({(r['bank'], r['method'], r['candidate']) for f in fit_folds for r in f['rows']})
    for bank, method, candidate in keys:
        rows = [r for f in fit_folds for r in f['rows'] if (r['bank'], r['method'], r['candidate']) == (bank, method, candidate)]
        good = [r for r in rows if r['fit']['status'] == 'ok' and r['coefficients'] is not None]
        entry = dict(bank=bank, method=method, candidate=candidate, successful_folds=len(good),
                     statuses=[r['fit']['status'] for r in rows])
        if len(good) == 3:
            for key in ('cross', 'ordinary', 'plain_cross', 'clean_validation', 'frozen_clean_energy', 'relative_coefficient_error'):
                entry[key] = float(np.mean([r[key] for r in good]))
        aggregates.append(entry)
    out = dict(system=name, seed=seed, noise_percent=5, geometry_seed=0, draws=DRAWS,
               reconstruction_correction_draws=old['draws'], candidates=candidates, labels=labels,
               folds=fit_folds, aggregates=aggregates,
               groups=[{k: v for k, v in group.items() if k not in ('point', 'pilot', 'weight')} for group in groups],
               input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
               source_sha256=source_hashes(), clean_data_sha256={v: R.digest(y) for v, y in p.data.items()},
               raw_data_sha256={v: R.digest(y) for v, y in raw.items()}, elapsed_seconds=time.monotonic()-started)
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json', out)
    print('DONE', name, seed, round(out['elapsed_seconds'], 1), 's', flush=True)


def main():
    if R.os.name == 'nt':
        R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=['ac', 'kdv_cossin', 'wave', 'ks'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[3, 4, 5])
    args = ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            path = ROOT/'results'/f'{name}_seed{seed}.json'
            if not path.exists():
                run(name, seed)


if __name__ == '__main__':
    main()
