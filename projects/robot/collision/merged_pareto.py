"""One structure-level Pareto front per formulation across all omitted recordings."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from discovery import weak_design, window
from pareto import COLS, LABELS, MASKS, coefficients, gram, pareto_flags, terms, write_rows
from physics import LAG, OUT, arena_for, design, fit_self, recordings


def aggregate(data, tags):
    """Align every candidate by mask before comparing the two fold-mean objectives."""
    blocks = [data.loc[data.heldout == tag].set_index('mask').sort_index() for tag in tags]
    if set(data.heldout) != set(tags) or any(not np.array_equal(b.index, MASKS) for b in blocks):
        raise ValueError('Expected every candidate exactly once for every held-out recording.')
    def stack(key):
        return np.stack([b[key].to_numpy() for b in blocks])
    wape = stack('screen_wape').mean(0)
    dyad = (stack('screen_dyad') / len(tags)).sum(0)
    eligible = stack('native_eligible').astype(bool).all(0)
    flags = pareto_flags(wape, dyad, eligible)
    ids = sorted(np.flatnonzero(flags), key=lambda i: (dyad[i], wape[i], i))
    # Held-out scores enter only after membership of the merged front is frozen.
    r2 = stack('heldout_r2')
    rms = {k: np.sqrt(np.mean(stack(k)**2, axis=0)) for k in
           ('validation_rmse', 'heldout_rmse', 'heldout_pair_normal_rmse')}
    return [dict(mask=i, terms=terms(i), size=len(COLS[i]), mean_screen_wape=wape[i],
                 mean_screen_dyad=dyad[i], mean_heldout_r2=r2[:, i].mean(),
                 **{k: v[i] for k, v in rms.items()}, **{'r2_' + t: r2[j, i] for j, t in enumerate(tags)},
                 eligible_structures=int(eligible.sum())) for i in ids]


def merged_fronts(archive, output_dir, tracks):
    columns = ['heldout', 'form', 'mask', 'native_eligible', 'screen_wape', 'screen_dyad',
               'validation_rmse', 'heldout_rmse', 'heldout_r2', 'heldout_pair_normal_rmse']
    data = pd.read_csv(archive, usecols=columns)
    forms = ('strong', 'weak')
    if set(data.form) != set(forms):
        raise ValueError('Expected both collision formulations.')
    fronts = {form: aggregate(data.loc[data.form == form], sorted(tracks)) for form in forms}
    # One reusable coefficient vector per retained structure. The validation and test
    # scores above always use the original fold fits, with the omitted recording unseen.
    train = list(tracks.values())
    arenas = arena_for(train)
    lookup, common = fit_self(train, arenas)
    matrices = {form: [] for form in forms}
    for tr in train:
        rows, x, base, y, _ = design(tr, lookup, common, arenas, stride=1)
        wx, wy = weak_design(tr.pos[rows], x, base, window(tr.pos[rows]))
        for form, features, target in (('strong', x[::LAG], y[::LAG].astype(float) - base[::LAG]),
                                       ('weak', wx, wy)):
            target = target.reshape(-1)
            features = features.reshape(len(target), -1)
            matrices[form].append(gram(np.column_stack([features, np.ones(len(target)), target])))
    for form, members in fronts.items():
        g = np.mean(matrices[form], axis=0)
        for row in members:
            ix = COLS[row['mask']]
            b = np.zeros(len(LABELS))
            b[ix] = np.linalg.lstsq(g[np.ix_(ix, ix)], g[ix, -1], rcond=1e-10)[0]
            row['all_recording_refit_equation'] = coefficients(b)
        write_rows(output_dir / f'merged_{form}.csv', members)
    from pareto_report import plot_merged
    plot_merged(output_dir, fronts)
    return fronts


if __name__ == '__main__':
    from run import check_inputs
    check_inputs()
    dest = OUT / 'pareto'
    with TemporaryDirectory(dir=dest) as tmp:
        stage = Path(tmp)
        generated = merged_fronts(dest / 'all_candidates.csv.gz', stage, recordings())
        for form, members in generated.items():
            (stage / f'merged_{form}.csv').replace(dest / f'merged_{form}.csv')
            print(form, len(members), flush=True)
        (stage / 'merged_fronts.png').replace(dest / 'merged_fronts.png')
