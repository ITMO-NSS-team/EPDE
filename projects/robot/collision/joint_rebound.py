"""Fit shared event-level rebound equations and hold out one recording at a time."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from epde.operators.common.multisample import joint_subset_table

RESULTS = ROOT / 'projects/robot/results'
VIDEO = RESULTS / 'collision_transfer/preprocessed/video_rebound_verification'
SWEEPS = RESULTS / 'pair_interactions_sweeps/physical'
DEST = VIDEO / 'joint'
MODELS = ('restitution', 'constant', 'affine')


def cohorts():
    events = pd.read_csv(VIDEO / 'event_predictions.csv')
    good = events[(events['mode'] == 'arena_registered') & events.quality_ok
                  & (events.incoming_px_s < 0)].copy()
    good['closing'] = -good.incoming_px_s
    good['outgoing'] = good.outgoing_px_s
    short = good[good.positions == 2]
    long = good[good.positions == 3]
    yield 'video_40ms', 'px/s', short
    yield 'video_80ms', 'px/s', long
    common = long[['recording', 'case']].merge(short[['recording', 'case']],
                                              validate='one_to_one')
    for name, frame in [('40ms', short), ('80ms', long)]:
        yield 'video_matched_' + name, 'px/s', frame.merge(common, validate='one_to_one')

    events = pd.read_csv(SWEEPS / 'events.csv.gz', usecols=[
        'experiment', 'N', 'frame', 'id_i', 'id_j', 'gate', 'incoming_short', 'immediate'])
    entries = events[(events.gate == 1) & (events.incoming_short < 0)].copy()
    assert not entries.duplicated(['experiment', 'N', 'frame', 'id_i', 'id_j']).any()
    diameter_mm = 48.7
    entries['closing'] = -entries.incoming_short * diameter_mm
    entries['outgoing'] = entries.immediate * diameter_mm
    yield 'track_entry_40ms', 'mm/s', entries.rename(columns={'experiment': 'recording'})


def joint_fit(groups):
    samples = [(name, group[['closing']].to_numpy(), group.outgoing.to_numpy(),
                np.ones(len(group))) for name, group in groups]
    # Bit 0 selects closing speed. Bit 1 selects the intercept.
    _, _, _, coefficients, full_rank = joint_subset_table(
        samples, subsets=[1, 2, 3], discrepancy='wape', weighting='samples')
    if not full_rank.all():
        raise ValueError('The shared coefficient or intercept is unidentifiable.')
    return coefficients


def errors(y, pred, weights):
    mse = np.average((y - pred) ** 2, weights=weights)
    var = np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)
    return np.sqrt(mse), 1 - mse / var if var > 0 else np.nan


def run():
    folds, summaries = [], []
    for cohort, units, data in cohorts():
        groups = list(data.groupby('recording', sort=True))
        full_fit = joint_fit(groups)
        predictions = []
        for held, test in groups:
            train = [(name, group) for name, group in groups if name != held]
            fitted = joint_fit(train)
            for model, (slope, intercept) in zip(MODELS, fitted):
                predicted = slope * test.closing.to_numpy() + intercept
                observed = test.outgoing.to_numpy()
                rmse, r2 = errors(observed, predicted, np.ones(len(test)))
                folds.append(dict(cohort=cohort, units=units, held=held, model=model,
                                  n=len(test), slope=slope, intercept=intercept, rmse=rmse, r2=r2))
                predictions.append(pd.DataFrame(dict(held=held, model=model,
                                                     observed=observed, predicted=predicted)))
        scores = pd.concat(predictions, ignore_index=True)
        for model, (slope, intercept) in zip(MODELS, full_fit):
            scored = scores[scores.model == model]
            weights = 1 / scored.groupby('held')['observed'].transform('size')
            rmse, r2 = errors(scored.observed, scored.predicted, weights)
            summaries.append(dict(cohort=cohort, units=units, model=model, events=len(scored),
                                  recordings=len(groups), all_data_slope=slope,
                                  all_data_intercept=intercept, heldout_rmse=rmse, heldout_r2=r2))
    DEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(DEST / 'by_recording.csv', index=False)
    pd.DataFrame(summaries).to_csv(DEST / 'summary.csv', index=False)


if __name__ == '__main__':
    run()
