"""Earlier-motion matching and held-out error intervals for the 80 ms distance law."""
from collections import defaultdict
import json
from zlib import crc32

import numpy as np
import pandas as pd

from interaction_paths import BASIS, KEY, OUT
from shared_interaction_compact import development
from shared_interaction_data import D_MM
from shared_interaction_fit import balanced, family, periods
from third_neighbor_study import CALIPERS, STRATA, mirror_matches


def block_means(frame, values):
    """Resample occupied 30-second blocks; one shared draw for all comparisons."""
    block_ids = frame.groupby(['stage', 'block'], sort=True).ngroup().to_numpy()
    counts = np.bincount(block_ids)
    sums = np.zeros((len(counts), values.shape[1]))
    np.add.at(sums, block_ids, values)
    seed = crc32(str(frame.recording.iloc[0]).encode()) + 731
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=4000)
    sampled = (weights @ sums) / (weights @ counts)[:, None]
    return values.mean(axis=0), sampled, len(counts)


def interval(draws):
    return np.quantile(draws, [.025, .975], axis=0)


def motion_matches(events):
    eligible = events[(events.abs_third >= .03) & ~events.third_flagged].dropna(
        subset=['earlier_t', 'lead_t']).copy()
    records, families, details, coverage = [], [], [], []
    measures = ['earlier_t', 'lead_t', 'long_t', 'side', 'y40_t', 'y80_t', 'y400_t']
    history_limits = {'common history': {}, 'lead 0.40': {'lead_t': .40},
                      'lead 0.25': {'lead_t': .25}, 'both 0.25': {'lead_t': .25, 'earlier_t': .25}}
    for name, extra in history_limits.items():
        limits = {**CALIPERS, **extra}
        pairs = mirror_matches(eligible, 1., limits, min_time_gap=1.280001)
        if not len(pairs):
            raise ValueError(f'No supported matches for {name}')
        assert len(np.unique(pairs)) == pairs.size
        pos, neg = [eligible.loc[ix].reset_index(drop=True) for ix in pairs.T]
        assert pos[STRATA].equals(neg[STRATA]) and (abs(pos.seconds-neg.seconds)>1.28).all()
        diff = pos[list(limits)] - neg[list(limits)]
        assert (abs(diff) <= pd.Series(limits) + 1e-12).all().all()
        table = pos[STRATA].assign(design=name).copy()
        for field in ['frame', 'i', 'j', 'third_t']:
            table['positive_'+field], table['negative_'+field] = pos[field], neg[field]
        table[measures] = D_MM*(pos[measures]-neg[measures])
        table['family'] = table.recording.map(family)
        details.append(table)
        for tag, pool in eligible.groupby('recording'):
            matched = table[table.recording==tag]
            coverage.append(dict(design=name, recording=tag, eligible=len(pool), pairs=len(matched),
                one_neighbor_pairs=int((matched.crowd==1).sum()), blocks=matched.groupby(['stage','block']).ngroups))
        for fam, group in table.groupby('family'):
            across, samples = [], []
            for tag, sub in group.groupby('recording'):
                point, draws, blocks = block_means(sub, sub[measures].to_numpy())
                across.append(point)
                samples.append(draws)
                lo, hi = interval(draws) if blocks>1 else (np.full(len(measures),np.nan),)*2
                for k, col in enumerate(measures):
                    records.append(dict(design=name, recording=tag, family=fam, measure=col,
                        pairs=len(sub), blocks=blocks, mean_mm_s=point[k], low=lo[k], high=hi[k]))
            points = np.stack(across)
            lo, hi = interval(np.mean(samples,axis=0))
            for k, col in enumerate(measures):
                families.append(dict(design=name, family=fam, measure=col, pairs=len(group),
                    recordings=len(points), positive_records=(points[:,k]>0).sum(), mean_mm_s=points[:,k].mean(),
                    low=lo[k], high=hi[k]))
    return {'motion_match_pairs.parquet':pd.concat(details,ignore_index=True),
            'motion_match_recordings.csv':pd.DataFrame(records), 'motion_match_families.csv':pd.DataFrame(families),
            'motion_match_coverage.csv':pd.DataFrame(coverage)}


def triangle_fit(train):
    z = np.clip((1.30-train.r.to_numpy())/.30, 0, 1)
    x = np.column_stack([z*train.side, train.third_t])
    w = np.sqrt(balanced(train))
    return np.linalg.lstsq(w[:, None]*x, w*(train.y80_t-train.side), rcond=None)[0]


def kernel_intervals(events):
    fronts = pd.read_csv(OUT/'distance_fronts.csv').query('horizon == 80')
    archive = pd.read_csv(OUT/'distance_scores.csv').query('horizon == 80').set_index(['fold','recording'])
    prefix, _ = periods(events)
    summaries = []
    group_values = defaultdict(list)
    for (design, fold), supports in fronts.groupby(['design','fold']):
        supports = supports.sort_values('mask')
        assert supports['mask'].tolist() == list(range(1,32,2))
        active = supports.on_front.to_numpy()
        previous = np.flatnonzero(supports.chosen.to_numpy()).item()
        assert active[previous]
        is_test = events.recording.eq(fold) if design=='recording' else events.family.ne(fold.removeprefix('train_'))
        tri = triangle_fit(events[~is_test & prefix])
        fits = np.stack(supports.refit_beta.map(json.loads))
        for tag, rec in events[is_test].groupby('recording'):
            z = np.clip((1.30-rec.r.to_numpy())/.30,0,1)
            x = np.column_stack([z*rec.side, rec[BASIS]])
            base = rec.side.to_numpy()-rec.y80_t.to_numpy()
            triangle = base + x[:,0]*tri[0] + rec.third_t.to_numpy()*tri[1]
            errors = D_MM*np.column_stack([triangle, base[:,None]+x@fits.T])
            overlap = (rec.r<1.) | (rec.nearest_third<1.)
            clean = ~rec.kernel_flagged
            for cohort, keep in [('all',np.ones(len(rec),bool)), ('no apparent overlap',clean & ~overlap),
                                 ('apparent overlap',clean & overlap)]:
                if not np.any(keep):
                    continue
                g = rec[keep]
                sq, sampled, blocks = block_means(g, errors[keep]**2)
                rmse, rep = np.sqrt(sq), np.sqrt(sampled)
                gains, draws = rmse[0]-rmse[1:], rep[:,[0]]-rep[:,1:]
                variance = (D_MM*g.y80_t.to_numpy()).var()
                r2 = 1-sq/variance
                lo, hi = interval(draws) if blocks>1 else (np.full(len(fits),np.nan),)*2
                if cohort=='all':
                    old = archive.loc[(fold,tag)]
                    np.testing.assert_allclose([rmse[0],rmse[previous+1],r2[previous+1]],
                        [old.triangle_rmse,old.learned_rmse,old.learned_r2],atol=1e-10)
                for k, row in enumerate(supports.itertuples()):
                    entry = dict(design=design, fold=fold, recording=tag, family=family(tag), cohort=cohort,
                        mask=row.mask, on_front=row.on_front, previous=row.chosen, events=len(g), blocks=blocks,
                        rmse=rmse[k+1], triangle_rmse=rmse[0], r2=r2[k+1], triangle_r2=r2[0],
                        gain=gains[k], low=lo[k], high=hi[k])
                    summaries.append(entry)
                    group_values[design,family(tag),cohort,str(row.mask)].append((entry,draws[:,k]))
                    if k==previous:
                        group_values[design,family(tag),cohort,'archived'].append((entry,draws[:,k]))
    groups = []
    for (design, fam, cohort, mask), rows in group_values.items():
        a = pd.DataFrame([r for r,_ in rows])
        lo, hi = interval(np.mean([v for _,v in rows],axis=0))
        groups.append(dict(design=design,family=fam,cohort=cohort,mask=mask,recordings=len(a),
            events=a.events.sum(),on_front_recordings=a.on_front.sum(),positive_records=(a.gain>0).sum(),
            rmse=a.rmse.mean(),triangle_rmse=a.triangle_rmse.mean(),r2=a.r2.mean(),triangle_r2=a.triangle_r2.mean(),
            gain=a.gain.mean(),low=lo,high=hi))
    return {'kernel80_recordings.csv':pd.DataFrame(summaries), 'kernel80_families.csv':pd.DataFrame(groups)}


if __name__ == '__main__':
    d = development().merge(pd.read_parquet(OUT/'event_details.parquet'),on=KEY,validate='one_to_one')
    assert len(d)==46467 and not d[KEY].duplicated().any()
    d['block'] = (d.seconds//30).astype(int)
    d['family'] = d.recording.map(family)
    outputs = motion_matches(d)
    outputs.update(kernel_intervals(d))
    for name, table in outputs.items():
        if name.endswith('.parquet'):
            table.to_parquet(OUT/name,index=False)
        else:
            table.to_csv(OUT/name,index=False)
    print('Scored earlier-motion matches and all 80 ms kernel supports on eight recordings.')
