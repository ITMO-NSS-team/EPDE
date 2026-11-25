"""Generate the mirrored-neighbor pilot sheet and score preregistered focal gates."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ('absent', 'beside_j_plus_t', 'beside_j_minus_t')
MODEL_HASH = 'e7e3c77b9ec5c0cb09974317001ef11eb1b7cac811e53ae692fa2cc01859d871'
HORIZONS = (40, 80, 400)


def schedule():
    rng = np.random.default_rng(11487)
    rows = []
    for session in range(1, 5):
        settings = list(itertools.product((33, 64), ('++', '+-', '-+', '--')))
        for triplet, ix in enumerate(rng.permutation(len(settings)), 1):
            speed, spin = settings[ix]
            for order, arm in enumerate(rng.permutation(ARMS), 1):
                rows.append(dict(trial=f'S{session}-T{triplet}-{order}', session=session,
                                 triplet=triplet, order=order, closing_target_mm_s=speed,
                                 spin_i_j=spin, arm=arm))
    sheet = pd.DataFrame(rows)
    fields = {'score_dir': 'string', 'stage': 'Int64', 'i': 'Int64', 'j': 'Int64',
              'third_id': 'Int64', 'gate_frame': 'Int64', 'actual_A_t': 'Float64',
              'pwm': 'Float64', 'capture_notes': 'string'}
    for col, dtype in fields.items():
        sheet[col] = pd.Series(pd.NA, index=sheet.index, dtype=dtype)
    return sheet


def score(sheet_path, out):
    sheet = pd.read_csv(sheet_path, dtype={'spin_i_j': str})
    assigned = schedule()
    fixed = list(assigned.columns[:7])
    pd.testing.assert_frame_equal(sheet[fixed].sort_values('trial').reset_index(drop=True),
        assigned[fixed].sort_values('trial').reset_index(drop=True), check_dtype=False)
    rows = []
    seen = set()
    cache = {}
    for row in sheet.itertuples(index=False):
        entry = dict(trial=row.trial, session=row.session, triplet=row.triplet, arm=row.arm,
                     actual_A_t=row.actual_A_t)
        status = 'awaiting acquisition'
        if pd.notna(row.score_dir):
            directory = (sheet_path.parent / row.score_dir).resolve()
            if directory not in cache:
                receipt = json.loads((directory / 'receipt.json').read_text())
                if (receipt['model_sha256'] != MODEL_HASH or not receipt['implementation_matches_freeze']
                        or receipt['identity'] != 'no known input match' or receipt['prediction_unit'] != 'mm/s'):
                    raise ValueError(f'{directory} does not meet the independent frozen-score criteria')
                predictions = pd.read_parquet(directory / 'predictions.parquet') if receipt['events'] else pd.DataFrame()
                cache[directory] = receipt['input_sha256'], predictions
            digest, data = cache[directory]
            status = 'no annotated gate'
            if pd.notna(row.gate_frame):
                fields = [row.stage, row.gate_frame, row.i, row.j]
                if not all(pd.notna(v) and float(v).is_integer() for v in fields) or row.i == row.j:
                    raise ValueError(f'Invalid focal identifiers, stage or frame for {row.trial}')
                key = (digest, int(row.stage), int(row.gate_frame), *sorted((int(row.i), int(row.j))))
                if key in seen:
                    raise ValueError(f'Duplicate focal gate for {row.trial}')
                seen.add(key)
                status = 'annotated gate unscorable'
                if not data.empty:
                    gate = data[(data.stage == row.stage) & (data.frame == row.gate_frame)
                                & (data.i == row.i) & (data.j == row.j) & (data.axis == 't')]
                    if not gate.empty:
                        if sorted(gate.horizon_ms) != list(HORIZONS):
                            raise ValueError(f'Incomplete or repeated horizons for {row.trial}')
                        status = 'scored'
                        for obs in gate.itertuples():
                            for model in ('observed', 'compact', 'full'):
                                entry[f'{model}_{obs.horizon_ms}'] = getattr(obs, model)
        rows.append(dict(**entry, status=status))
    audit = pd.DataFrame(rows)
    contrasts = []
    for (session, triplet), entries in audit.groupby(['session', 'triplet']):
        if not entries.status.eq('scored').all():
            continue
        arms = entries.set_index('arm')
        for ms, model in itertools.product(HORIZONS, ('observed', 'compact', 'full')):
            absent, at_plus, at_minus = arms.loc[list(ARMS), f'{model}_{ms}']
            contrasts.append(dict(session=session, triplet=triplet, horizon_ms=ms, model=model,
                mirror_mm_s=at_minus-at_plus, midpoint_minus_absent_mm_s=(at_plus+at_minus)/2-absent))
    effects = pd.DataFrame(contrasts, columns=['session','triplet','horizon_ms','model','mirror_mm_s',
                                              'midpoint_minus_absent_mm_s'])
    session_effects = effects.groupby(['session','horizon_ms','model']).agg(
        triplets=('triplet','size'), mirror_mm_s=('mirror_mm_s','mean'),
        midpoint_minus_absent_mm_s=('midpoint_minus_absent_mm_s','mean')).reset_index()
    overall = session_effects.groupby(['horizon_ms','model']).agg(
        sessions=('session','size'), triplets=('triplets','sum'), mirror_mm_s=('mirror_mm_s','mean'),
        midpoint_minus_absent_mm_s=('midpoint_minus_absent_mm_s','mean')).reset_index()
    out.mkdir(parents=True, exist_ok=False)
    audit.to_csv(out/'trial_audit.csv', index=False)
    effects.to_csv(out/'triplet_contrasts.csv', index=False)
    session_effects.to_csv(out/'session_contrasts.csv', index=False)
    overall.to_csv(out/'equal_session_contrasts.csv', index=False)
    print(f'{len(sheet)} assigned trials, {audit.status.eq("scored").sum()} scored.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('make')
    make.add_argument('sheet', type=Path)
    analyze = commands.add_parser('score')
    analyze.add_argument('sheet', type=Path)
    analyze.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'make':
        schedule().to_csv(args.sheet, mode='x', index=False)
    else:
        score(args.sheet, args.output)
