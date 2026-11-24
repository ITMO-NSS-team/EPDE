"""Drain audit work left by the live watcher's early process-complete exit.

The watcher can see processes_fast.json on the same iteration that its one
pending record finishes, before it has enqueued every remaining result. Keep
its original report and audit any remaining records without changing results.
"""
from bridge import *
import audit_verified as A
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    process = read(ROOT / 'processes_fast.json')
    assert len(process['rows']) == 69
    initial = ROOT / 'audit_live.json'
    if not initial.exists():
        initial.write_bytes((ROOT / 'audit.json').read_bytes())
    paths = sorted((ROOT / 'results').glob('*.json'))
    assert len(paths) == 69
    missing = []
    for path in paths:
        record = ROOT / 'audit_records' / path.name
        if not record.exists():
            missing.append(path)
        else:
            d = read(record)
            if d['status'] != 'PASS':
                preserved = ROOT / 'audit_initial_records' / record.name
                preserved.parent.mkdir(exist_ok=True)
                if not preserved.exists():preserved.write_bytes(record.read_bytes())
                missing.append(path)
            else:
                assert d['result_sha256'] == sha(path)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(A.safe, str(p)) for p in missing]):
            row = future.result()
            print(row['record'], row['status'], flush=True)
    records = sorted((ROOT / 'audit_records').glob('*.json'))
    rows = [read(p) for p in records]
    keys = ('operator_scores', 'regions', 'certificates', 'matrix_replays',
            'refit_replays', 'field_hashes', 'hash_links')
    out = dict(status='PASS' if len(rows) == 69 and all(r['status'] == 'PASS' for r in rows) else 'FAIL',
               records=len(rows), expected=69,
               totals={k: sum(r.get(k, 0) for r in rows) for k in keys}, rows=rows,
               source_sha256=sources('finish_audit.py', 'audit_verified.py', 'audit_full_fast.py', 'full_pipeline_fast.py',
                                     'pilot_roundoff_protocol.txt'),
               input_sha256={str(p): sha(p) for p in [initial, ROOT / 'processes_fast.json', *records]},
               orchestration_note='Preserved live watcher report and failed records; drained remaining results and independently checked ill-conditioned ellipsoids at 80 digits before aggregation.')
    dump(ROOT / 'audit.json', out)
    print(out['status'], len(rows), out['totals'], flush=True)
    assert out['status'] == 'PASS'


if __name__ == '__main__':
    main()
