"""Execute the tutorial notebooks with a named, fresh cache and durable evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

PIC = Path(__file__).resolve().parents[1]

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--notebooks', default='00,01,02,03,04,05,06,07,08,09,10')
    parser.add_argument('--campaign', default='baseline_clean_2026-10-09')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    evidence = PIC / 'results' / args.name
    evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'name': args.name, 'created_utc': now(), 'python': sys.executable,
        'cache': str(evidence / 'search_cache'), 'campaign': args.campaign, 'notebooks': {}}
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg',
                      EPDE_BENCH_NOTEBOOK_CACHE=manifest['cache'],
                      EPDE_BENCH_NOTEBOOK_CAMPAIGN=args.campaign)
    selected = set(args.notebooks.split(','))
    failures = 0
    with tempfile.TemporaryDirectory(prefix='epde-notebook-kernel-') as temporary:
        kernel_dir = Path(temporary) / 'epde-review'
        kernel_dir.mkdir()
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'EPDE review interpreter', 'language': 'python'}))
        for path in sorted((PIC / 'notebooks').glob('*.ipynb')):
            if path.name[:2] not in selected:
                continue
            notebook = nbformat.read(path, as_version=4)
            if path.name.startswith('10_'):
                notebook.cells = [c for c in notebook.cells if not (
                    c.cell_type == 'markdown' and c.source.startswith(
                        'The tables below are historical saved outputs'))]
                notebook.cells[0].source = """# 10 · Comparing methods on all records

**Question.** How often do EPDE and sparse regression recover the true equation or system across repeated seeds?

**Protocol.** The fresh clean-data campaign covers 15 core records, two methods and five seeds: 150 requested runs. Every run uses its dataset configuration. The shared evaluation compares methods only where both support the problem and reports unsupported records separately.

**Metrics.** Truth on the front means a discovered equation or system has the true set of terms under documented equivalent forms. The compromise equation is selected without the truth. Per-run recovery, recovery for at least one seed, equation recovery and whole-system recovery are separate quantities.

**Running.** The commands below reproduce this campaign. The displayed report is generated from its saved records during this notebook execution. Completed, failed, timed-out and unsupported runs remain visible in the report.
"""
            if notebook.nbformat_minor < 5:
                for cell in notebook.cells:
                    cell.pop('id', None)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
                    cell.metadata.pop('execution', None)
            record = {'started_utc': now(), 'status': 'running', 'cells': [],
                      'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest['notebooks'][path.name] = record
            def save():
                manifest_path.write_text(json.dumps(manifest, indent=2))
                nbformat.write(notebook, path)
            save()
            started = time.perf_counter()
            cell_started = {}
            with (evidence / (path.stem + '.log')).open('w', buffering=1) as log:
                def on_start(cell, cell_index, **kwargs):
                    cell_started[cell_index] = time.perf_counter()
                    record['active_cell'] = {'index': cell_index, 'started_utc': now()}
                    save()
                    log.write(f'CELL {cell_index} START {now()}\n{cell.source}\n')
                def on_complete(cell, cell_index, **kwargs):
                    # nbclient's completion hook precedes output collection; timings
                    # and outputs are saved by the executed hook below.
                    pass
                def on_executed(cell, cell_index, **kwargs):
                    elapsed = time.perf_counter() - cell_started[cell_index]
                    errors = [dict(name=o.ename, value=o.evalue, traceback=o.traceback)
                              for o in cell.outputs if o.output_type == 'error']
                    record['cells'].append({'index': cell_index, 'seconds': elapsed,
                                            'execution_count': cell.execution_count,
                                            'errors': errors})
                    for output in cell.outputs:
                        log.write(json.dumps(output, default=str) + '\n')
                    log.write(f'CELL {cell_index} END {elapsed:.3f}s\n')
                    save()
                manager = KernelManager(kernel_name='epde-review',
                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
                client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                    resources={'metadata': {'path': str(path.parent)}},
                    on_cell_start=on_start, on_cell_executed=on_executed,
                    record_timing=True, allow_errors=False)
                print(f'START {path.name} {now()}', flush=True)
                try:
                    client.execute(cleanup_kc=True)
                    record['status'] = 'ok'
                except Exception as exc:
                    record.update(status='error', error=f'{type(exc).__name__}: {exc}',
                                  traceback=traceback.format_exc())
                    log.write(record['traceback'])
                    failures += 1
                finally:
                    record.pop('active_cell', None)
                    record.update(seconds=time.perf_counter() - started, finished_utc=now())
                    notebook.metadata['review_execution'] = {
                        'name': args.name, 'started_utc': record['started_utc'],
                        'finished_utc': record['finished_utc'], 'status': record['status'],
                        'python': sys.executable, 'cache': manifest['cache']}
                    save()
                    print(f"END {path.name} {record['status']} {record['seconds']:.1f}s", flush=True)
    return int(failures > 0)

if __name__ == '__main__':
    raise SystemExit(main())
