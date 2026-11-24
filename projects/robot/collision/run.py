"""Run or resume the five recording-level transfer folds. See results/collision_transfer/report.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'

from physics import OUT, SEEDS, swarm_files, swarm_meta


def jobs(stage, tags):
    if stage == 'search':
        return [(stage, tag, form, str(seed)) for seed in SEEDS for form in ('strong','weak') for tag in tags]
    return [(stage, tag) for tag in tags]


def receipt(job):
    stage, tag, *rest = job
    name = '_'.join(rest) if stage == 'search' else {'prepare':'training','assess':'assessment','events':'impacts'}[stage]
    return OUT/tag/(name+'.json')


def check_inputs():
    """A resumed campaign must refer to the same five instrument records."""
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'inputs.json'
    with FileLock(str(path)+'.lock'):
        hashes={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in swarm_files()}
        if path.exists():
            if json.loads(path.read_text())!=hashes:
                raise ValueError('Recording hashes changed. Set EPDE_COLLISION_OUTPUT to a new result directory.')
        else:
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(hashes,indent=1),encoding='utf-8')
            tmp.replace(path)


def launch(job):
    path = receipt(job)
    from filelock import FileLock
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        if path.exists() and (job[0]!='prepare' or (path.parent/'prepared.npz').exists()):
            saved = json.loads(path.read_text())
            if job[0]!='events' or saved.get('physical_clearance_5',{}).get('distance_precision')=='float64':
                return
        print('Starting', *job, flush=True)
        log = path.with_suffix('.log')
        with log.open('w', encoding='utf-8') as fh:
            code = subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',*job],
                                   stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONHASHSEED='0'))
        if code or not path.exists():
            raise RuntimeError(f'{job} failed; {log}\n'+log.read_text(errors='replace')[-1800:])
        print('Finished', *job, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all','prepare','search','events','assess','rebound','early','pareto','expanded','projected','pair-normal','report','verify'], default='all')
    p.add_argument('--tag', action='append', choices=[swarm_meta(f)['tag'] for f in swarm_files()])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--worker', nargs='+', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.worker:
        stage, tag, *extra = args.worker
        if stage == 'prepare':
            from discovery import prepare
            prepare(tag)
        elif stage == 'search':
            from discovery import native_search
            native_search(tag, extra[0], int(extra[1]))
        elif stage == 'events':
            from impacts import analyze
            analyze(tag)
        elif stage == 'assess':
            from assessment import assess
            assess(tag)
        return
    all_tags = [swarm_meta(f)['tag'] for f in swarm_files()]
    tags = list(dict.fromkeys(args.tag or all_tags))
    check_inputs()
    if args.stage=='pair-normal':
        if set(tags)!=set(all_tags):
            p.error('The pair-normal search uses all five recording folds.')
        prior = OUT/'preprocessed'/'search'
        if not all((prior/name).is_file() for name in ['comparison_by_recording.csv','summary.json',
                       'cross_recording/projected_by_recording.csv.gz','cross_recording/pooled_common_law.json']):
            subprocess.run([sys.executable,__file__,'--stage','projected','--workers',str(args.workers)],check=True)
        from pair_normal_search import run as normal_search
        normal_search(args.workers)
        return
    if args.stage in ('expanded','projected'):
        if set(tags)!=set(all_tags):
            p.error('The merged expanded search requires all five folds.')
        os.environ['EPDE_COLLISION_GEOMETRY'] = 'projected' if args.stage=='projected' else 'raw'
        from expanded_data import prepare as build_expanded
        from expanded_native import launch as search_factors
        from expanded_search import run as score_expanded
        if args.stage == 'projected':
            from projected_report import run as report_expanded
            if not all((OUT/'preprocessed'/f'{tag}.npz').exists() for tag in tags):
                from preprocess_tracks import run as preprocess
                preprocess()
            raw = OUT/'expanded_factors'
            if not (raw/'validation_selected.csv').exists() or len(list((raw/'statistics').glob('*.npz')))!=45:
                subprocess.run([sys.executable,__file__,'--stage','expanded','--workers',str(args.workers)],check=True)
        else:
            from expanded_report import run as report_expanded
        for tag in tags:
            if not (OUT/tag/'training.json').exists():
                launch(('prepare',tag))
        build_expanded()
        for library in ('focused','broad'):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda tag:search_factors(tag,library), tags))
        score_expanded(True)
        report_expanded()
        return
    stages = ['verify','prepare','search','events','assess','rebound','early','pareto','report'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage == 'verify':
            from checks import verify, legacy, verify_outputs
            verify()
            if not (OUT/'legacy.json').exists():
                legacy()
            if (OUT/'report.md').exists():
                verify_outputs()
        elif stage in ('rebound', 'early', 'pareto', 'report'):
            if args.stage=='all' and set(tags)!=set(all_tags) and not all(
                    receipt(('assess',tag)).exists() for tag in all_tags):
                if stage == 'report':
                    print('Selected folds completed. The combined report uses all five folds.')
                continue
            if stage in ('rebound', 'early'):
                from contact_rebound import score_rebound
                score_rebound(early=stage == 'early')
                continue
            if stage == 'pareto':
                from pareto import run as evaluate_fronts
                evaluate_fronts()
                continue
            from report import report
            report()
            from checks import verify_outputs
            verify_outputs()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(launch, jobs(stage,tags)))


if __name__ == '__main__':
    main()
