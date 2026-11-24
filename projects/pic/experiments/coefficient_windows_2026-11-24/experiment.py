"""Replay the archived screen and regenerate native training plans on fresh noise."""
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'complementary_windows_2026-11-24'
sys.path.insert(0, str(PREVIOUS))
from cpu_limit import cap_worker

if __name__ == '__main__':
    cap_worker()

from run import SYSTEMS, load_search_config, run, set_active_config
from screen import configured_width, training_plan
from uncertainty import select

REPLAY = range(923211, 923217)
FRESH = range(926301, 926307)


def main():
    cfg = load_search_config(overrides=dict(evaluation_form='weak', system_rps_mode='plain',
                                            instability_metric='dyad', equation_terms_max_number=7))
    set_active_config(cfg)
    width = configured_width(cfg.search_space)
    dest = ROOT/'results'
    dest.mkdir(exist_ok=True)
    for system in ['kdv_cossin', *(s for s in SYSTEMS if s != 'kdv_cossin')]:
        for seed in (*REPLAY, *FRESH):
            path = dest/f'{system}_{seed}.json'
            if path.exists():
                continue
            elapsed = []

            def choose(train, axes):
                start = process_time()
                plan = training_plan(train, axes, width)
                elapsed.append(process_time()-start)
                return plan

            fresh = seed in FRESH
            case = run(system, seed, select_geometry=select, include_truth=True,
                       panel_seed=REPLAY.start if fresh else seed, choose_plan=choose if fresh else None)
            case['native_plan_cpu_seconds'] = elapsed[0] if fresh else None
            tmp = path.with_suffix('.pending')
            tmp.write_text(json.dumps(case, indent=1, allow_nan=False), encoding='utf8')
            tmp.replace(path)
        print(f'{system} complete', flush=True)


if __name__ == '__main__':
    main()
