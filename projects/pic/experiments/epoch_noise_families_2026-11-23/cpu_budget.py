"""Reserve a quarter of the host's logical CPUs across the two full-record experiments."""
import argparse
import os
from pathlib import Path
import time

import psutil
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'epoch_full_record_2026-11-23'
SCRIPTS = {ROOT / n for n in ('campaign.py', 'noise_worker.py', 'noise_results.py')}
SCRIPTS.update(BASELINE / n for n in ('run.py', 'patience.py', 'score.py', 'report.py'))
LOGICAL = psutil.cpu_count() or 1
CPUS = tuple(range(max(1, LOGICAL * 3 // 4)))


def restrict(process):
    current = process.cpu_affinity()
    allowed = sorted(set(current).intersection(CPUS))
    if not allowed:
        raise RuntimeError(f'PID {process.pid} has no CPUs in the shared pool {CPUS}')
    if sorted(current) != allowed:
        process.cpu_affinity(allowed)
    if os.name == 'nt' and process.nice() == psutil.NORMAL_PRIORITY_CLASS:
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    return process.cpu_affinity()


def experiment_script(process):
    args = process.cmdline()[1:]
    for arg in args:
        if arg in ('-m', '-c'):
            return None
        if arg.startswith('-'):
            continue
        script = Path(arg)
        if script.name not in {p.name for p in SCRIPTS}:
            return None
        if not script.is_absolute():
            script = Path(process.cwd()) / script
        return script if script in SCRIPTS else None
    return None


def apply_running():
    matched = 0
    for process in psutil.process_iter(['name']):
        if (process.info['name'] or '').lower() not in ('python', 'python.exe', 'pythonw.exe'):
            continue
        try:
            script = experiment_script(process)
            if script is not None:
                cpus = restrict(process)
                print(process.pid, script.parent.name, script.name, cpus, flush=True)
                matched += 1
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as error:
            print(f'Permission denied inspecting or limiting Python PID {process.pid}: {error}', flush=True)
    return matched


def await_capacity():
    # A new single-thread search needs roughly one logical CPU of host-wide headroom.
    restrict(psutil.Process())
    threshold = 100 * max(1, len(CPUS) - 1) / LOGICAL
    with FileLock(ROOT / 'admission.lock'):
        load = psutil.cpu_percent(interval=3)
        if load > threshold:
            print(f'Waiting for CPU headroom: host {load:.1f}%, admission threshold {threshold:.1f}%.', flush=True)
        while load > threshold:
            time.sleep(9)
            load = psutil.cpu_percent(interval=3)
        print(f'Admitted at {load:.1f}% host CPU, shared pool {list(CPUS)}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    count = apply_running()
    print(f'{count} matching processes; shared CPUs {list(CPUS)} of {LOGICAL}', flush=True)
