"""Build frozen forced-field validation records using the existing full panel."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "absent_term_controls_2026-11-18"
sys.path.insert(0, str(PREVIOUS))
from controls import E, TrainingMoments, corrected_fit, np, records, score


def build(seed, directory=ROOT):
    records.ROOT = directory
    path = directory / "records" / f"kdv_cossin_{seed}.json"
    if not path.exists():
        records.build(("kdv_cossin", seed))
    return seed


if __name__ == "__main__":
    (ROOT / "records").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed in pool.map(build, range(788001, 788025)):
            print(seed, flush=True)
