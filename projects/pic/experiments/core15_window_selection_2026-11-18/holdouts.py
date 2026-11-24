from concurrent.futures import ProcessPoolExecutor, as_completed
from setup_study import ROOT, SYSTEMS, SEEDS, records


def build(job):
    records.ROOT = ROOT
    if not (ROOT / "records" / f"{job[0]}_{job[1]}.json").exists():
        records.build(job)
    return job


if __name__ == "__main__":
    (ROOT / "records").mkdir(exist_ok=True)
    jobs = [(s, seed) for seed in SEEDS for s in SYSTEMS]
    with ProcessPoolExecutor(max_workers=9) as pool:
        for f in as_completed([pool.submit(build, job) for job in jobs]):
            print(f.result(), flush=True)
