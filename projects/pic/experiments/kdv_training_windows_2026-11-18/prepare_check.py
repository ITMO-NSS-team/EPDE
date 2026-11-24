"""Build fresh paired measurements for the isolated forcing-projection comparison."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from holdouts import ROOT, build
from training import train

DEST = ROOT / "projection_check"


def prepare(job):
    system, seed = job
    if system == "kdv_cossin":
        build(seed, DEST)
    return train(job, DEST)


if __name__ == "__main__":
    for name in ("records", "training"):
        (DEST / name).mkdir(exist_ok=True)
    jobs = [(system, start + i) for i in range(24)
            for system, start in (("kdv_cossin", 790001), ("kdv", 791001))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(prepare, job) for job in jobs]):
            print(f.result(), flush=True)
