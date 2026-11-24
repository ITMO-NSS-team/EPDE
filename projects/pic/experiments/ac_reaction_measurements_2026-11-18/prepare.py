"""Freeze the ordinary training matrices and unchanged held-out measurements."""
from concurrent.futures import ProcessPoolExecutor
from common import ROOT, SEEDS, records, train


def prepare(seed,directory=ROOT):
    records.ROOT=train.ROOT=directory
    for tag,method in (("records",records.build),("training",train.run)):
        if not (directory/tag/f"ac_{seed}.json").exists():
            method(("ac",seed))
    return seed


if __name__=="__main__":
    for tag in ("records","training"):
        (ROOT/tag).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=7) as pool:
        for seed in pool.map(prepare,SEEDS):
            print(seed,flush=True)
