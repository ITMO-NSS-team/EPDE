"""Prospective repeat of the three-arm live wave screen on six further seeds."""
from concurrent.futures import ThreadPoolExecutor

SEEDS = tuple(range(917401,917407))

if __name__=='__main__':
    from batch import launch
    with ThreadPoolExecutor(max_workers=6) as workers:
        codes=list(workers.map(launch,[('wave',n,a) for n in SEEDS for a in ('wape','wls','tls')]))
    raise SystemExit(int(any(codes)))
