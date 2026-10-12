# Clean-data baseline: EPDE and PySINDy

Core suite (15 records) × two methods (`default` = EPDE as shipped, `pysindy`) ×
seeds 0–4, noise 0 %, 900 s per run: 150 runs. Every run used its record's
configuration and the shared protocol; no setting was tuned after seeing results.
CPU, Python 3.13, BLAS threads 1, two runs in parallel on a 16 GB machine.

```bash
python projects/pic/bench.py campaign --suite core --variants default,pysindy \
       --noise 0 --seeds 0-4 --workers 2 --timeout 900 --name baseline_clean_2026-10-09
python projects/pic/bench.py report projects/pic/results/baseline_clean_2026-10-09
```

Outcome: 130 `ok`, 10 `timeout` (EPDE on Kuramoto–Sivashinsky and Navier–Stokes),
10 `unsupported` (PySINDy cannot represent `ns` and `pde_divide`). Timeouts count
as misses; unsupported runs are excluded from denominators.

| record | EPDE: truth on front | EPDE: pick | PySINDy: truth on front | PySINDy: pick |
|---|---|---|---|---|
| ode | 4/5 | 3/5 | 5/5 | 5/5 |
| vdp | 5/5 | 5/5 | 5/5 | 5/5 |
| duffing | 5/5 | 5/5 | 5/5 | 5/5 |
| lv | 2/5 | 2/5 | 5/5 | 5/5 |
| lorenz | 0/5 | 0/5 | 5/5 | 5/5 |
| ac | 4/5 | 0/5 | 0/5 | 0/5 |
| burgers | 5/5 | 3/5 | 5/5 | 5/5 |
| burgers_inviscid | 5/5 | 5/5 | 5/5 | 5/5 |
| kdv | 2/5 | 2/5 | 0/5 | 0/5 |
| kdv_cossin | 5/5 | 5/5 | 5/5 | 5/5 |
| ks | 0/5 (timeout) | 0/5 | 0/5 | 0/5 |
| wave | 5/5 | 5/5 | 5/5 | 5/5 |
| pde_compound | 5/5 | 5/5 | 5/5 | 5/5 |
| pde_divide | 5/5 | 5/5 | unsupported | unsupported |
| ns | 0/5 (timeout) | 0/5 | unsupported | unsupported |

"Truth on front": the true set of terms (or a documented equivalent form) is on the
final Pareto front. "Pick": the equation chosen from the front without the truth is
correct. Coefficient errors, Wilson intervals, timings and the class ranking are in
`report/summary.md`; `report/runs.csv` has one row per run.

Notable results: on Allen–Cahn the true structure is on EPDE's front in 4 of 5 runs
but the compromise pick never selects it; EPDE does not recover the Lorenz system
within the shipped budget, while PySINDy does in every run; on KdV EPDE recovers the
structure in 2 of 5 runs and PySINDy in none, while on the sine–cosine KdV record both
succeed in every run.

## Files

- `campaign.json` — the requested grid, the identity of every planned run and the environment.
- `runs/<record>/<variant>__noise0__seed<s>.json` — one raw record per run: discovered
  equations, the Pareto front, metrics, settings and the run identity (settings, job,
  content hash of the computational code, input data and dependency versions).
- `report/` — tables and figures generated from the records by `bench.py report`.

## Execution note

All runs were computed from one frozen copy of the computational code. The original
campaign process could not stop timed-out Kuramoto–Sivashinsky runs in its sandboxed
environment (listing child processes was refused), so those runs kept computing past
the limit. That controller was stopped; the 21 runs without a record (five EPDE runs on
`ks`, six on `pde_divide`, ten on `ns`) were then executed by the same job runner from the same
frozen code, with a kill fallback that also works without access to the process table.
Every one of the 150 records was checked against the planned identity in
`campaign.json`. The fallback is now part of `epde_bench/campaign.py`.

The campaign ran before EPDE's per-offspring debug output (`runtime.verbose_params.candidate_objectives`)
was switched off in `configs/_base.yaml`. That setting only changes printed output, but it is part of the
recorded configuration, so rerunning the command above now produces runs with a different identity
(and needs a new campaign name); the search results themselves are unaffected.
