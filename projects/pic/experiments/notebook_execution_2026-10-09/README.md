# Notebook execution record

All eleven notebooks were executed from a clean state on 9 October 2026 with
`scripts/execute_notebooks.py --name notebook_execution_2026-10-09`: saved outputs
were cleared, a new empty search cache was used, and no earlier search result was
reused. CPU, Python 3.13, BLAS threads 1.

`manifest.json` records, per notebook, the status, the source hash, start and end
times and the duration and status of every cell. All 135 code cells completed without
errors. The execution followed the campaign in `../baseline_clean_2026-10-09`, which
notebooks 08 and 10 read; they start no searches themselves. EPDE's per-offspring
debug output (`candidate_objectives`) is switched off in the shared protocol, so the
outputs show the search results without it; the results are identical with it on.

Executing a notebook is not the same as recovering an equation: the outputs state,
for every search, whether the discovered structure matches the known law.
