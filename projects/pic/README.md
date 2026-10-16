# Data sets, runs and benchmark for EPDE

This project collects the data sets of the repository in one place and gives one way
to work with all of them: load a record, check that its known law is visible in the
data, run the search, score and inspect the result, compare methods. It works the same
on Windows, Linux and macOS, from the command line.

What it consists of:

- **a collection of 29 records**: synthetic ordinary and partial differential
  equations with known laws and measured records (pendulums, a robot arm, a ball on a
  beam, sea surface temperature). Each record comes with its grid, its known law and
  the equivalent forms of that law that also count as correct;
- **layered settings**: one shared protocol with a single broad token pool for every
  problem, a short file per record with only what that problem needs, method variants
  for comparisons, and changes given at run time;
- **a command line and short scripts per data folder**, all on top
  of the same code;
- **a benchmark**: series of runs in parallel processes that resume after an
  interruption, with reports and confidence intervals.

The library itself and the group's original research scripts are not changed; all of
this lives next to them.

## Installation

### With pip

Python 3.11–3.13, from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch                       # or a CUDA build from pytorch.org
pip install -e .
pip install -r requirements.txt         # required: the package declares no dependencies
pip install -r projects/pic/requirements-bench.txt
```

The commands always use EPDE from the working copy they belong to, even when another
copy is installed in the environment.

## Quick start

```bash
python projects/pic/bench.py list                    # every record
python projects/pic/bench.py info burgers            # data, known law, final settings
python projects/pic/bench.py check ac --noise 0,1,5  # is the law visible in the data
python projects/pic/bench.py run burgers --noise 1 --seed 0
python projects/pic/scripts/kdv.py --plot
```

The short scripts accept the same options as the command line (noise, seed, method
variant, changed settings, signal check, plots, output file) and can be started from
any folder.

## Settings

A run is configured in layers; a later layer wins:

1. the **shared protocol**: coordinates and sine/cosine of frequency 2 in every pool,
   powers of the variable up to 3 and of its derivatives up to 2, up to 10 terms of up
   to 2 factors, population 16, 5 epochs, finite differences — the setup of the group's
   former benchmark;
2. the **file of the record**: a larger population for systems, first derivatives for
   first-order systems, smoothing for measured data, special tokens;
3. a **method variant** compared by the benchmark;
4. **changes given at run time**.

Three values are filled for each record automatically: the boundary excluded from
fitting (10 % of every axis), the highest derivative orders (2 in time, 4 in space)
and the dimensionality of the token families.

## Data sets

The full list with classes, shapes, laws and notes is in [DATASETS.md](DATASETS.md):

- **core** — 15 synthetic problems with a known law, the main benchmark tables;
- **extended** — 3 more synthetic problems;
- **real** — 4 measured records;
- **other** — load, but are not benchmarked (no known law, missing files, or too large
  for a routine run).

A new record needs its files in the data folder, a loader that returns grids, fields
and (if known) the law in EPDE's text form, and a short settings file with what
differs from the shared protocol. The documentation and the script for the record are
then regenerated, and the signal check should confirm on clean data that the law holds
before any search is run.

## Signal check

The check evaluates the terms of the known law on the data, prepared exactly as the
search prepares them, and fits their coefficients by least squares. It reports how
much of the left-hand side the true structure explains and how much each term
contributes. A term that explains almost nothing cannot be identified under this
preprocessing. This is a diagnostic of the data, not a proof that no method can
recover the law.

```bash
python projects/pic/bench.py check pend_single --noise 0
python projects/pic/bench.py check ode --noise 0,1,5 --variant poly
```

Notes on particular records:

- **Turbulence slice:** the derivatives and external fields come from the direct
  numerical simulation. Artificial noise is not allowed: noisy fields would be compared
  with noise-free gradients.
- **Sea surface temperature:** by default the largest rectangular ocean region that is
  finite on every day is used; the original box with land masked can be loaded for
  plotting. The governing law is not known.
- **Darcy flow:** the data files are missing; the loader is kept as an unverified
  prototype.

Shapes, finiteness of the data and a non-empty region after the boundary are checked
before every search.

## Comparing methods

Besides EPDE, a sparse-regression baseline (PySINDy) runs on a library built from the
same fields, derivatives and extra tokens. Problems it cannot represent (a coefficient
in front of the time derivative, the continuity equation of Navier–Stokes) are reported
as unsupported, not as failures.

```bash
python projects/pic/bench.py campaign --datasets ode,lorenz \
    --variants default,pysindy --noise 0,1 --seeds 0-2 --workers 2 --name comparison
python projects/pic/bench.py report projects/pic/results/comparison
```

Every run of a campaign is a separate process. Repeating the command continues the same
campaign; failed runs can be retried. A run is identified by its settings, data, code
and dependency versions: changing any of them requires a new campaign, so old and new
results are never mixed.

The report shows the success rates with 95 % Wilson intervals, a ranking by problem
class on the problems every variant supports, and run times. Planned and unsupported
runs are listed separately; errors and timeouts count as failures.

## Checks

```bash
python -m unittest discover -s projects/pic/tests -v
```

Tests cover the available data, term parsing, metrics and run protocol. Checks that
need files missing from the repository are skipped and reported as such.

