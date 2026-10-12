# Chronological EPDE discovery and forecast evidence

The raw scalar ODE record was split once: 224/320 samples (70%) for training,
96/320 samples (30%) for evaluation. Seed 0, shipped population 16 / epochs 5,
FD derivatives of order 2, CPU, BLAS threads 1. The training Problem was built
before any differentiation or search and had its truth equation removed.

The Pareto front and its coefficients were obtained from actual EPDE search.
Selection used minimum normalized highest-derivative MSE on the training
interior. The selected equation includes a small additional squared-velocity
term; this is a discovered equation, not the supplied true law. The coefficients
were frozen before integrating the entire held-out interval. Initial value at
t=11.15 is the last training observation; initial velocity is a one-sided
fourth-degree local polynomial estimate from the last nine training samples.

Training metrics are fitted-record reconstruction on the boundary-cropped
training interior (180 samples, t=1.1..10.05), not an independent validation.
Held-out metrics use all 96 future samples, t=11.2..15.95. No future observations
enter the integration. Known analytic forcing (time and sin(2t)) is evaluated
at future times.

`record.json`, `trajectory.png`, and `trajectory.npz` contain the primary run.
`adversarial_changed_test/` contains a second real EPDE run with identical seed
and configuration but all future u values replaced by 10000+100*arange(96).
`leakage_audit.json` verifies the training hash, full Pareto coefficients,
selected equation, training selection scores, and forecast values are identical.
Only the final test evaluation changes. `run.log` records both real runs.

Primary held-out R2 = 0.9998324293217595, RMSE = 0.01992835372894183,
MAE = 0.014939868267107281. Integration success=true and coverage=1.0.
This result concerns a clean synthetic oscillator, one seed and one budget.
It does not establish robust real-data forecasting or uncertainty calibration.

Reproduce from EPDE_ready with PYTHONPATH=projects/pic and the bundled
projects/pic/.venv/bin/python, setting OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, MPLBACKEND=Agg and a writable MPLCONFIGDIR:

```python
from epde_bench.heldout import run_heldout
record = run_heldout('ode', train_fraction=0.7, seed=0, output_dir='path/to/output')
```
