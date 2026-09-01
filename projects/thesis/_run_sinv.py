"""Run lorenz with ScaleInvariantDiscrepancy as the primary metric (otherwise
identical to the 'new' pipeline: VWSRSparsity + use_pic). Non-invasive: injects
a 'new_sinv' pipeline into the runtime _PIPELINE_SETTINGS and reuses run_smoke,
so the result JSON is analyze_pareto-compatible. Seed 0 (matches the WAPE run)."""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import thesis_runner as tr

tr._PIPELINE_SETTINGS['new_sinv'] = {
    'discrepancy_metric': 'scale_invariant',
    'sparsity_cls': tr.VWSRSparsity,
    'use_pic': True,
}

cfg = tr.load_config('lorenz')
tr.run_smoke(cfg, reps=1, pipelines=('new_sinv',), seed_base=0, resume=False, outdir='sinv_test')
