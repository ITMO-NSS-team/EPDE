"""Print the concrete lorenz overfit equations (full text) with their CV-ratio,
to make 'what are overfits' concrete. An overfit = drives IN-SAMPLE residual
low (often below truth) but its CV-WAPE blows up vs truth -> it fits FD-error /
noise / time-coordinate structure with spurious terms, not the physics."""
import os, sys, json
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)
from term_contribution import build_pool
from analyze_pareto import _clean_lines
from lorenz_dominator_check import _design, _insample_wape, _cv_wape
from thesis_runner import _load_yaml, CONFIGS_DIR

_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_THIS, 'results', 'lorenz', 'new_rep00.json')
rec = json.load(open(_PATH, encoding='utf-8'))
truth_eqs = _load_yaml(os.path.join(CONFIGS_DIR, 'lorenz.yaml')).get('truth_equations')
cfg, search, all_vars = build_pool('lorenz')
n_eq = len(truth_eqs)
truth_cv = {}
for k in range(n_eq):
    Xt, yt, _ = _design(truth_eqs[k], search.pool, all_vars)
    truth_cv[all_vars[k]] = _cv_wape(Xt, yt)

# collect all distinct equation forms per var with their CV
seen = {v: {} for v in all_vars}
for s in rec.get('discovered_text_per_solution') or []:
    lines = _clean_lines(s)
    for k in range(min(n_eq, len(lines))):
        var = all_vars[k]
        if lines[k] in seen[var]:
            continue
        try:
            X, y, _ = _design(lines[k], search.pool, all_vars)
            seen[var][lines[k]] = (_insample_wape(X, y), _cv_wape(X, y))
        except Exception:
            pass

for k in range(n_eq):
    var = all_vars[k]
    print(f"\n{'='*100}\n### {var}   (truth CV={truth_cv[var]:.2e};  truth: {truth_eqs[k]})")
    worst = sorted(seen[var].items(), key=lambda kv: -kv[1][1])[:3]
    for line, (inw, cvw) in worst:
        print(f"\n  CV={cvw:.2e} ({cvw/truth_cv[var]:.0f}x truth)  in-sample={inw:.2e}")
        print(f"    {line}")
