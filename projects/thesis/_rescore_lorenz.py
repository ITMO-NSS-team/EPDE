"""Re-score all lorenz `new` reps against the (expanded) truth_alternatives:
verifies the new YAML alt parses + shows whether it credits any prior failure."""
import os, sys, json, glob
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)
from thesis_runner import load_config
from thesis_metrics import canonical_tokens, structural_success_any, hamming_best

cfg = load_config('lorenz')
truth_alts = (cfg.truth_tokens,) + tuple(cfg.truth_alternatives)
print(f"truth_alternatives (incl. primary): {len(truth_alts)} forms")

changed = 0
for f in sorted(glob.glob(os.path.join(_THIS, 'results', 'lorenz', 'new_rep*.json'))):
    rec = json.load(open(f, encoding='utf-8'))
    sols = rec.get('discovered_text_per_solution') or []
    canons = []
    for s in sols:
        try:
            canons.append(canonical_tokens(s))
        except Exception:
            pass
    succ = any(structural_success_any(c, truth_alts) for c in canons)
    best_ham = min((hamming_best(c, truth_alts) for c in canons), default=None)
    o_succ, o_ham = rec.get('structural_success'), rec.get('hamming')
    flag = ''
    if succ != o_succ or (best_ham is not None and o_ham is not None and best_ham < o_ham):
        flag = '  <-- CHANGED'; changed += 1
    print(f"  {os.path.basename(f):>16}: success {o_succ}->{succ}  hamming {o_ham}->{best_ham}{flag}")
print(f"\n{changed} reps changed by the expanded credit list.")
