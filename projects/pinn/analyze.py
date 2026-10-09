"""Gate tables, integrity checks and paired contrasts for gate.py results.

usage
  python analyze.py [--tags baseline,stage1_start,stage1,stage1_end,stage1_basis]
                    [--systems ac,duffing,burgers]

Per system:
  1. INTEGRITY -- nothing below is read until these hold:
     * duplicate controls bitwise equal to their originals;
     * the pre-patch baseline B equals the stored Sep 18 2026 values (AC, Duffing);
     * B-ctl run at the START / END of the Stage-1 batch equals the pre-patch B
       (the patch left the default path alone / nothing drifted in the batch);
     * null arms (``null_of``) equal their reference arm bitwise;
     * paired arms start from the same initial weights as S0 (pairing digest);
     * trace replay at the shipped cap equals the plain solve.
  2. GATE TABLE -- per-seed gate = ln(RMSE_wrong / RMSE_true), min over seeds, the
     bar (> 0 on EVERY seed). Trace arms are also read at the WALL-CLOCK budget
     T_B (median seconds of the plain shipped-recipe solves on that system) and
     at 0.9 T_B; a gate read where the arm had already hit its iteration cap
     before the budget is flagged as censored.
  3. CONTRASTS -- pre-specified paired contrasts of the per-seed gate
     (pinn_common.contrast; unresolved rows print mde and seeds_needed).
Then, across systems, the pre-registered ranking: eligible = every solve finite
and the bar passed at T_B and 0.9 T_B on every system; score = the minimum gate
over (system, seed) at T_B, in nats; arms within 0.10 of the top tie.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "pic", "data"))    # pinn_common
sys.path.insert(0, HERE)
from gate import (ARMS, RESULTS, SYSTEM_VARS, inert_reference, load_rows,  # noqa: E402
                  system_forms)

DEFAULT_TAGS = ("baseline,stage1_start,stage1,stage1_end,stage1_basis,"
                "stage1b_start,stage1b,stage1b_end,stage1b_basis")
#: Stage 2 refits candidates WITHOUT a free coefficient unless the form writes
#: one (row 'intercept_rule' = 'from_form'); every earlier tag forced one. The
#: two are never analysed together (main() refuses): read Stage 2 with
#:   --tags stage2_start,stage2,stage2_end
#: and its patch-neutrality control with  --tags baseline,stage2_legacy
STAGE2_TAGS = "stage2_start,stage2,stage2_end"
#: Stages 0 / 1 / 1b recalculated under the no-bias rule (run_recalc_gpu.sh)
RECALC_TAGS = "r_baseline,r_start,r_stage1,r_stage1b,r_end"
#: tags whose B-ctl rows are kept apart from the pre-patch baseline's
CONTROL_TAGS = ("stage1_start", "stage1_end", "stage1b_start", "stage1b_end",
                "stage2_start", "stage2_end", "stage2_legacy", "r_start", "r_end")

# ------------------------------------------------------------ stored references
# Per-seed held-out RMSE of the shipped recipe (cap 2000), gtol_{ac,duffing}.json
# arm 'ctl', Sep 18 2026. Forms: 'true' = with diffusion / with cubic.
REFERENCE_RMSE = {
    "ac": {"true": {0: 0.0014048906505597996, 1: 0.0016836012437451405,
                    2: 0.002020395737798916},
           "wrong": {0: 0.0034917411228727302, 1: 0.0036876200661360714,
                     2: 0.003869409725283864}},
    "duffing": {"true": {0: 0.0001403844984162819, 1: 8.994542483356039e-05,
                         2: 0.00041050944472153866},
                "wrong": {0: 0.023633467029023573, 1: 0.028270871354000023,
                          2: 0.02955507780815881}},
}
# Per-seed gate at L-BFGS caps, replayed from the 15000-iteration traces
# (maxiter_sweep_fixed.json, exact-cap semantics).
REFERENCE_GATE = {
    "ac": {1000: [0.34177844055276413, 0.11315146035808642, -0.011254919757859059],
           2000: [0.9104410299120047, 0.7840461847884734, 0.6498085676152715],
           3000: [0.8801497974109909, 0.8589403332842205, 0.8162098563337415],
           4000: [0.872081784965507, 0.8883198224875736, 0.8288684067551125]},
    "duffing": {1000: [3.2562442943652194, 0.912899332457536, -0.2987693737950388],
                2000: [5.1260340055618805, 5.7503841771179, 4.2766119094357204],
                4000: [5.1260340055618805, 5.7503841771179, 4.2766119094357204]},
}

# ------------------------------------------------------------ pre-specified
BASELINE = "B"
# (name, arm A, arm B, budget): per-seed gate of A minus that of B, higher is
# better. budget 'own' = each arm at its own cap / plain solve; 'T_B' = the
# wall-clock budget (trace arms only; plain arms are read as solved).
CONTRASTS = [
    ("S0 - B    drop Adam (cold L-BFGS only)", "S0", "B", "T_B"),
    ("B-4k - B  L-BFGS cap 4000 vs 2000", "B-4k", "B", "own"),
    ("S1 - S0   affine inputs", "S1", "S0", "T_B"),
    ("S2 - S0   moment outputs", "S2", "S0", "T_B"),
    ("S3 - S0   per-solve data prefit", "S3", "S0", "T_B"),
    ("S4 - S0   lstsq last layer", "S4", "S0", "T_B"),
    ("S6 - S0   float64", "S6", "S0", "T_B"),
    ("S7 - S0   scale-inv loss, live", "S7", "S0", "T_B"),
    ("S8 - S0   scale-inv loss, data", "S8", "S0", "T_B"),
    ("S9 - S0   shared data fit", "S9", "S0", "T_B"),
    ("S9 - S3   converged shared fit vs short prefit", "S9", "S3", "T_B"),
    ("B9 - B    shipped recipe from the shared fit", "B9", "B", "own"),
    # Stage 1c / 1b (Sep 29)
    ("S1-cap - B  S1 capped at 2000 vs shipped", "S1-cap", "B", "own"),
    ("S10 - S1  shared fit on scaled inputs", "S10", "S1", "T_B"),
    ("S7f - S0  scale-inv live, floored", "S7f", "S0", "T_B"),
    ("S8f - S0  scale-inv data, floored", "S8f", "S0", "T_B"),
    ("P1 - B    DoG instead of Adam", "P1", "B", "own"),
    ("P2 - B    Prodigy instead of Adam", "P2", "B", "own"),
    ("D1 - B    Adam lr from prefit distance", "D1", "B", "own"),
    ("D2 - B    Adam lr selected on data", "D2", "B", "own"),
    ("S5-32 - S0-32  LM vs L-BFGS on [32]x4", "S5-32", "S0-32", "own"),
    # Stage 2 (on S1): each factor alone against G-0000 (== S1)
    ("G-m000 - G-0000  S1+S2 vs S1: moment outputs", "G-m000", "G-0000", "T_B"),
    ("G-m000 - G-0000  S1+S2 vs S1, own cap 4000", "G-m000", "G-0000", "own"),
    ("G-0p00 - G-0000  periodic embedding", "G-0p00", "G-0000", "T_B"),
    ("G-00g0 - G-0000  data-gradient collocation", "G-00g0", "G-0000", "T_B"),
    ("G-00f0 - G-0000  flat warp, f32 (placement null)", "G-00f0", "G-0000", "T_B"),
    ("G-000d - G-0000  float64", "G-000d", "G-0000", "T_B"),
    ("G-00fd - G-000d  flat warp, f64 (placement null)", "G-00fd", "G-000d", "T_B"),
    ("G-0000 - B  S1 route vs shipped", "G-0000", "B", "T_B"),
    # two-way interactions as a difference of paired differences, (A1-A2)-(B1-B2):
    # moments and float64 may share one cause (the float32 line-search stall);
    # the gradient warp widens the residual's dynamic range, where float32 stalls
    ("moments x float64", ("G-m00d", "G-000d"), ("G-m000", "G-0000"), "T_B"),
    ("gradient warp x float64", ("G-00gd", "G-000d"), ("G-00g0", "G-0000"), "T_B"),
    # Stage 2b: scale-invariant losses on S1, each vs G-0000 (== S1)
    ("L-os - G-0000  scale-inv data loss, live", "L-os", "G-0000", "T_B"),
    ("L-od - G-0000  scale-inv data loss, data mass", "L-od", "G-0000", "T_B"),
    ("L-ps - G-0000  scale-inv physics loss, live (on S1)", "L-ps", "G-0000", "T_B"),
    ("L-pos - G-0000  both scale-invariant", "L-pos", "G-0000", "T_B"),
    ("L-pos - L-ps  the data term's share", "L-pos", "L-ps", "T_B"),
]
REPLAY_CAPS = [1000, 2000, 3000, 4000]
TIE = 0.10
#: the fitness host's failure value (LOSS_NAN_VAL): a failed solve, not an error
#: level. ln(1e7 / 1e7) = 0 once read as a "gate" of +0.000 (S4 on AC).
FAILED_RMSE = 1e6

#: Rows whose wall-clock was inflated ~1.3-1.4x by a CPU basis sweep running
#: alongside the GPU batch on 2026-09-28 (~14:05-15:23). Found by the Stage-1
#: verification (L-BFGS speed 70-76 ms/iteration vs 50-57 on clean neighbours).
#: Their GATES are valid; their WALL-CLOCK reads are not.
CONTAMINATED = {("stage1_start", "ac", "B-ctl"), ("stage1", "ac", "B-4k"),
                ("stage1", "ac", "S0", "true", 0), ("stage1", "ac", "S0", "true", 1),
                # 2026-09-29: a 1-thread CPU diagnostic 09:28:30-10:18 (stage1b.jsonl
                # lines 3-18) and the unit suite 10:27:41-10:29:35 (lines 21-22)
                ("stage1b", "ac", "S1-cap", "true", 2), ("stage1b", "ac", "S1-cap", "wrong", 0),
                ("stage1b", "ac", "S1-cap", "wrong", 1), ("stage1b", "ac", "S1-cap", "wrong", 2),
                ("stage1b", "ac", "S10"), ("stage1b", "ac", "S7f"),
                ("stage1b", "ac", "S8f", "true", 2), ("stage1b", "ac", "S8f", "wrong", 0),
                # Stage-2 patch tests in a separate worktree, 13:03:52-13:04:09,
                # 13:05:13-13:07:07 and 13:13:50-13:15:49 (lines 81-82 and 86)
                ("stage1b", "duffing", "P1", "true", 2), ("stage1b", "duffing", "P1", "wrong", 0),
                ("stage1b", "duffing", "P2", "true", 1),
                # obs_loss tests in a worktree, ~15:10-15:11 (r_baseline lines 21-22)
                ("r_baseline", "duffing", "B-ctl", "true", 2),
                ("r_baseline", "duffing", "B-ctl", "wrong", 0),
                # loss/objective tests in a worktree, 15:17:19-15:17:30 (line 25)
                ("r_baseline", "burgers", "B", "true", 0),
                # target-swap check and obs tests, 15:31-15:36 (lines 33-35)
                ("r_baseline", "burgers", "B-ctl", "true", 2),
                ("r_baseline", "burgers", "B-ctl", "wrong", 0),
                ("r_baseline", "burgers", "B-ctl", "wrong", 1),
                # 2026-09-30: this analysis at low priority, 14:28:39-14:29:30
                # (stage2 line 58; the solve logged +19.2 s overhead vs ~5 s)
                ("stage2", "ac", "L-os", "wrong", 0),
                # this analysis again, 20:04:14-20:04:56 (stage2 line 190)
                ("stage2", "duffing", "G-00gd", "wrong", 2)}


# ============================================================ trace replay
class Trace:
    """One L-BFGS trace: held-out RMSE at every iterate, and when it existed.

    Row n is written inside the line search of iteration n, BEFORE the move,
    so its RMSE is that of iterate n-1. Iterate k is therefore read from the
    first row with n > k; an iteration that took no line search (torch's
    ``gtd > -tolerance_change`` break) did not move the parameters, which this
    rule handles without special cases. Past the last row the parameters are
    the final ones.
    """

    def __init__(self, rec, var=None):
        self.rec = rec
        rows = rec["rows"]
        pick = (lambda x: x) if var is None else (lambda x: x[var])
        self.ns = [int(r[0]) for r in rows]
        if len(set(self.ns)) != len(self.ns):
            raise RuntimeError("duplicate line-search rows")
        self.r_before = [float(pick(r[1])) for r in rows]
        self.time = [float(r[7]) for r in rows]
        self.instr = [float(r[8]) for r in rows] if rows and len(rows[0]) > 8 else [0.0] * len(rows)
        self.final_n = rec["final_n_iter"] or 0
        self.final_rmse = pick(rec["rmse_final_replica"])
        self.t0 = float(rec.get("T_solve_start", 0.0))
        self.exit = (rec.get("lbfgs_exit") or {}).get("reason")
        self.cap = int((rec.get("config") or {}).get("lbfgs_maxiter", 0))

    def rmse_at(self, k):
        for n, r in zip(self.ns, self.r_before):
            if n > k:
                return r
        return self.final_rmse

    def end(self, cap):
        """Where a run with L-BFGS cap ``cap`` stops (exact-cap semantics)."""
        return min(int(cap), int(self.final_n))

    def last_k_within(self, budget):
        """Largest iterate reachable within ``budget`` seconds of solve time,
        and whether that read is censored (the cap ended the run first)."""
        best = 0
        within_all = True
        for i, n in enumerate(self.ns):
            if self.time[i] - self.instr[i] - self.t0 <= budget:
                best = n
            else:
                within_all = False
                break
        k = min(best, int(self.final_n))
        censored = within_all and self.exit == "maxiter"
        return k, censored


_TRACES = {}


def load_trace(row):
    rel = row["trace"].replace("\\", "/")
    path = rel if os.path.isabs(rel) else os.path.join(HERE, *rel.split("/"))
    key = (path, row.get("equation_of"))
    if key not in _TRACES:
        with open(path) as fh:
            _TRACES[key] = Trace(json.load(fh), row.get("equation_of"))
    return _TRACES[key]


def differing_equations(system):
    """The equations whose text differs between the true and the wrong form.
    Only their objectives carry the bar; the others are the same equation in
    both candidates and move only through the coupling."""
    forms = system_forms(system)
    if not isinstance(forms["true"], dict):
        return set(SYSTEM_VARS[system])
    return {v for v in SYSTEM_VARS[system] if forms["true"][v] != forms["wrong"][v]}


#: (system, arm) -> reference cell, for inert Stage-2 cells whose seed-0 solves
#: matched their reference bitwise; BROKEN_ALIAS holds the ones that did not
ALIASED, BROKEN_ALIAS = {}, {}


def apply_aliases(rows):
    """An inert Stage-2 cell (``gate.inert_reference``) ran on seed 0 only. When
    those solves equal its reference's bitwise (every equation, both forms), the
    reference's other seeds are copied in under the cell's name. A mismatch is
    recorded instead and the cell is left UNREADABLE. Chains (a 1-D periodic
    flat cell -> the flat cell -> G-0000) resolve in dependency order."""
    out = list(rows)
    changed = True
    while changed:
        changed = False
        have = {}
        for r in out:
            have[(r["system"], r["arm"], r.get("mode"), r["form"], r["seed"])] = r
        cells = sorted({k[:3] for k in have if "factors" in ARMS.get(k[1], {})})
        for system, arm, mode in cells:
            if (system, arm) in ALIASED or (system, arm) in BROKEN_ALIAS:
                continue
            ref = inert_reference(system, arm)
            if ref is None:
                continue
            if inert_reference(system, ref) is not None and (system, ref) not in ALIASED:
                continue                      # its reference is not final yet
            mine = {k[3:]: r for k, r in have.items() if k[:3] == (system, arm, mode)}
            theirs = {k[3:]: r for k, r in have.items() if k[:3] == (system, ref, mode)}
            if set(theirs) <= set(mine):
                continue          # run on every seed (--no-alias): read as its own cell
            checked = [(fs, mine[fs]["rmse"], theirs[fs]["rmse"]) for fs in mine if fs in theirs]
            if {fs[0] for fs, _, _ in checked} != {"true", "wrong"}:
                continue          # a form is missing at the checked seed: never aliased
            if all(a == b for _, a, b in checked):
                ALIASED[(system, arm)] = ref
                out += [dict(r, arm=arm, aliased_from=ref) for fs, r in theirs.items()
                        if fs not in mine]
                changed = True
                break           # rebuild the index: a chained cell must see these rows
            else:
                BROKEN_ALIAS[(system, arm)] = (ref, checked)
    return out


def resolve(system, arm):
    """The cell whose solves an (aliased) cell actually reuses."""
    while (system, arm) in ALIASED:
        arm = ALIASED[(system, arm)]
    return arm


def per_equation(rows):
    """Each equation of a coupled system is its own objective: a coupled row
    becomes one row per equation, as system '<system>:<var>' with that
    equation's RMSE. Single-variable rows pass through unchanged."""
    out = []
    for r in rows:
        if not isinstance(r["rmse"], dict):
            out.append(r)
            continue
        for var, value in r["rmse"].items():
            e = dict(r, system=f"{r['system']}:{var}", rmse=value, equation_of=var,
                     base_system=r["system"])
            if isinstance(r.get("rmse_final_replica"), dict):
                e["rmse_final_replica"] = r["rmse_final_replica"][var]
            out.append(e)
    return out


# ============================================================ tables
def index(rows):
    """{(system, arm, mode): {form: {seed: row}}}. B-ctl rows of the control
    tags are kept apart as 'B-ctl@<tag>'. The LAST row wins otherwise."""
    out = {}
    for r in rows:
        arm = r["arm"]
        if r.get("tag") in CONTROL_TAGS:
            arm = f"{arm}@{r['tag']}"
        out.setdefault((r["system"], arm, r["mode"]), {}) \
           .setdefault(r["form"], {})[r["seed"]] = r
    return out


def gates(by_form, value):
    """Per-seed gate over seeds present for BOTH forms, sorted by seed."""
    t, w = by_form.get("true", {}), by_form.get("wrong", {})
    seeds = sorted(set(t) & set(w))
    out = []
    for s in seeds:
        if w[s]["rmse"] >= FAILED_RMSE or t[s]["rmse"] >= FAILED_RMSE:
            out.append(float("nan"))           # a failed solve fails the seed
            continue
        a, b = value(w[s], s, "wrong"), value(t[s], s, "true")
        out.append(math.log(a / b) if (a > 0 and b > 0 and math.isfinite(a)
                                       and math.isfinite(b)) else float("nan"))
    return seeds, out


def contaminated(by_form):
    """Seeds of this arm whose wall-clock read cannot be trusted."""
    bad = set()
    for f, per in by_form.items():
        for seed, r in per.items():
            system = r.get("base_system", r["system"])
            keys = {(r.get("tag"), system, r["arm"]),
                    (r.get("tag"), system, r["arm"], f, seed)}
            if keys & CONTAMINATED:
                bad.add(seed)
    return sorted(bad)


def fmt_gates(seeds, g):
    if not g:
        return "(incomplete)"
    ok = all(math.isfinite(x) for x in g)
    bar = "PASS" if ok and all(x > 0 for x in g) else "FAIL"
    m = min(g) if ok else float("nan")
    return (" ".join(f"{x:+.3f}" for x in g) + f"   min {m:+.3f}  {bar}"
            + f"  [seeds {','.join(map(str, seeds))}]")


def _bitwise(pairs):
    return all(a == b for a, b in pairs) and len(pairs) > 0


#: (system, arm) -> reason, filled by integrity(); printed on the arm's rows
UNREADABLE = {}


def integrity(system, idx):
    print(f"--- integrity ({system})")
    ok = True
    base = idx.get((system, BASELINE, "solve"))
    for (sys_, arm, mode), by_form in sorted(idx.items()):
        if sys_ != system:
            continue
        name = arm.split("@")[0]
        spec = ARMS.get(name, {})
        ref_arm = spec.get("control_of") or spec.get("null_of") or spec.get("equals")
        if ref_arm is None:
            continue
        ref_mode = "solve" if ref_arm == BASELINE else mode
        ref = idx.get((system, ref_arm, ref_mode))
        if ref is None:
            continue
        pairs = [(ref[f][s]["rmse"], by_form[f][s]["rmse"])
                 for f in by_form for s in by_form[f] if s in ref.get(f, {})]
        same = _bitwise(pairs)
        if "equals" in spec:
            # a different code state than the reference's rows: a mismatch
            # says the Stage-2 patch moved S1, not that the batch drifted
            print(f"  {arm} == {ref_arm} (Stage-1 rows) bitwise on {len(pairs)} solves: {same}"
                  + ("" if same or not pairs else
                     "  -- the patch moved S1 on this device: compare grid cells "
                     "with G-0000 only"))
            continue
        ok &= same
        what = "null check" if "null_of" in spec else "control"
        if not same and "null_of" in spec:
            # a null that moves the result sets the noise floor of every arm
            # sharing its seam: S3-null covers any GPU work done BEFORE training
            # -- the init arms and the data-driven lr rules -- read from ARMS so
            # a new arm of that kind cannot be missed
            for affected, aspec in ARMS.items():
                c = aspec["config"]
                if (aspec["backend"] == "deepxde" and "null_of" not in aspec
                        and "control_of" not in aspec
                        and (c.get("init") is not None or c.get("lr_rule") is not None)):
                    UNREADABLE[(system, affected)] = (
                        f"{arm} != {ref_arm}: per-seed gates of arms adding GPU work "
                        f"before training move under a null change")
        print(f"  {what} {arm} == {ref_arm} bitwise on {len(pairs)} solves: {same}")
        for a, b in pairs:
            if a != b:
                print(f"      {a!r} vs {b!r}  rel {abs(b - a) / abs(a):.2e}")
    forced = base and all(r.get("intercept_rule", "forced") == "forced"
                          for f in base.values() for r in f.values())
    if system in REFERENCE_RMSE and base and forced:     # the Sep 18 refs forced one
        pairs = [(REFERENCE_RMSE[system][f][s], base[f][s]["rmse"])
                 for f in base for s in base[f] if s in REFERENCE_RMSE[system].get(f, {})]
        same = _bitwise(pairs)
        ok &= same
        print(f"  {BASELINE} reproduces the stored Sep 18 RMSEs bitwise on {len(pairs)} solves: {same}")
    # pairing: every arm starts from the initial weights of the S0-family arm
    # built on the SAME network (S0 for [64]x4, S0-32 for [32]x4); S6 upcasts
    # to float64, so its bytes differ by construction and it is not compared
    # A Stage-2 cell with a periodic embedding widens the first layer, so its
    # initial weights differ by construction (seed-matched, not weight-paired);
    # where no S0 exists (the coupled systems) the grid pairs with G-0000.
    refs = {}
    for ref_name in ("S0", "S0-32", "G-0000"):
        net_key = tuple(ARMS[ref_name]["config"].get("net", [64, 64, 64, 64]))
        if (system, ref_name, "trace") in idx and net_key not in refs:
            refs[net_key] = ref_name
    for (sys_, arm, mode), by_form in sorted(idx.items()):
        name = arm.split("@")[0]
        if sys_ != system or name in ("S0", "S0-32", "S6") or name not in ARMS:
            continue
        if ARMS[name]["config"].get("periodic_embedding") is not None:
            continue
        net = tuple(ARMS[name]["config"].get("net", [64, 64, 64, 64]))
        ref_name = refs.get(net)
        if ref_name is None or ref_name == name:
            continue
        ref = idx[(system, ref_name, "trace")]
        digests = [(ref[f][s].get("pairing") or {}).get("theta0") for f in by_form
                   for s in by_form[f] if s in ref.get(f, {})]
        mine = [(by_form[f][s].get("pairing") or {}).get("theta0") for f in by_form
                for s in by_form[f] if s in ref.get(f, {})]
        if all(d is not None for d in digests + mine) and mine:
            same = digests == mine
            ok &= same
            print(f"  {arm}: initial weights equal {ref_name}'s on {len(mine)} solves: {same}")
    b4 = idx.get((system, "B-4k", "trace"))
    if base and b4:
        pairs = []
        for f in b4:
            for s, r in b4[f].items():
                if s in base.get(f, {}):
                    tr = load_trace(r)
                    pairs.append((base[f][s]["rmse"], tr.rmse_at(tr.end(2000))))
        same = _bitwise(pairs)
        ok &= same
        print(f"  B-4k replayed at cap 2000 == plain {BASELINE} bitwise on {len(pairs)} solves: {same}")
    base_system = system.split(":")[0]
    for (sys_, arm), ref in sorted(ALIASED.items()):
        if sys_ == base_system:
            print(f"  inert {arm} == {ref} bitwise on seed 0: aliased (other seeds reuse {ref})")
    for (sys_, arm), (ref, checked) in sorted(BROKEN_ALIAS.items()):
        if sys_ == base_system:
            ok = False
            UNREADABLE[(system, arm)] = (f"expected inert, but seed 0 differs from {ref}: "
                                         f"rerun with drive --no-alias")
            print(f"  inert {arm} == {ref} on seed 0: FALSE")
            for fs, a, b in checked:
                if a != b:
                    print(f"      {fs}: {a!r} vs {b!r}")
    print(f"  => {'OK' if ok else 'INTEGRITY FAILURE -- do not read the affected arms'}")
    return ok


def budget_gates(by_form, budget):
    censored = []

    def value(r, s, f):
        tr = load_trace(r)
        k, c = tr.last_k_within(budget)
        if c:
            censored.append((f, s))
        return tr.rmse_at(k)
    seeds, g = gates(by_form, value)
    return seeds, g, censored


def gate_table(system, idx, T_B):
    print(f"--- gates ({system}); bar = gate > 0 on every seed; T_B = "
          + (f"{T_B:.0f} s" if T_B else "n/a (no plain B solves on this system)"))
    table = {}
    for (sys_, arm, mode), by_form in sorted(idx.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if sys_ != system or arm.startswith("_"):
            continue
        seeds, g = gates(by_form, lambda r, s, f: r["rmse"])
        backend = next(iter(next(iter(by_form.values())).values())).get("backend", "deepxde")
        label = f"{arm}{' [trace]' if mode == 'trace' else ''}{' [basis]' if backend == 'basis' else ''}"
        tt = by_form.get("true", {})
        exits = sorted({f"{(r.get('lbfgs_exit') or {}).get('reason')}"
                        for f in by_form.values() for r in f.values()})
        secs = [r["seconds"] for f in by_form.values() for r in f.values()]
        notes = []
        if (system, arm) in UNREADABLE:
            notes.append(f"INTEGRITY: {UNREADABLE[(system, arm)]}")
        if (system.split(":")[0], arm) in ALIASED:
            notes.append(f"alias of {ALIASED[(system.split(':')[0], arm)]}: identical by "
                         f"construction here (seed 0 checked bitwise)")
        if ARMS.get(arm.split("@")[0], {}).get("role") == "null":
            notes.append("placement null: a control, not a candidate (not ranked)")
        if (arm in ARMS and inert_reference(system.split(":")[0], arm) and len(seeds) == 1
                and (system.split(":")[0], arm) not in ALIASED):
            notes.append("inert cell NOT resolved: seed 0 only (reference missing, or a "
                         "form failed) -- not eligible")
        n_failed = sum(1 for f in by_form.values() for r in f.values()
                       if r["rmse"] >= FAILED_RMSE)
        if n_failed:
            notes.append(f"{n_failed} solve(s) FAILED (1e7)")
        bad = contaminated(by_form)
        if bad:
            notes.append(f"wall-clock CONTAMINATED on seeds {bad}")
        how = "own budget" if mode == "trace" else "at completion (not time-matched)"
        print(f"  {label:<30} {how}: {fmt_gates(seeds, g)}")
        for note in notes:
            print(f"  {'':<30} !! {note}")
        print(f"  {'':<30} true RMSE " + " ".join(f"{tt[s]['rmse']:.3e}" for s in seeds)
              + f"   exits {exits}   median {np.median(secs):.0f}s/solve")
        table[(arm, "own")] = (seeds, g)
        if mode == "trace" and T_B:
            for frac in (1.0, 0.9):
                sb, gb, cens = budget_gates(by_form, frac * T_B)
                tag = "T_B" if frac == 1.0 else "0.9T_B"
                note = f"  CENSORED {cens}" if cens else ""
                print(f"  {'':<30} @{tag:<7} {fmt_gates(sb, gb)}{note}")
                table[(arm, tag)] = (sb, gb)
            caps = []
            for cap in REPLAY_CAPS:
                sc, gc = gates(by_form, lambda r, s, f, cap=cap:
                               load_trace(r).rmse_at(load_trace(r).end(cap)))
                if gc and all(math.isfinite(x) for x in gc):
                    caps.append(f"{cap}:{min(gc):+.3f}")
            print(f"  {'':<30} min gate by L-BFGS cap  " + "  ".join(caps))
        elif T_B:
            # A plain solve has no trace to cut, so it is read at completion and
            # counts at a budget only if it actually finished within it. The
            # shipped recipe and its controls define T_B and are exempt.
            base_like = arm.split("@")[0] in (BASELINE,) or \
                ARMS.get(arm.split("@")[0], {}).get("control_of") == BASELINE
            for frac, key in ((1.0, "T_B"), (0.9, "0.9T_B")):
                if base_like:
                    table[(arm, key)] = (seeds, g)
                    continue
                over = [s for s in seeds
                        if max(by_form["true"][s]["seconds"], by_form["wrong"][s]["seconds"])
                        > frac * T_B]
                table[(arm, key)] = (seeds, [float("nan") if s in over else x
                                             for s, x in zip(seeds, g)])
                if over and frac == 1.0:
                    print(f"  {'':<30} !! took longer than T_B on seeds {over}: "
                          f"not read at T_B")
        extra = []
        for r in (x for f in by_form.values() for x in f.values()):
            st = r.get("sinv_stats")
            if st and st.get("data_mass"):
                extra.append(st["data_mass"][0])
        if extra:
            print(f"  {'':<30} data-mass: zero share {extra[0]['zero_share']:.3f}, "
                  f"max/median 1/mass {extra[0]['inv_max_over_median']:.2e}")
    return table


def contrasts(system, table):
    try:
        sys.path.insert(0, DATA)
        from pinn_common import contrast, contrast_table
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (contrasts unavailable: {exc})")
        return
    base_system = system.split(":")[0]

    def side(x, key):
        """Per-seed gates of an arm, or the paired difference of two."""
        if isinstance(x, tuple):
            first, second = side(x[0], key), side(x[1], key)
            if first is None or second is None:
                return None
            shared = sorted(set(first[0]) & set(second[0]))
            return shared, [first[1][first[0].index(s)] - second[1][second[0].index(s)]
                            for s in shared]
        return table.get((x, key))

    rows, same = [], []
    for name, a, b, budget in CONTRASTS:
        key = "own" if budget == "own" else budget
        if not isinstance(a, tuple) and not isinstance(b, tuple) \
                and (a, key) in table and (b, key) in table \
                and resolve(base_system, a) == resolve(base_system, b):
            same.append(f"{name}: identical by construction ({a} and {b} are one set of solves)")
            continue
        sa_ga, sb_gb = side(a, key), side(b, key)
        if sa_ga is None or sb_gb is None:
            continue
        sa, ga = sa_ga
        sb, gb = sb_gb
        shared = sorted(set(sa) & set(sb))
        if len(shared) < 2:
            continue
        va = [ga[sa.index(s)] for s in shared]
        vb = [gb[sb.index(s)] for s in shared]
        rows.append(contrast(name, va, vb, lower_is_better=False))
    for line in same:
        print(f"  {line}")
    if rows:
        print(f"--- paired contrasts on the per-seed gate ({system})")
        contrast_table(rows, label="gate", fmt="%+.3f")


def ranking(tables):
    print("\n================ ranking (pre-registered): eligible = finite + bar at T_B and "
          "0.9 T_B on every system; score = min gate at T_B over (system, seed)")
    arms = set()
    for t in tables.values():
        arms |= {a for a, _ in t}
    rows = []
    for arm in sorted(arms):
        if ARMS.get(arm.split("@")[0], {}).get("role") == "null":
            continue                          # a control, not a candidate
        per = []
        eligible = True
        for system, t in tables.items():
            full = set(t.get((BASELINE, "own"), ([], []))[0])
            if (system, arm) in UNREADABLE:
                eligible = False
            for key in ("T_B", "0.9T_B"):
                seeds, g = t.get((arm, key), ([], []))
                if not g or not all(math.isfinite(x) and x > 0 for x in g):
                    eligible = False
                if full and not full <= set(seeds):
                    eligible = False          # fewer seeds than the shipped recipe
            seeds, g = t.get((arm, "T_B"), ([], []))
            per.append(min(g) if g and all(math.isfinite(x) for x in g) else float("nan"))
        score = min(per) if all(math.isfinite(x) for x in per) else float("nan")
        rows.append((arm, eligible, score, per))
    rows.sort(key=lambda r: (not r[1], -(r[2] if math.isfinite(r[2]) else -1e9)))
    top = next((r[2] for r in rows if r[1]), None)
    for arm, eligible, score, per in rows:
        tie = " (tie with top)" if top is not None and eligible and top - score <= TIE else ""
        print(f"  {arm:<16} {'ELIGIBLE' if eligible else 'not eligible':<13} score {score:+.3f}  "
              + " ".join(f"{s} {x:+.3f}" for s, x in zip(tables, per)) + tie)


def _factor_label(token):
    """'u{power: 3.0}' -> 'u^3'; the power is shown only when it is not 1."""
    name = token.split("{")[0]
    power = token.split("power:")[1].split(",")[0].split("}")[0].strip() if "power:" in token else "1.0"
    return name if float(power) == 1.0 else f"{name}^{float(power):g}"


def true_coefficients(system):
    """{var: ([(term, coefficient), ...], free_coefficient)} of the TRUE form's
    text: the ground-truth equation the gate's true candidate is refit from."""
    forms = system_forms(system)["true"]
    texts = forms if isinstance(forms, dict) else {SYSTEM_VARS[system][0]: forms}
    out = {}
    for var, text in texts.items():
        lhs = text.split(" = ")[0]
        terms, free = [], 0.0
        for group in lhs.split(" + "):
            parts = group.split(" * ")
            if len(parts) == 1:
                free = float(parts[0])
            else:
                terms.append((" * ".join(_factor_label(p) for p in parts[1:]), float(parts[0])))
        out[var] = (terms, free)
    return out


def coefficient_report(rows, systems):
    """How close the candidate's PDE is to the true equation: the refit
    coefficients of the TRUE form (fixed before every solve -- the PINN does not
    change them, so every arm and seed carries the same values) against the true
    equation's, per term. Printed, and returned for results/coefficients_<rule>.json."""
    report = {}
    print("\n================ coefficients: refit true form vs the true equation")
    for system in systems:
        row = next((r for r in rows if r["system"] == system and r["form"] == "true"
                    and r.get("coeffs") is not None), None)
        if row is None or system not in SYSTEM_VARS:
            continue
        refit = row["coeffs"] if isinstance(row["coeffs"], dict) else \
            {SYSTEM_VARS[system][0]: row["coeffs"]}
        for var, (terms, free) in true_coefficients(system).items():
            got = refit[var]
            entries = []
            print(f"  {system}:{var}  (true  ->  refit, relative error)")
            for (term, want), have in zip(terms, got[:-1]):
                rel = (have - want) / abs(want) if want else float("nan")
                entries.append({"term": term, "true": want, "refit": have, "rel_error": rel})
                print(f"    {term:<16} {want:>+12.6g}  ->  {have:>+14.8g}   {rel:>+9.3%}")
            print(f"    {'free term':<16} {free:>+12.6g}  ->  {got[-1]:>+14.8g}")
            worst = max((abs(e["rel_error"]) for e in entries), default=float("nan"))
            print(f"    max |relative error| {worst:.3%}")
            report[f"{system}:{var}"] = {"terms": entries, "free_true": free,
                                         "free_refit": got[-1], "max_abs_rel_error": worst}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--systems", default="ac,duffing,burgers",
                    help="a coupled system (lv, ns, lorenz) expands to one gate per equation")
    a = ap.parse_args(argv)
    rows = [r for tag in a.tags.split(",") for r in load_rows(tag)]
    if not rows:
        print(f"no rows for tags {a.tags} under {RESULTS}")
        return
    rules = {r.get("intercept_rule", "forced") for r in rows}
    if len(rules) > 1:
        raise SystemExit(f"these tags mix candidate rules {sorted(rules)} (free coefficient "
                         f"forced vs from the form): analyse them separately")
    idx = index(per_equation(apply_aliases(rows)))
    tables = {}
    objectives = []
    for name in a.systems.split(","):
        for var in SYSTEM_VARS.get(name, [None]):
            label = name if len(SYSTEM_VARS.get(name, [])) <= 1 else f"{name}:{var}"
            if any(k[0] == label for k in idx):
                objectives.append((name, label, var))
    bar = {}
    for name, system, var in objectives:
        bar[system] = var is None or len(SYSTEM_VARS[name]) == 1 \
            or var in differing_equations(name)
        print(f"\n================ {system}"
              + ("" if bar[system] else
                 "   (same equation in both forms: coupling only, not part of the bar)"))
        integrity(system, idx)
        base = idx.get((system, BASELINE, "solve"), {})
        secs = [r["seconds"] for f in base.values() for r in f.values()]
        T_B = float(np.median(secs)) if secs else None
        tables[system] = gate_table(system, idx, T_B)
        contrasts(system, tables[system])
    ranked = {s: t for s, t in tables.items() if bar[s]}
    if len(ranked) > 1:
        ranking(ranked)
    report = coefficient_report(rows, a.systems.split(","))
    if report:
        path = os.path.join(RESULTS, f"coefficients_{rules.pop()}.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
