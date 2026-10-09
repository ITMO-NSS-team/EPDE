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
DATA = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from gate import ARMS, load_rows       # noqa: E402

DEFAULT_TAGS = ("baseline,stage1_start,stage1,stage1_end,stage1_basis,"
                "stage1b_start,stage1b,stage1b_end,stage1b_basis")
#: tags whose B-ctl rows are kept apart from the pre-patch baseline's
CONTROL_TAGS = ("stage1_start", "stage1_end", "stage1b_start", "stage1b_end")

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
                ("stage1", "ac", "S0", "true", 0), ("stage1", "ac", "S0", "true", 1)}


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

    def __init__(self, rec):
        self.rec = rec
        rows = rec["rows"]
        self.ns = [int(r[0]) for r in rows]
        if len(set(self.ns)) != len(self.ns):
            raise RuntimeError("duplicate line-search rows")
        self.r_before = [float(r[1]) for r in rows]
        self.time = [float(r[7]) for r in rows]
        self.instr = [float(r[8]) for r in rows] if rows and len(rows[0]) > 8 else [0.0] * len(rows)
        self.final_n = rec["final_n_iter"] or 0
        self.final_rmse = rec["rmse_final_replica"]
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
    path = os.path.join(HERE, row["trace"])
    if path not in _TRACES:
        with open(path) as fh:
            _TRACES[path] = Trace(json.load(fh))
    return _TRACES[path]


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
            keys = {(r.get("tag"), r["system"], r["arm"]),
                    (r.get("tag"), r["system"], r["arm"], f, seed)}
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
        ref_arm = spec.get("control_of") or spec.get("null_of")
        if ref_arm is None:
            continue
        ref_mode = "solve" if ref_arm == BASELINE else mode
        ref = idx.get((system, ref_arm, ref_mode))
        if ref is None:
            continue
        pairs = [(ref[f][s]["rmse"], by_form[f][s]["rmse"])
                 for f in by_form for s in by_form[f] if s in ref.get(f, {})]
        same = _bitwise(pairs)
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
    if system in REFERENCE_RMSE and base:
        pairs = [(REFERENCE_RMSE[system][f][s], base[f][s]["rmse"])
                 for f in base for s in base[f] if s in REFERENCE_RMSE[system].get(f, {})]
        same = _bitwise(pairs)
        ok &= same
        print(f"  {BASELINE} reproduces the stored Sep 18 RMSEs bitwise on {len(pairs)} solves: {same}")
    # pairing: every arm starts from the initial weights of the S0-family arm
    # built on the SAME network (S0 for [64]x4, S0-32 for [32]x4); S6 upcasts
    # to float64, so its bytes differ by construction and it is not compared
    refs = {}
    for ref_name in ("S0", "S0-32"):
        if (system, ref_name, "trace") in idx:
            refs[tuple(ARMS[ref_name]["config"].get("net", [64, 64, 64, 64]))] = ref_name
    for (sys_, arm, mode), by_form in sorted(idx.items()):
        name = arm.split("@")[0]
        if sys_ != system or name in ("S0", "S0-32", "S6") or name not in ARMS:
            continue
        net = tuple(ARMS[name]["config"].get("net", [64, 64, 64, 64]))
        ref_name = refs.get(net)
        if ref_name is None:
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
    print(f"--- gates ({system}); bar = gate > 0 on every seed; T_B = {T_B:.0f} s")
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
    rows = []
    for name, a, b, budget in CONTRASTS:
        ka, kb = (a, "own" if budget == "own" else budget), (b, "own" if budget == "own" else budget)
        if ka not in table or kb not in table:
            continue
        sa, ga = table[ka]
        sb, gb = table[kb]
        shared = sorted(set(sa) & set(sb))
        if len(shared) < 2:
            continue
        va = [ga[sa.index(s)] for s in shared]
        vb = [gb[sb.index(s)] for s in shared]
        rows.append(contrast(name, va, vb, lower_is_better=False))
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
        per = []
        eligible = True
        for system, t in tables.items():
            for key in ("T_B", "0.9T_B"):
                seeds, g = t.get((arm, key), ([], []))
                if not g or not all(math.isfinite(x) and x > 0 for x in g):
                    eligible = False
            seeds, g = t.get((arm, "T_B"), ([], []))
            per.append(min(g) if g and all(math.isfinite(x) for x in g) else float("nan"))
        score = min(per) if all(math.isfinite(x) for x in per) else float("nan")
        rows.append((arm, eligible, score, per))
    rows.sort(key=lambda r: (not r[1], -(r[2] if math.isfinite(r[2]) else -1e9)))
    top = next((r[2] for r in rows if r[1]), None)
    for arm, eligible, score, per in rows:
        tie = " (tie with top)" if top is not None and eligible and top - score <= TIE else ""
        print(f"  {arm:<16} {'ELIGIBLE' if eligible else 'not eligible':<13} score {score:+.3f}  "
              f"per system {' '.join(f'{x:+.3f}' for x in per)}{tie}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAGS)
    ap.add_argument("--systems", default="ac,duffing,burgers")
    a = ap.parse_args(argv)
    rows = [r for tag in a.tags.split(",") for r in load_rows(tag)]
    idx = index(rows)
    tables = {}
    for system in a.systems.split(","):
        if not any(k[0] == system for k in idx):
            continue
        print(f"\n================ {system}")
        integrity(system, idx)
        base = idx.get((system, BASELINE, "solve"), {})
        secs = [r["seconds"] for f in base.values() for r in f.values()]
        T_B = float(np.median(secs)) if secs else None
        tables[system] = gate_table(system, idx, T_B)
        contrasts(system, tables[system])
    if len(tables) > 1:
        ranking(tables)


if __name__ == "__main__":
    main()
