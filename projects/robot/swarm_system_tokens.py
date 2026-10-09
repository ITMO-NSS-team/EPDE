r"""The swarm as ONE system: may robot i's law use tokens of every other robot?

For each robot i and coordinate, on top of its own law
    base   c_i, s_i, push_i, wall_i, vb_i, 1          (own propulsion, contact push, wall, persistence)
two kinds of extra tokens are offered:
    local  sum over robots touching i (d < 1.1 sigma) of their velocity over the last 0.4 s --
           an identity-free neighbour coupling
    system for EVERY other robot j: vb_j (x, y), c_j, s_j, x_j, y_j (arena-centred) -- identity tokens,
           6 (N - 1) = 174-234 columns
Target: forward increment over 0.4 s (all tokens causal, at time t). Exhaustive subsets are out of
reach at ~240 columns, so system tokens enter by greedy forward selection (up to 12), the number
chosen by held-out R2 on interleaved 33 s blocks (select on one half of the blocks, test on the other,
and swap). Reported per recording: held-out R2 of base, base + local, base + system, base + local +
system; and for the robots whose tokens were picked, how often they touch robot i.

python swarm_system_tokens.py      -> results/swarm_system_tokens.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
from swarm_data import swarm_files, ROBOT_DIAMETER_PX as SIGMA   # noqa: E402

MAX_ADD, N_BLOCKS = 12, 10
SYS_TOK = ["vb_x", "vb_y", "c", "s", "x", "y"]


def neighbour_velocity(R):
    """Sum of touching robots' last-step velocity, and the contact indicator matrix per row."""
    pos = np.stack([R["tok"][..., SS.TI["x"]], R["tok"][..., SS.TI["y"]]], -1)
    vb = np.stack([R["tok"][..., SS.TI["vb_x"]], R["tok"][..., SS.TI["vb_y"]]], -1)
    T, N = pos.shape[:2]
    nv, contact = np.zeros((T, N, 2)), np.zeros((N, N))
    for t0 in range(0, T, 500):
        sl = slice(t0, min(T, t0 + 500))
        D = np.linalg.norm(pos[sl][:, :, None] - pos[sl][:, None], axis=-1)
        D[:, np.arange(N), np.arange(N)] = np.inf
        C = (D < 1.1 * SIGMA).astype(float)
        nv[sl] = np.einsum("tij,tjc->tic", C, vb[sl])
        contact += C.sum(0)
    return nv, contact / T


def grams(X, y, blk):
    return [(X[blk == k].T @ X[blk == k], X[blk == k].T @ y[blk == k], float(y[blk == k] @ y[blk == k]),
             float(y[blk == k].sum()), int((blk == k).sum())) for k in (0, 1)]


def solve(G, b, S):
    return np.linalg.solve(G[np.ix_(S, S)] + 1e-9 * np.trace(G) / len(G) * np.eye(len(S)), b[S])


def sse(g, S, coef):
    G, b, yy = g[:3]
    return yy - 2 * coef @ b[S] + coef @ G[np.ix_(S, S)] @ coef


def sst(g):
    _, _, yy, ys, n = g
    return yy - ys ** 2 / n


def greedy(G, b, yy, start, cands, k_max):
    S, path = list(start), []
    for _ in range(k_max):
        best, best_rss = None, np.inf
        for c in cands:
            if c in S:
                continue
            T = S + [c]
            rss = yy - b[T] @ solve(G, b, T)
            if rss < best_rss:
                best, best_rss = c, rss
        S.append(best)
        path.append(best)
    return path


def cv_curve(gs, start, cands, k_max):
    """Held-out R2 after 0..k_max greedy additions: select on one block half, test on the other."""
    err, tot = np.zeros(k_max + 1), 0.0
    for tr, te in ((0, 1), (1, 0)):
        G, b, yy = gs[tr][:3]
        path = greedy(G, b, yy, start, cands, k_max)
        for k in range(k_max + 1):
            S = list(start) + path[:k]
            err[k] += sse(gs[te], S, solve(G, b, S))
        tot += sst(gs[te])
    return 1 - err / tot


def main():
    out = {}
    for f in swarm_files():
        R = SS.recording(f)
        N = len(R["ids"])
        nv, contact = neighbour_velocity(R)
        T = R["tok"].shape[0]
        blk = (np.arange(T) * N_BLOCKS // T) % 2
        res = dict(r2={k: [] for k in ("base", "local", "system", "local+system")}, picked_contact=[],
                   all_contact=[], picked_types={}, n_added=[])
        for i in range(N):
            for comp, v in enumerate(("x", "y")):
                own = [R["tok"][:, i, SS.TI[k]] for k in ("c", "s", f"push_{v}", f"wall_{v}", f"vb_{v}", "1")]
                others = [j for j in range(N) if j != i]
                sys_cols = [R["tok"][:, j, SS.TI[k]] for j in others for k in SYS_TOK]
                X = np.column_stack(own + [nv[:, i, comp]] + sys_cols)
                y = (R["fx"] if v == "x" else R["fy"])[:, i]
                gs = grams(X, y, blk)
                base, loc = list(range(6)), [6]
                cands = list(range(7, X.shape[1]))
                r_base = cv_curve(gs, base, [], 0)[0]
                r_loc = cv_curve(gs, base + loc, [], 0)[0]
                c_sys = cv_curve(gs, base, cands, MAX_ADD)
                c_ls = cv_curve(gs, base + loc, cands, MAX_ADD)
                res["r2"]["base"].append(r_base); res["r2"]["local"].append(r_loc)
                res["r2"]["system"].append(float(c_sys.max())); res["r2"]["local+system"].append(float(c_ls.max()))
                # which robots does the system pick (selection on all rows, size chosen by held-out)?
                k = int(np.argmax(c_sys))
                res["n_added"].append(k)
                G = sum(g[0] for g in gs); b = sum(g[1] for g in gs); yy = sum(g[2] for g in gs)
                for c in greedy(G, b, yy, base, cands, k):
                    j = others[(c - 7) // len(SYS_TOK)]
                    res["picked_contact"].append(float(contact[i, j]))
                    t = SYS_TOK[(c - 7) % len(SYS_TOK)]
                    res["picked_types"][t] = res["picked_types"].get(t, 0) + 1
                res["all_contact"] += [float(contact[i, j]) for j in others]
        r2 = {k: np.array(v) for k, v in res["r2"].items()}
        pc, ac = np.array(res["picked_contact"]), np.array(res["all_contact"])
        print(f"\n### {R['tag']} ({N} robots, {2 * N} equations)")
        print("  held-out R2 (median over equations): " + ", ".join(f"{k} {np.median(v):.4f}" for k, v in r2.items()))
        print(f"  gain over base: local {np.median(r2['local'] - r2['base']):+.4f}, system "
              f"{np.median(r2['system'] - r2['base']):+.4f}, local+system {np.median(r2['local+system'] - r2['base']):+.4f}; "
              f"system beyond local {np.median(r2['local+system'] - r2['local']):+.4f}")
        print(f"  system tokens chosen per equation: median {np.median(res['n_added']):.0f} "
              f"(0 in {100 * np.mean(np.array(res['n_added']) == 0):.0f}% of equations); types "
              + ", ".join(f"{k} {v}" for k, v in sorted(res["picked_types"].items(), key=lambda kv: -kv[1])))
        if len(pc):
            print(f"  picked robots touch robot i {100 * np.mean(pc):.0f}% of the time (median {100 * np.median(pc):.0f}%) "
                  f"vs {100 * np.mean(ac):.0f}% for a random other robot; frequent neighbours (>20% contact) "
                  f"{100 * np.mean(pc > 0.2):.0f}% of picks vs {100 * np.mean(ac > 0.2):.0f}% of robots", flush=True)
        out[R["tag"]] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
        json.dump(out, open(os.path.join(HERE, "results", "swarm_system_tokens.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
