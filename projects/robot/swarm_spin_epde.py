"""Per-robot spin laws from the swarm recordings with EPDE.

For every robot in every recording, EPDE searches the heading-vector system (c, s) = (cos phi, sin phi),
where a constant spin is the rotation  c' = -omega s,  s' = omega c  (see robot_discovery.py for why
phi itself cannot be the variable). Each robot runs in its own subprocess -- EPDE keeps
process-global caches -- under a hard timeout. Results go to results/swarm_spin_epde.csv.

    python swarm_spin_epde.py                         # all recordings, all robots
    python swarm_spin_epde.py --one N30_PWM10_D41 38  # a single robot, printed
    python swarm_spin_epde.py --summary               # PWM scaling / replicate tables from the CSV

Sparsity defaults to 'knee': the default 'vwsr' penalises terms by sliding-window coefficient
instability, and for robots whose spin is irregular (jammed, many contacts) it pruned the
rotation term, left the equation degenerate and stalled initialisation for 15-35 minutes.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# one BLAS thread per run: several run in parallel
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
# insert, not append: a stale pip-installed epde would otherwise shadow the repo
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../..')))
sys.path.insert(0, HERE)

import numpy as np

from swarm_data import swarm_files, swarm_meta, load_swarm

RESULTS = os.path.join(HERE, "results", "swarm_spin_epde.csv")
FIELDS = ["tag", "pwm", "n_bots", "arena_cm", "robot_id", "omega_phase", "omega_c", "omega_s",
          "sparsity", "status", "seconds", "front"]
WINDOW = 9  # SG window: spins reach 6.6 rad/s, so 15 frames would span ~4 rad of a period


def run_robot(tag, robot_id, sparsity="knee"):
    from epde.interface.interface import EpdeSearch
    path = next(p for p in swarm_files() if swarm_meta(p)['tag'] == tag)
    md = swarm_meta(path)
    t, x, y, phi, ids = load_swarm(path)
    ph = phi[:, list(ids).index(robot_id)]
    t0 = time.time()
    search = EpdeSearch(multiobjective_mode=True, use_solver=False, sparsity_cls=sparsity,
                        boundary_width=WINDOW,
                        default_preprocessor_type='poly',
                        preprocessor_kwargs={'polynomial_window': WINDOW, 'poly_order': 3, 'mp_poolsize': 1},
                        population_size=6, training_epochs=3,
                        verbose_params={'show_iter_idx': False})
    _, domain = search.createDomain((t,), ID=0)
    _, traj = search.createTrajectory({'c': np.cos(ph), 's': np.sin(ph)}, domain, cache_id=0)
    search.fit(data=[traj], max_deriv_order=(1,), data_fun_pow=1, equation_terms_max_number=4,
               equation_factors_max_number={'factors_num': [1], 'probas': [1.0]})
    front = search.equations(only_print=False, num=1)[0]
    texts = sorted({s.text_form.replace('\n', ' ').split(" {'max_terms")[0].replace('{power: 1.0}', '').strip()
                    for s in front})
    found = omega_from_front(' || '.join(texts))
    return dict(tag=tag, pwm=md['pwm'], n_bots=md['n'], arena_cm=md['arena_cm'], robot_id=robot_id,
                omega_phase=float(np.polyfit(t, ph, 1)[0]),
                omega_c=found.get('c', (None, np.nan))[1], omega_s=found.get('s', (None, np.nan))[1],
                sparsity=sparsity, status='ok', seconds=round(time.time() - t0, 1),
                front=' || '.join(texts))


def _run_subprocess(tag, robot_id, sparsity, timeout):
    """One robot in a fresh interpreter; on timeout the process is killed and a row says so."""
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', tag, str(robot_id),
                              '--sparsity', sparsity, '--json'],
                             capture_output=True, text=True, timeout=timeout, cwd=HERE)
        line = [l for l in out.stdout.splitlines() if l.startswith('{"tag"')]
        if line:
            return json.loads(line[-1])
        status = f'error: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "no output"}'
    except subprocess.TimeoutExpired:
        status = f'timeout {timeout}s'
    return dict(tag=tag, robot_id=robot_id, sparsity=sparsity, status=status,
                seconds=round(time.time() - t0, 1), omega_c=np.nan, omega_s=np.nan, front='')


def run_all(workers=6, sparsity="knee", timeout=300):
    jobs, phase = [], {}  # each pickle is read once here
    for path in swarm_files():
        md = swarm_meta(path)
        t, _, _, phi, ids = load_swarm(path)
        for k, rid in enumerate(ids):
            jobs.append((md['tag'], int(rid)))
            phase[jobs[-1]] = (md, float(np.polyfit(t, phi[:, k], 1)[0]))
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, newline='') as fh:
            done = {(r['tag'], int(r['robot_id'])) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} robot-runs, {len(done)} already in {RESULTS}, {len(todo)} to go", flush=True)
    new_file = not os.path.exists(RESULTS)
    with ThreadPoolExecutor(workers) as pool, open(RESULTS, "a", newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            wr.writeheader()
        futures = [pool.submit(_run_subprocess, tag, rid, sparsity, timeout) for tag, rid in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            md, w_phase = phase[(row['tag'], row['robot_id'])]
            row.update(pwm=md['pwm'], n_bots=md['n'], arena_cm=md['arena_cm'], omega_phase=w_phase)
            wr.writerow({k: row.get(k) for k in FIELDS}); fh.flush()
            print(f"[{i}/{len(todo)}] {row['tag']} id {row['robot_id']}: phase {w_phase:+.3f}, "
                  f"EPDE c {row['omega_c']:+.3f} s {row['omega_s']:+.3f} ({row['status']}, {row['seconds']}s)",
                  flush=True)


def _parse_equation(text):
    """'<k> * <term> + ... + <icpt> = <target>' -> ({term: k}, target)."""
    lhs, target = text.rsplit(' = ', 1)
    terms = {}
    for item in lhs.split(' + '):
        if ' * ' in item:
            k, name = item.split(' * ', 1)
            terms[name.strip()] = float(k)
    return terms, target.strip().split(' ')[0]


def omega_from_front(front):
    """omega per equation from the front text, whichever equation slot holds it (the SoEq keys its
    equations by pool position, not by variable). Normal forms c' = -w s, s' = w c; inverted forms
    s = -c'/w, c = s'/w (biased: they regress on a noisy derivative). Normal wins when both occur."""
    found = {}
    for soeq in front.split(' || '):
        soeq = soeq.split(" {'max_terms")[0]
        for eq in (e.strip(' /\\') for e in soeq.replace('\\', '|').split('|')):
            if ' = ' not in eq:
                continue
            terms, tgt = _parse_equation(eq)
            if len(terms) != 1:
                continue
            (name, k), = terms.items()
            form = {('dc/dx0', 's'): ('c', 'normal', -k), ('ds/dx0', 'c'): ('s', 'normal', k),
                    ('s', 'dc/dx0'): ('c', 'inverted', -1 / k), ('c', 'ds/dx0'): ('s', 'inverted', 1 / k)
                    }.get((tgt, name))
            if form and (form[0] not in found or found[form[0]][0] == 'inverted'):
                found[form[0]] = form[1:]
    return found


def summary():
    import pandas as pd
    df = pd.read_csv(RESULTS)
    df['front'] = df['front'].fillna('')
    parsed = df.front.apply(omega_from_front)
    for v in ('c', 's'):
        df[f'omega_{v}'] = parsed.apply(lambda d: d.get(v, (None, np.nan))[1])
        df[f'form_{v}'] = parsed.apply(lambda d: d.get(v, ('missing', None))[0])
    df['omega_epde'] = df[['omega_c', 'omega_s']].mean(axis=1)
    forms = pd.concat([df.form_c, df.form_s]).value_counts()
    print(f"runs: {len(df)}; status {df.status.value_counts().to_dict()}; equation forms {forms.to_dict()}; "
          f"median {df.seconds.median():.0f}s, max {df.seconds.max():.0f}s")
    rel = (df.omega_epde - df.omega_phase) / df.omega_phase.abs()
    print(f"EPDE omega vs phase-slope omega: median rel. diff {rel.median():+.4f}, IQR "
          f"[{rel.quantile(.25):+.4f}, {rel.quantile(.75):+.4f}], max |diff| {rel.abs().max():.3f}")
    piv = df.pivot_table(index='robot_id', columns='tag', values='omega_epde')
    print("\nper-robot EPDE omega [rad/s]:")
    print(piv.round(2).to_string())
    print("\nmean |omega| per recording:", piv.abs().mean().round(2).to_dict())
    for lo, hi in (("N40_PWM10_D40", "N40_PWM20_D40"), ("N40_PWM10_D41", "N40_PWM20_D41")):
        if lo in piv and hi in piv:
            r = (piv[hi] / piv[lo]).dropna()
            print(f"{hi} / {lo}: omega ratio median {r.median():.3f}, mean {r.mean():.3f} +- {r.std():.3f} "
                  f"(n={len(r)}); corr(|w_PWM10|, ratio) {np.corrcoef(piv[lo].abs()[r.index], r)[0, 1]:+.2f}")
    for a, b in (("N40_PWM10_D40", "N40_PWM10_D41"), ("N40_PWM20_D40", "N40_PWM20_D41"),
                 ("N30_PWM10_D41", "N40_PWM10_D41")):
        if a in piv and b in piv:
            r = (piv[b] / piv[a]).dropna()
            print(f"{b} / {a}: ratio median {r.median():.3f} +- {r.std():.3f} (n={len(r)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sparsity", default="knee", choices=["knee", "vwsr", "lasso"])
    ap.add_argument("--timeout", type=int, default=300, help="seconds per robot")
    ap.add_argument("--one", nargs=2, metavar=("TAG", "ROBOT_ID"))
    ap.add_argument("--json", action="store_true", help="with --one: print the row as JSON")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.one:
        row = run_robot(a.one[0], int(a.one[1]), a.sparsity)
        print(json.dumps(row, default=float) if a.json else row)
    else:
        run_all(a.workers, a.sparsity, a.timeout)
