#!/bin/bash
# Stage 2: the grid on route S1 -- output transform x periodic embedding x
# collocation density x precision, 24 cells G-<o><p><c><x> (gate.py), trace
# mode, seeds 0-2, one code state. G-m000 is S1+S2 (moment outputs on scaled
# inputs). Coupled systems are scored per equation (one gate per equation).
#
# Pre-flight, by hand, with no other work on the machine:
#   1. Stage-1b batch finished (STAGE1B_ALL_DONE in its log);
#   2. results/ moved here from projects/pic/data/solver_gate/results;
#   3. scratchpad patch_stage2.py, then patch_fit64.py, applied; tests appended;
#      the unit suite green; `python gate.py list` shows the 24 G cells.
# Inert cells (gate.inert_reference) run seed 0 only: analyze.py checks them
# bitwise against their reference and reuses the reference's other seeds.
cd "$(dirname "$0")"
set -x
# Code that must go in between the previous batch and this one (it changes files
# the previous batch's tags record) is applied while this waits: an operator (or
# a watcher) puts results/STAGE2_HOLD down first and results/STAGE2_GO when done.
if [ -f results/STAGE2_HOLD ]; then
  until [ -f results/STAGE2_GO ]; do sleep 60; done
fi
LOSSES=L-os,L-ps,L-pos,L-od
MAIN=G-0000,G-m000,G-0p00,G-00g0,G-00f0,G-000d,G-m00d,G-00gd,G-00fd
# G-m0fd, G-0pfd, G-mpfd (flat-warp float64 nulls beyond G-00fd) are left out:
# no contrast reads them and they cost ~4.5 GPU hours
REST=G-mp00,G-m0g0,G-m0f0,G-0pg0,G-0pf0,G-mpg0,G-mpf0,G-0p0d,G-mp0d,G-m0gd,G-0pgd,G-mpgd

# 0. the periodicity the inert-cell rule assumes (CPU seconds, no GPU)
for s in ac burgers duffing lv lorenz ns; do
  python gate.py profile --system $s || exit 1
done

# (patch neutrality under the old candidate rule is checked by run_queue.sh
#  before anything else: gate.py neutrality --tag stage2_legacy)

# 1. mechanics first, before any GPU hour is spent on controls: coupled rows per
#    equation, replica == host per equation, every decisive file the rows ran
#    with, and every new code path once (3-D warp on ns)
for s in lv lorenz ns; do
  python gate.py drive --system $s --arms _smoke --seeds 0 --mode trace --tag stage2_smoke || exit 1
done
for s in ac duffing burgers lv lorenz; do
  python gate.py drive --system $s --arms _smoke-G --seeds 0 --mode trace --tag stage2_smoke || exit 1
done
python gate.py drive --system ns --arms _smoke-m,_smoke-c --seeds 0 --mode trace --tag stage2_smoke || exit 1
for s in ac lv; do
  python gate.py drive --system $s --arms _smoke-L --seeds 0 --mode trace --tag stage2_smoke || exit 1
done
python gate.py check --tag stage2_smoke || exit 1

# 2. the candidates Stage 2 solves (no free term unless the form writes one),
#    and the start controls of this code state (vs the stage2 B rows)
for s in ac duffing burgers lv lorenz ns; do
  python gate.py forms --system $s
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage2_start
done

# 3. the shipped recipe under this candidate rule: T_B and the reference of
#    every system (the Stage-0 B rows were refit with a free coefficient)
for s in duffing ac burgers lv lorenz; do
  python gate.py drive --system $s --arms B --seeds 0-2 --tag stage2
done

# 3b. scale-invariant losses on S1: data (live, data mass), physics, both
#     (L2, i.e. DeepXDE's mean rho^2; ~6 GPU hours)
for s in duffing ac burgers lv lorenz; do
  python gate.py drive --system $s --arms $LOSSES --seeds 0-2 --mode trace --tag stage2
done

# 4. the grid: the cells behind the pre-registered contrasts on every system
#    first (a stopped batch still has its contrasts), then the rest
for s in duffing ac burgers lv lorenz; do
  python gate.py drive --system $s --arms $MAIN --seeds 0-2 --mode trace --tag stage2
done
for s in duffing ac burgers lv lorenz; do
  python gate.py drive --system $s --arms $REST --seeds 0-2 --mode trace --tag stage2
done

# 5. end controls
for s in ac duffing burgers lv lorenz; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage2_end
done
echo STAGE2_GRID_DONE

# 6. Navier-Stokes, reduced (~25 min per solve): the shipped recipe, S1 and
#    S1+S2; its control on seed 0. Stop here if a B solve takes over ~45 min.
python gate.py drive --system ns --arms B --seeds 0-2 --tag stage2
python gate.py drive --system ns --arms G-0000,G-m000 --seeds 0-2 --mode trace --tag stage2
python gate.py drive --system ns --arms B-ctl --seeds 0 --tag stage2_end
echo STAGE2_ALL_DONE
