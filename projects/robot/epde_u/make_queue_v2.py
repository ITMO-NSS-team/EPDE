"""Nov 12: the shared work queue of the robot revision on the frozen snapshot 9bd360e (queue_v2.txt).
Lines: <Arm> <Form> <TargetTime 0/1> <Sys> <Epochs> [<selected cap>]. Arms: U2 strong, UW2 weak (both EPDE defaults),
UT2 strong / UWT2 weak with system_rps_params target_axes 0 (fields only: spin and drift have the time axis alone).
Longest runs first (Nov 10/11 walls: drift ~520-600 s, strong fields ~450 s, spin ~150-250 s, weak fields ~160 s), so
the lanes finish together. python make_queue_v2.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPIN = ['spin_' + r for r in ['N30_PWM10_D41_12', 'N30_PWM10_D41_42', 'N40_PWM10_D41_17', 'N40_PWM10_D41_20',
                              'N40_PWM20_D41_66', 'N1_PWM10_1']]
DRIFT = ['drift%s_%s' % (a, r) for r in ['N30_PWM10_D41_12', 'N40_PWM10_D40_27', 'N40_PWM20_D41_18'] for a in 'xy']
DRIFT.append('driftx_N1_PWM10_1')
TAGS = ['N40_PWM10_D40', 'N30_PWM10_D41', 'N40_PWM10_D41', 'N40_PWM20_D40', 'N40_PWM20_D41']
FIELDS = ['fieldv2_%s_%s' % (v, t) for t in TAGS for v in ['rho', 'Px', 'Py', 'Wx', 'Wy']]

lines = ['# Nov 12 robot revision on snapshot 9bd360e: <Arm> <Form> <TargetTime> <Sys> <Epochs> [<cap>]']
lines += ['UW2 weak 0 %s 20 8' % s for s in DRIFT]
lines += ['U2 strong 0 %s 20 8' % s for s in DRIFT]
for f in FIELDS:                       # strong fields, the two arms interleaved per field
    lines += ['U2 strong 0 %s 30' % f, 'UT2 strong 1 %s 30' % f]
lines += ['U2 strong 0 %s 20' % s for s in SPIN]
for f in FIELDS:
    lines += ['UW2 weak 0 %s 30' % f, 'UWT2 weak 1 %s 30' % f]
lines += ['UW2 weak 0 %s 20' % s for s in SPIN]
open(os.path.join(HERE, 'queue_v2.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines) - 1, 'runs')
