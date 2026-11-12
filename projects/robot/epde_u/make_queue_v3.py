"""Nov 12 (USER "Need a way to avoid unit-circle relation" -> both ways, as separate arms):
  * queue_v2.txt is rewritten (atomically: lanes re-read it before every run) with the spin work first:
    spin_ U2 / UW2 (already queued), then the new per-variable spin spinvc_ / spinvs_ U2 / UW2 (flag 0: the running
    lanes' old queue2.ps1 handles them), then every other line in its old order;
  * queue_v3.txt (only lanes on the new queue2.ps1 read it): the invariant arm UI2 strong / UWI2 weak on spin_
    (flag I: invariant_terms='refuse' on epde_snap_9bd360e_inv), then queue_v2's lines.
python make_queue_v3.py"""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS = ['N30_PWM10_D41_12', 'N30_PWM10_D41_42', 'N40_PWM10_D41_17', 'N40_PWM10_D41_20', 'N40_PWM20_D41_66',
          'N1_PWM10_1']


def write_atomic(name, lines):
    path = os.path.join(HERE, name)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:            # a lane is reading it
            time.sleep(0.2)
    raise RuntimeError('could not replace ' + path)


old = [l.rstrip('\n') for l in open(os.path.join(HERE, 'queue_v2.txt'), encoding='utf-8')]
head = [l for l in old if l.startswith('#')]
body = [l for l in old if l.strip() and not l.startswith('#')]
spin = [l for l in body if l.split()[3].startswith('spin_')]
spin = sorted(spin, key=lambda l: l.split()[0] != 'U2')            # strong first, file order kept within
spinv = ['%s %s 0 spinv%s_%s 20' % (arm, form, v, rb) for arm, form in (('U2', 'strong'), ('UW2', 'weak'))
         for rb in ROBOTS for v in 'cs']
rest = [l for l in body if l not in spin]
new_v2 = head + ['# Nov 12: spin first (spin_ U2/UW2, then per-variable spinvc_/spinvs_), then the rest'] + spin + \
    spinv + rest
assert len(set(new_v2)) == len(new_v2)
write_atomic('queue_v2.txt', new_v2)
inv = ['%s %s I spin_%s 20' % (arm, form, rb) for arm, form in (('UI2', 'strong'), ('UWI2', 'weak')) for rb in ROBOTS]
write_atomic('queue_v3.txt', ['# Nov 12: invariant arm (flag I) first, then queue_v2'] + inv + new_v2[len(head):])
print('queue_v2: %d runs (%d spin, %d spinv, %d rest); queue_v3: %d invariant + the same' % (
    len(spin) + len(spinv) + len(rest), len(spin), len(spinv), len(rest), len(inv)))
