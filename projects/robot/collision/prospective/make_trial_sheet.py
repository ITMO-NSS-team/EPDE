"""Balanced pilot schedule for the controlled pair experiment."""
import csv
from itertools import product
from pathlib import Path
import random


def main():
    rng = random.Random(25481)
    settings = list(product(('++','--','+-','-+'),(150,290),(0.,.65,.97,1.03)))
    rows = []
    for block in range(4):
        order = list(enumerate(settings))
        rng.shuffle(order)
        for idx,(spin,speed,b) in order:
            # The two grazing settings need the same mirror within a speed/spin/block.
            spin_index = idx // 8
            speed_index = (idx // 4) % 2
            mirror = 1 if (spin_index+speed_index+block)%2==0 else -1
            rows.append(dict(trial=len(rows)+1,block=block+1,target_spin_A=spin[0],target_spin_B=spin[1],
                target_relative_speed_px_s=speed,target_signed_impact_parameter_over_D=mirror*b if b else 0.,
                intended_path={0.:'head-on',.65:'oblique',.97:'grazing contact',1.03:'near miss'}[b],
                robot_A='',robot_B='',PWM_A='',PWM_B='',session='',clip='',measured_speed_px_s='',
                measured_impact_parameter_over_D='',measured_spin_A_rad_s='',measured_spin_B_rad_s='',
                observed_contact='',contact_frame='',deviation=''))
    with Path(__file__).with_name('two_robot_trials.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=='__main__':
    main()
