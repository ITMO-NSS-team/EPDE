from common import *
import time_pilot_case as X
import time_pilot_student as T

def main():
    folder=ROOT/'time_pilot_preflight';X.ROOT=folder;X.DRAWS=4;T.ROOT=folder
    X.run('ac',47,'positive');T.run('ac','positive',47)
    dump(folder/'check.json',dict(status='PASS',draws=4,seed=47,not_primary=True,source_sha256=hashes('time_pilot_preflight.py','time_pilot_case.py','time_pilot_student.py','time_pilot_study.py')))
if __name__=='__main__':main()
