from bridge import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    import full_pipeline_saved as X
    X.ROOT=ROOT/'preflight_saved';X.BOOTSTRAP=4
    X.run(job[0],job[1],50)
    return dict(system=job[0],profile=job[1],status='PASS')

def main():
    assert read(ROOT/'metric_checks.json')['status']=='PASS';rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,j) for j in [('ac','positive'),('ks','positive'),('ns','iid'),('ode','iid')]]):rows.append(f.result())
    dump(ROOT/'preflight_saved/check.json',dict(status='PASS',draws=4,seed=50,not_primary=True,rows=rows,source_sha256=sources('preflight_saved.py','full_pipeline_saved.py','coverage_decision.txt')))
if __name__=='__main__':main()
