from calibration_context import *
def main():
    rows=[]
    for key in ('ode_iid_seed60','ode_positive_seed57','ode_negative_seed60'):
        e,d,a=Old.load_case(key);rng=np.random.default_rng(123)
        for j in range(3):
            fields,_=e.draw(rng);start=time.monotonic();model,info=Fit.estimate(fields['u']);elapsed=time.monotonic()-start;rows.append(dict(key=key,seconds=elapsed,status=info['status'],trigger=info['guard_triggered']))
    dump(ROOT/'benchmark.json',dict(status='COMPLETE',rows=rows,source_sha256=src('benchmark.py')));print(rows,flush=True)
if __name__=='__main__':main()
