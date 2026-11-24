from research_context import *


def main():
    jobs=[('ode',51,-1),('ode',52,-1),('ode',53,-1),('ode',52,112),('vdp',52,-1),('lv',51,-1)];rows=[]
    for name,seed,draw in jobs:
        e,d,path=primary(name,'iid',seed);v=next(iter(e.raw));field=e.raw[v];fit=d['fit_info'][v];model=d['model'][v]
        generating=[.05*float(np.std(e.p.data[v])),0.]
        if draw>=0:
            rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,seed]))
            for j in range(draw+1):fields,_=e.draw(rng)
            assert {var:R.digest(y) for var,y in fields.items()}==d['bootstrap_field_sha256'][draw]
            field=fields[v];model,fit=F.estimate(field);saved=d['refits'][draw]['model'][v]
            np.testing.assert_allclose(model[0],saved[0],rtol=1e-7);np.testing.assert_allclose(model[1],saved[1],rtol=1e-7,atol=1e-9)
            generating=d['model'][v][:2];generating=[generating[0],generating[1][0]]
        key=f'{name}_seed{seed}_'+('observed' if draw<0 else f'bootstrap{draw}')
        target=ROOT/'noise_inputs'/(key+'.npz');target.parent.mkdir(exist_ok=True);np.savez_compressed(target,field=field)
        out=dict(key=key,system=name,seed=seed,draw=draw,variable=v,model=model,fit_info=fit,
                 generating_noise_for_evaluation=generating,field_sha256=R.digest(field),array_sha256=sha(target),
                 source_sha256=sources('prepare_noise.py','protocol.txt'),input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))})
        dump(target.with_suffix('.json'),out);rows.append(key);print(key,field.shape,model,flush=True)
    dump(ROOT/'noise_preparation.json',dict(status='COMPLETE',objects=rows,source_sha256=sources('prepare_noise.py')))


if __name__=='__main__':main()
