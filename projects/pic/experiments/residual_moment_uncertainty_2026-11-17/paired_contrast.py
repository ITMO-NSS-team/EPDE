from pathlib import Path
import sys,json,subprocess,time,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'contrast_stability_2026-11-17'))
import metric as C
R,np=C.R,C.np


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(name,seed):
    start=time.monotonic();path=ROOT/'results'/f'{name}_seed{seed}.json';meta=json.loads(path.read_text())
    fp=ROOT/'frozen'/path.name;f=json.loads(fp.read_text());a=np.load(path.with_suffix('.npz'));cs=f['candidates']
    p=R.datasets.load(name);allcs,_=R.panel(p);C.K.common_width(max(len(c['xi']) for c in allcs));_,old=C.E.F.baseline(name,0)
    ex=C.Extractor(cs,old['base_plan']['shape']);pilot=ex.extract(a['local_pilot'])[0];clean=ex.extract(a['local_clean'])[0]
    arrays=dict(pilot_contrasts=pilot,clean_contrasts=clean);rows=[]
    for tag in ('raw','corrected'):
        for ensemble in ('calibration','check','actual'):
            matrix=a['local_'+tag+'_'+ensemble];dd=[];defects=[]
            for m in matrix:
                d,_,df=ex.extract(m);dd.append(d);defects.append(df)
            arrays[tag+'_'+ensemble]=np.stack(dd);arrays[tag+'_'+ensemble+'_deficient']=np.stack(defects)
        model=C.NoiseModel(ex,pilot,arrays[tag+'_calibration']);arrays[tag+'_bias']=model.bias;arrays[tag+'_variance']=model.variance
        boot=model.score(arrays[tag+'_check']-pilot);actual=model.score(arrays[tag+'_actual']-clean)
        arrays[tag+'_check_q']=boot;arrays[tag+'_actual_q']=actual
        for j,c in enumerate(cs):
            rad=float(np.quantile(boot[:,j],.95));direct=next(r for r in meta['rows'] if r['arm']=='local_'+tag and r['candidate']==c['id'] and r['kind']=='exact')
            rows.append(dict(system=name,seed=seed,arm=tag,candidate=c['id'],label=c['label'],noise_coverage_count=int(np.sum(actual[:,j]<=rad)),
                direct_noise_coverage_count=direct['noise_coverage_count'],radius=rad,nonfinite=int(np.sum(~np.isfinite(boot[:,j]))+np.sum(~np.isfinite(actual[:,j]))),
                actual_deficient_count=int(np.sum(arrays[tag+'_actual_deficient'][:,j]>0))))
    dest=ROOT/'paired_contrast';dest.mkdir(exist_ok=True);ap=dest/path.with_suffix('.npz').name;np.savez_compressed(ap,**arrays)
    R.dump(ap.with_suffix('.json'),dict(system=name,seed=seed,rows=rows,groups=ex.groups,axis_groups=ex.axis_groups,
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path(C.__file__),ROOT/'paired_contrast_protocol.txt']},
        input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),fp]},array_sha256=sha(ap),seconds=time.monotonic()-start))


def worker(job):
    name,seed=job;path=ROOT/'logs'/f'paired_{name}_seed{seed}.txt'
    with path.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),name,str(seed)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=name,seed=seed,exit_code=r.returncode);print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    if len(sys.argv)>1:run(sys.argv[1],int(sys.argv[2]))
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:rr=[f.result() for f in as_completed([pool.submit(worker,(n,s)) for s in (24,25,26) for n in ('ac','kdv_cossin','ks','wave')])]
        R.dump(ROOT/'paired_processes.json',rr);sys.exit(any(r['exit_code'] for r in rr))
