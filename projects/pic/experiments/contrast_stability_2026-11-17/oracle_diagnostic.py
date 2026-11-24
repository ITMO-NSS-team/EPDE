import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
from pathlib import Path
import json,time


def run(name):
    start=time.monotonic();seed=21;path=ROOT/'results'/f'{name}_seed{seed}.json';meta=json.loads(path.read_text())
    frozen=json.loads((ROOT/'frozen'/path.name).read_text());a=np.load(path.with_suffix('.npz'))
    assert meta['source_sha256'][C.__file__]==E.S.sha(C.__file__)
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    C.K.common_width(max(len(c['xi']) for c in cs));_,old=E.F.baseline(name,0);plan={k:tuple(v) for k,v in old['base_plan'].items()}
    bank=R.LocalBank(axes,terms,opaque,plan);ext=C.Extractor(cs,plan['shape']);clean=ext.extract(bank.matrix(p.data))[0]
    np.testing.assert_array_equal(clean,a['clean_contrasts']);arrays={};rows=[]
    models={'clean_estimated_noise':frozen['noise_model'],
            'clean_actual_noise':{v:[.05*float(np.std(y)),[0.]*y.ndim,[]] for v,y in p.data.items()}}
    for tag,model in models.items():
        stacks={}
        for label,count,domain in [('calibration',256,720130),('check',128,720131)]:
            rng=np.random.default_rng(np.random.SeedSequence([domain,seed]));draws=[]
            for b in range(count):
                field={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in p.data.items()}
                draws.append(ext.extract(bank.matrix(field))[0])
            stacks[label]=np.stack(draws);arrays[tag+'_'+label]=stacks[label]
        noise=C.NoiseModel(ext,clean,stacks['calibration'])
        qboot=noise.score(stacks['check']-clean);qactual=noise.score(a['benchmark_contrasts']-clean);qpoint=noise.score(a['observed_contrasts'])
        arrays[tag+'_bias']=noise.bias;arrays[tag+'_variance']=noise.variance
        arrays[tag+'_noise_q_check']=qboot;arrays[tag+'_noise_q_actual']=qactual;arrays[tag+'_point_q']=qpoint
        for j,c in enumerate(cs):
            q95=float(np.quantile(qboot[:,j],.95));base95=float(np.quantile(a['bootstrap_noise_q'][:,j],.95))
            rows.append(dict(system=name,candidate=c['id'],registered=c['registered'],reference=c['label']=='reference',features=len(c['xi']),
                control=tag,noise_coverage=float(np.mean(qactual[:,j]<=q95)),standard_noise_coverage=float(np.mean(a['benchmark_noise_q'][:,j]<=base95)),
                noise_q95=q95,point_q=float(qpoint[j]),standard_point_q=float(a['observed_q_bias'][j])))
        print(name,tag,'DONE',round(time.monotonic()-start,1),'s',flush=True)
    arr=ROOT/'oracle_diagnostic'/f'{name}.npz';arr.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(arr,**arrays)
    R.dump(arr.with_suffix('.json'),dict(system=name,seed=21,scope='Oracle attribution only; same benchmark fields, no deployable promotion.',rows=rows,
        source_sha256={str(p):E.S.sha(p) for p in [Path(__file__),Path(C.__file__),ROOT/'oracle_diagnostic_protocol.txt']},
        input_sha256={str(p):E.S.sha(p) for p in [path,path.with_suffix('.npz'),ROOT/'frozen'/path.name]},array_sha256=E.S.sha(arr),seconds=time.monotonic()-start))


if __name__=='__main__':
    for name in ('ac','kdv_cossin','ks','wave'):run(name)
