"""Objective-only ablation on saved, matched full candidate panels."""
import criteria as K
R,F,M,np,ROOT=K.R,K.F,K.M,K.np,K.ROOT
from pathlib import Path
import json,argparse,time


def sources():
    paths=[Path(__file__),Path(K.__file__),Path(R.__file__),Path(R.SV.__file__),ROOT/'protocol.txt']
    return {str(p):K.S.sha(p) for p in paths}


def run(name,seed):
    start=time.monotonic();basepath,old=F.baseline(name,seed);vpath=M.ROOT/'results'/f'{name}_seed{seed}.json'
    saved=json.loads(vpath.read_text());p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    assert [c['id'] for c in cs]==[c['id'] for c in saved['candidates']]
    K.common_width(max(len(c['xi']) for c in cs));raw=p.noisy(5,seed)
    for g in old['groups']:
        assert {v:R.digest(y[g['indices']]) for v,y in raw.items()}==g['observed_sha256']
    plan={k:tuple(v) for k,v in old['base_plan'].items()}
    bank=R.LocalBank(axes,terms,opaque,plan);matrix=bank.matrix(raw)
    values,details=K.evaluate(matrix,cs,old['base_plan']['shape'])
    np.testing.assert_allclose(values['dyad_sum'],saved['instability'],rtol=2e-10,atol=2e-11)
    discrepancy={};data_rows={}
    for arm,kind in [('q0_white128','joint'),('q0_white128','baseline'),('q0_plain','joint')]:
        key=arm+'/'+kind;rr=[r for r in saved['rows'][arm] if r['kind']==kind]
        assert [r['candidate'] for r in rr]==[c['id'] for c in cs]
        discrepancy[key]=[r['cross'] for r in rr];data_rows[key]=rr
    cleanfit=old['aggregates']['loo'][str(old['draws'])]['clean_fit_energy'];fronts=[]
    for arm,d in discrepancy.items():
        for metric,instability in values.items():
            for slot in sorted({c['slot'] for c in cs}):
                ff=K.front(d,instability,cs,slot);registered=[j for j in ff if cs[j]['registered']]
                ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                fronts.append(dict(arm=arm,metric=metric,slot=slot,retained=bool(registered),reference_retained=ref in ff,
                    size=len(ff),registered=len(registered),unregistered=len(ff)-len(registered),candidate_ids=[cs[j]['id'] for j in ff],
                    mean_features=float(np.mean([len(cs[j]['xi']) for j in ff])),
                    mean_clean_fit_energy=float(np.mean([cleanfit[j] for j in ff])),
                    max_clean_fit_energy=float(np.max([cleanfit[j] for j in ff]))))
    out=dict(system=name,seed=seed,scope='reused full-panel objective ablation',candidates=cs,labels=labels,
        instability=values,per_candidate=details,discrepancy=discrepancy,fronts=fronts,clean_fit_energy=cleanfit,
        maximum_dyad_replay_error=float(np.max(np.abs(np.array(values['dyad_sum'])-saved['instability']))),
        raw_data_sha256={v:R.digest(y) for v,y in raw.items()},input_sha256={str(p):K.S.sha(p) for p in (basepath,vpath)},
        source_sha256=sources(),elapsed_seconds=time.monotonic()-start)
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    print(name,seed,'DONE',round(out['elapsed_seconds'],1),'s',flush=True)


if __name__=='__main__':
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--system',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    if not (ROOT/'results'/f'{a.system}_seed{a.seed}.json').exists():run(a.system,a.seed)
