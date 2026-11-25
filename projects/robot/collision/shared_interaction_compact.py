"""Fit and audit fixed two-term interaction laws; freeze the pooled prospective scorer."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from shared_interaction_fit import NAMES, balanced, family, history_basis, library, periods
import numpy as np
import pandas as pd

from shared_interaction_data import BASE, DEST, D_MM, MS
from shared_interaction_score import KEY, MODEL, MODELS, REFERENCES, digest, evaluate, pose_hash, score_file

OUT=DEST/'compact'
ROBOT=Path(__file__).resolve().parents[1]
MASK={'n':3,'t':33}


def development():
    frames=[pd.read_parquet(p).query("mode=='native' and kind=='near'")
            for p in sorted((DEST/'data').glob('*_pairs.parquet'))]
    return pd.concat(frames,ignore_index=True)


def fitted(data,label,training):
    prefix,_=periods(data)
    train=data[prefix & data.recording.isin(training)]
    root=DEST/'drive_sensitivity'/('native_'+label)
    selected=pd.read_csv(root/'fronts.csv').query("selected and timing=='separate'")
    isolated=pd.read_csv(DEST/'folds'/('native_'+label)/'free_fits.csv').set_index('horizon')
    model={}
    w=np.sqrt(balanced(train))
    for ms in MS:
        block={'isolated_motion':json.loads(isolated.loc[ms,'coeff'])}
        for a in 'nt':
            y=train[f'y{ms}_{a}'].to_numpy()
            f=train['recent_'+a].to_numpy()
            x=library(train,ms/1000,a,f)
            cols=[k for k in range(6) if MASK[a]&(1<<k)]
            b,_,rank,_=np.linalg.lstsq(w[:,None]*x[:,cols],w*(y-f),rcond=None)
            if rank!=len(cols):
                raise ValueError(f'Rank-deficient compact fit in {label}, {ms}, {a}')
            compact=np.zeros(6)
            compact[cols]=b
            full=selected[(selected.horizon==ms)&(selected.axis==a)].coeff.unique()
            assert len(full)==1
            hist=np.linalg.lstsq(w[:,None]*history_basis(train,a),w*y,rcond=None)[0]
            block[a]=dict(compact=compact.round(13).tolist(),full=json.loads(full[0]),history=hist.round(13).tolist())
        model[str(ms)]=block
    return {'horizons':model}


def sources():
    for p in sorted(BASE.glob('N*_PWM*_D*.npz')):
        arena=json.loads((BASE.parent/p.stem/'training.json').read_text())['arenas'][p.stem[-2:]]
        yield p,dict(arena=arena)
    for p in sorted(ROBOT.glob('Experiment_[123].zip')):
        yield p,dict(calibration=ROBOT/'results/pair_interactions_sweeps/physical/scale.csv')


def freeze(data):
    model=fitted(data,'pooled',sorted(data.recording.unique()))
    files,members,poses={},{},{}
    for path,_ in sources():
        files[digest(path)]=path.stem
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if '/data_N=' in name:
                        with z.open(name) as stream:
                            members[hashlib.file_digest(stream,'sha256').hexdigest()]=path.stem
        else:
            with np.load(path,allow_pickle=False) as z:
                poses[pose_hash(z['raw_pos'],z['ids'])]=path.stem
    model.update(version='shared-pair-v1',diameter_mm=D_MM,
        training_recordings=sorted(data.recording.unique()),training_period='first 75% with 400 ms outcome guard',
        development_inputs=files,development_members=members,development_poses=poses,
        term_order=NAMES,primary_model='full',secondary_model='compact',
        primary_endpoints=['n40','n80','t40','t80','t400'],secondary_endpoints=['n400'],
        comparison='Each endpoint against lowest RMSE of the five frozen reference forecasts on the same recording.',
        uncertainty='2000 paired draws of occupied 10-second blocks within a recording, seed 731; conditional intervals per endpoint.',
        gate='Inward crossing of 1.10 diameters after 1.30 clearance; axis points from i to j at gate.',
        implementation_sha256={p.name:digest(p) for p in [Path(__file__).with_name(n) for n in
            ['shared_interaction_score.py','shared_interaction_data.py','shared_interaction_fit.py','impacts.py','new_pair_interactions.py']]})
    payload=(json.dumps(model,indent=2,allow_nan=False)+'\n').encode()
    if MODEL.exists():
        if MODEL.read_bytes()!=payload:
            raise ValueError('The frozen shared-pair-v1 file differs. Preserve it when creating a new version.')
    else:
        with MODEL.open('xb') as stream:
            stream.write(payload)
    return model


def compact_comparison(data):
    tags=sorted(data.recording.unique())
    by_recording=[];summaries=[]
    designs=[('recording',t,[u for u in tags if u!=t],[t]) for t in tags]
    for fam in ('swarm','sweep'):
        designs.append(('cross-family','train_'+fam,[u for u in tags if family(u)==fam],
                        [u for u in tags if family(u)!=fam]))
    blocks=[]
    for design,label,tr,te in designs:
        model=fitted(data,label,tr)
        score=evaluate(data[data.recording.isin(te)],model)
        root=DEST/'drive_sensitivity'/('native_'+label)
        archive=pd.read_parquet(root/'fixed_support.parquet').query("timing=='separate'")
        original=pd.read_parquet(root/'predictions.parquet').query("kind=='near'")
        k=['recording','stage','frame','i','j','axis','horizon_ms']
        check=score.merge(original.rename(columns={'horizon':'horizon_ms'})[k+['separate','y']],on=k,validate='one_to_one')
        assert len(check)==len(score)==len(original)
        np.testing.assert_allclose(check.full,check.separate*D_MM,atol=1e-8)
        np.testing.assert_allclose(check.observed,check.y*D_MM,atol=1e-8)
        for (held,ms,a),d in score.groupby(['recording','horizon_ms','axis']):
            y=d.observed.to_numpy()
            losses=(d[MODELS].to_numpy()-y[:,None])**2
            rmse=np.sqrt(losses.mean(0))
            r2=1-losses.mean(0)/y.var()
            ref=archive[(archive.held==held)&(archive.axis==a)&(archive.horizon==ms)&(archive['mask']==MASK[a])].iloc[0]
            np.testing.assert_allclose([rmse[1],r2[1]],[ref.rmse_mm_s,ref.r2],atol=1e-8)
            for name,err,r in zip(MODELS,rmse,r2):
                by_recording.append(dict(design=design,recording=held,family=family(held),axis=a,horizon_ms=ms,
                                         model=name,events=len(d),rmse_mm_s=err,r2=r,compact_on_front=bool(ref.on_front)))
            block=pd.DataFrame(losses,columns=MODELS).assign(stage=d.stage.to_numpy(),block=d.seconds.to_numpy()//10,n=1)
            sums=block.groupby(['stage','block']).sum().to_numpy()
            rng=np.random.default_rng(731+tags.index(held))
            boot=sums[rng.integers(len(sums),size=(2000,len(sums)))].sum(1)
            blocks.append((design,family(held),a,ms,rmse,np.sqrt(boot[:,:-1]/boot[:,-1,None])))
    by=pd.DataFrame(by_recording)
    by.to_csv(OUT/'by_recording.csv',index=False)
    for key,_ in by.groupby(['design','family','axis','horizon_ms']):
        rows=[b for b in blocks if b[:4]==key]
        rm=np.mean([b[4] for b in rows],axis=0)
        boots=np.mean([b[5] for b in rows],axis=0)
        strongest=rm[2:].min()
        gain=strongest-rm[1]
        margins=boots[:,2:].min(1)-boots[:,1]
        lo,hi=np.quantile(margins,[.025,.975])
        dl,dh=np.quantile(boots[:,1]-boots[:,0],[.025,.975])
        summaries.append(dict(zip(('design','family','axis','horizon_ms'),key),
            reference=REFERENCES[rm[2:].argmin()],benchmark_rmse=strongest,full_rmse=rm[0],
            compact_rmse=rm[1],gain_mm_s=gain,gain_low=lo,gain_high=hi,
            compact_minus_full=rm[1]-rm[0],difference_low=dl,difference_high=dh,
            retained_gain_fraction=gain/(strongest-rm[0]) if strongest>rm[0] else np.nan))
    result=pd.DataFrame(summaries)
    result.to_csv(OUT/'transfer.csv',index=False)
    return result


def replay(data,model):
    rows=[]
    with TemporaryDirectory() as tmp:
        check_synthetic_inputs(Path(tmp))
        for path,opts in sources():
            folder=Path(tmp)/path.stem
            receipt=score_file(path,folder,**opts)
            assert receipt['identity']=='development overlap' and receipt['matched_development_recording']==path.stem
            got=pd.read_parquet(folder/'predictions.parquet')
            expected=evaluate(data[data.recording==path.stem],model)
            key=KEY+['horizon_ms','axis']
            assert not got.duplicated(key).any()
            actual=got.set_index(key).sort_index()
            want=expected.set_index(key).sort_index()
            pd.testing.assert_index_equal(actual.index,want.index)
            cols=MODELS+['observed']
            np.testing.assert_allclose(actual[cols],want[cols],atol=1e-9,rtol=1e-12)
            rows.append(dict(recording=path.stem,container=path.suffix,encounters=receipt['events'],
                forecasts=len(got),max_absolute_error_mm_s=np.max(abs(actual[cols].to_numpy()-want[cols].to_numpy())),
                input_sha256=receipt['input_sha256'],model_sha256=receipt['model_sha256']))
            print(path.stem,receipt['events'],'gate crossings replayed',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'replay.csv',index=False)


def check_synthetic_inputs(tmp):
    """Known motion, blank recording, and changed future through the prospective entry point."""
    for hz in (25,50):
        t=np.arange(3*hz)/hz
        separation=1.6-.3*t
        gate=np.flatnonzero(separation<1.10)[0]
        outputs=[]
        for variant in ('linear','turn','no_gate'):
            r=separation.copy()
            if variant=='turn':
                r+=.8*np.maximum(t-t[gate],0)
            elif variant=='no_gate':
                r[:]=1.6
            xy=np.zeros((len(t),2,2))
            xy[:,1,0]=r
            root=tmp/f'check_{hz}_{variant}'
            file=root.with_suffix('.npz')
            np.savez(file,raw_pos=xy,phi=np.column_stack([t,-t]),frame=np.arange(len(t)),
                     recording='test',fps=hz,pwm=20,diameter=1.,arena=[0.,0.,10.],
                     ids=[17,23],quality_flags=np.zeros(xy.shape[:2],int))
            receipt=score_file(file,root)
            assert receipt['identity']=='no known input match' and receipt['implementation_matches_freeze']
            if variant=='no_gate':
                assert receipt['events']==0 and not (root/'scores.csv').exists()
                continue
            assert receipt['events']==1
            d=pd.read_parquet(root/'predictions.parquet').set_index(['axis','horizon_ms']).sort_index()
            np.testing.assert_allclose(d.loc['n','observed'],D_MM*(-.3 if variant=='linear' else .5))
            np.testing.assert_allclose(d.loc['t','observed'],0)
            outputs.append(d[MODELS].to_numpy())
        np.testing.assert_allclose(*outputs,rtol=0,atol=1e-12)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    d=development()
    frozen=freeze(d)
    table=compact_comparison(d)
    print(table.query("design=='cross-family'")[['family','axis','horizon_ms','gain_mm_s','compact_minus_full']].round(2).to_string(index=False))
    if args.replay:
        replay(d,frozen)
