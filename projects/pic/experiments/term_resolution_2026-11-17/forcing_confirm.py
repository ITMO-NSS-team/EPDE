from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import projected as experiment

ROOT=experiment.ROOT
OUT=ROOT/'forcing'


def run(job):
    experiment.OUT=OUT
    experiment.run(job)


def main():
    OUT.mkdir(exist_ok=True)
    source=experiment.SOURCE/'results'/'kdv_cossin_iid_seed57.json'
    d=json.loads(source.read_text())
    assert experiment.sha(source.with_suffix('.npz'))==d['array_sha256']
    ref=next(c for c in d['candidates'] if c['label']=='reference')
    column=ref['xi'][2]
    assert column in d['deterministic'] and 'cos(t)sin(x)' in str(d['labels'][column])
    with np.load(source.with_suffix('.npz')) as z:
        directions={}
        for b in experiment.BANKS:
            signal=z[b+'_pilot'][:,column]
            np.testing.assert_array_equal(signal,z[b+'_observed'][:,column])
            directions[b]=signal/np.linalg.norm(signal)
    np.savez(OUT/'directions.npz',vectors=experiment.vectors(d),**directions)
    frozen=dict(protocol=experiment.sha(ROOT/'forcing_protocol.txt'),base_protocol=experiment.sha(ROOT/'projection_protocol.txt'),
                training_sha256=experiment.sha(source),directions=experiment.sha(OUT/'directions.npz'),
                pilot_directions=experiment.sha(ROOT/'projected'/'directions.npz'),
                source=experiment.sha(ROOT/'forcing_confirm.py'),base_source=experiment.sha(ROOT/'projected.py'))
    (OUT/'lock.json').write_text(json.dumps(frozen,indent=2))
    jobs=[(p,s) for p in ('iid','positive','negative') for s in range(4705,4709)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(run,jobs))


if __name__=='__main__':
    main()
