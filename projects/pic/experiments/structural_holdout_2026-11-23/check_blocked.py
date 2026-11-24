"""Change a held-out raw slab and verify the fitted search archive is identical."""
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from blocked_search import discover,fold_path,slabs
from experiment import ROOT,live,np
from assess_swap import read_archive


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        sys.exit(subprocess.run([sys.executable,__file__],env=dict(os.environ,PYTHONHASHSEED='0')).returncode)
    name,seed,fold='duffing',923301,1
    problem=live.load_data(name)
    _,parts=slabs(live.R.axes_of(problem))
    cuts,_=parts[fold]
    original=type(problem).noisy
    def perturb(instance,level,s):
        raw=original(instance,level,s)
        for values in raw.values():
            values[cuts]=100*np.random.default_rng(8731).normal(size=values[cuts].shape)
        return raw
    with TemporaryDirectory() as tmp:
        dest=Path(tmp)
        with patch.object(type(problem),'noisy',perturb):
            discover(name,seed,fold,dest)
        base=fold_path(name,seed,fold)
        a,b=read_archive(base),read_archive(dest)
        for key in ('settings','variables','plan','train_hashes','trig','labels','points','systems','system_events'):
            assert a[key]==b[key],key
        ma,mb=[json.loads((d/'fold.json').read_text()) for d in (base,dest)]
        for i in range(4):
            assert (ma['sample_hashes'][i]==mb['sample_hashes'][i])==(i!=fold)
    (ROOT/'blocked_exclusion_check.json').write_text(json.dumps(dict(dataset=name,seed=seed,held_fold=fold,
        unchanged_fitted_points=len(a['points']),unchanged_actual_systems=len(a['systems']),held_hash_changed=True)))
