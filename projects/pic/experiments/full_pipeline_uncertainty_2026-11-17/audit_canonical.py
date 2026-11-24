from bridge import *

def main():
    d=read(ROOT/'pilot_reproducibility.json');links={str(ROOT/'pilot_reproducibility.json'):sha(ROOT/'pilot_reproducibility.json')};errors=[]
    for r in d['rows']:
        path=ROOT/'pilot_replays'/f"seed{r['seed']}_rep{r['replay']}.npz";assert sha(path)==r['array_sha256'];links[str(path)]=sha(path)
        with np.load(path) as z:assert {v:R.digest(z[v]) for v in z.files}==r['pilot_sha256']
    for r in d['comparisons']:
        path=ROOT/'pilot_replays'/f"seed{r['seed']}_rep{r['replay']}.npz";canonical=Path(r['canonical']);fpath=ROOT/'factorial_frozen'/f"ks_positive_seed{r['seed']}.json";f=read(fpath);links[str(fpath)]=sha(fpath)
        with np.load(path) as z,np.load(canonical) as zz:
            assert {v:R.digest(zz[v]) for v in zz.files}==f['pilot_sha256']['correlated'];error=max(float(np.max(np.abs(z[v]-zz[v]))/np.max(np.abs(zz[v]))) for v in z.files);assert error<=1e-9;errors.append(error)
    dump(ROOT/'audit_canonical.json',dict(status='PASS',reconstructions=12,max_relative_difference=max(errors),source_sha256=sources('audit_canonical.py','pilot_roundoff_protocol.txt'),input_sha256=links));print('PASS',max(errors),flush=True)
if __name__=='__main__':main()
