"""Report extreme fitted noise parameters; never reject or replace a draw."""
from bridge import *
from math import ceil


def quantile(values):return float(np.sort(values)[ceil(.95*(len(values)+1))-1])


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==69;rows=[];fits=0
    for path in files:
        d=read(path)
        for j,fit in enumerate(d['refits']):
            for v,model in fit['model'].items():
                fits+=1;anchor=d['model'][v];ratio=model[0]/anchor[0]
                if ratio<=2 and max(abs(np.array(model[1])))<=.95:continue
                with np.load(path.with_suffix('.npz')) as z:
                    scores={}
                    for mode in ('fixed_pilot','refit_pilot'):
                        scores[mode]={}
                        for b in ('local','broad','joint'):
                            values=np.maximum(z[mode+'_local_scores'],z[mode+'_broad_scores']) if b=='joint' else z[mode+'_'+b+'_scores']
                            old=quantile(values);left=np.delete(values,j);new=quantile(left)
                            scores[mode][b]=dict(score=float(values[j]),rank_ascending=int(np.sum(values<values[j])+1),
                                                original_cutoff=old,leave_one_out_cutoff=new,relative_cutoff_change=new/old-1)
                rows.append(dict(system=d['system'],profile=d['profile'],seed=d['seed'],draw=j,variable=v,
                                 anchor_model=anchor,refitted_model=model,sigma_ratio=ratio,fit_status=fit['info'][v]['status'],
                                 projected_gradient=fit['info'][v]['projected_gradient'],
                                 temporal_innovation_sigma_ratio=ratio*np.sqrt((1-model[1][0]**2)/(1-anchor[1][0]**2)),
                                 correlation_length_over_record=-1/np.log(abs(model[1][0]))/d['fit_info'][v]['shape'][0],
                                 scores=scores))
    out=dict(status='COMPLETE',datasets=len(files),bootstrap_variable_fits=fits,flagged_fits=len(rows),rows=rows,
             diagnostic_rule='Flag bootstrap sigma/anchor sigma >2 or any absolute fitted noise correlation >0.95. No filtering.',
             interpretation='Large marginal variance with a nearly unit correlation can leave innovation scale '
             'similar while producing very weak information about level variance in curvature contrasts. '
             'This is a statistical-identification warning even when the optimizer score is small. '
             'Leave-one-out cutoffs are attribution diagnostics only; every original fit/draw remains in the study.',
             source_sha256=sources('noise_tail_diagnostic.py'),input_sha256={str(p):sha(p) for p in files})
    dump(ROOT/'noise_tail_diagnostic.json',out);print(json.dumps({k:v for k,v in out.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__=='__main__':main()
