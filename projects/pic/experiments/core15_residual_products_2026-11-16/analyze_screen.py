"""Audit common dyad partitions and summarize every equation separately."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import run_screen as R
import numpy as np
from epde.interface.search_config import active_config, set_active_config


def common_width(width):
    config = active_config()
    set_active_config(replace(config, search_space=replace(config.search_space,
        equation_terms_max_number=width,system_rps_mode='repair')))
    assert all(R.SV._reference_width(p)==width for p in range(width+1))


def audit(path):
    x = json.loads(path.read_text(encoding='utf-8'))
    problem = R.datasets.load(x['system'])
    cs,labels = R.panel(problem)
    assert [c['id'] for c in cs] == [c['id'] for c in x['candidates']]
    width = max(len(c['terms']) for c in cs)
    common_width(width)
    terms,opaque = R.symbols(problem,labels)
    plan = {k:tuple(v) for k,v in x['base_plan'].items()}
    assert R.SV.row_floor('dyad',plan['shape'],width) is None
    bank = R.LocalBank(R.axes_of(problem),terms,opaque,plan)
    raw = problem.noisy(x['noise_percent'],x['seed'])
    mat = bank.matrix(raw)
    instability = [float(np.sum(R.SV.dyad_scores(mat[:,c['xi']],mat[:,c['yi']],np.ones(len(mat)),
                   plan['shape'],fit_intercept=False))) if c['xi'] else 0. for c in cs]
    x['original_instability'] = x['instability']
    x['instability'] = instability
    x['partition_audit'] = dict(width=width,why='Same production dyad partition for every candidate',
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for method,counts in x['aggregates'].items():
        for count,entry in counts.items():
            for score in ('cross','ordinary','oracle','uncorrected_cross'):
                entry[score]['dominance'] = R.dominance(entry[score]['scores'],instability,cs)
    return x


def frontier(scores, instability, candidates, slot):
    ids = [i for i,c in enumerate(candidates) if c['slot']==slot]
    front = [i for i in ids if not any(j!=i and scores[j]<=scores[i] and instability[j]<=instability[i]
        and (scores[j]<scores[i] or instability[j]<instability[i]) for j in ids)]
    return [candidates[i]['id'] for i in front if candidates[i]['registered']]


def summarize(results):
    rows = []
    for x in results:
        cs = x['candidates']
        for method,counts in x['aggregates'].items():
            for count,entry in counts.items():
                for slot in sorted({c['slot'] for c in cs}):
                    ref = next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                    row = dict(system=x['system'],seed=x['seed'],slot=slot,variable=cs[ref]['variable'],
                               bandwidth=method,draws=int(count),reference_I=x['instability'][ref],
                               clean_ref_fit_energy=entry['clean_fit_energy'][ref])
                    for score in ('cross','ordinary','oracle','uncorrected_cross'):
                        d = entry[score]['dominance'][slot]
                        row[score+'_D'] = entry[score]['scores'][ref]
                        row[score+'_dominators'] = '|'.join(d['unregistered'])
                        row[score+'_registered_dominators'] = '|'.join(d['registered'])
                        row[score+'_valid_front'] = '|'.join(frontier(entry[score]['scores'],x['instability'],cs,slot))
                    rows.append(row)
    return rows


def report(results,rows,output):
    with (output/'per_equation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary = {}
    for method in ('gcv','loo'):
        for count in sorted({r['draws'] for r in rows}):
            rs = [r for r in rows if r['bandwidth']==method and r['draws']==count]
            summary[f'{method}_{count}'] = dict(equation_trials=len(rs),
                cross_reference_not_dominated=sum(not r['cross_dominators'] and not r['cross_registered_dominators'] for r in rs),
                cross_without_unregistered_dominator=sum(not r['cross_dominators'] for r in rs),
                ordinary_without_unregistered_dominator=sum(not r['ordinary_dominators'] for r in rs),
                cross_valid_on_front=sum(bool(r['cross_valid_front']) for r in rs),
                ordinary_valid_on_front=sum(bool(r['ordinary_valid_front']) for r in rs),
                oracle_valid_on_front=sum(bool(r['oracle_valid_front']) for r in rs))
    R.dump(output/'summary.json',summary)
    # A figure is an exportable scientific artifact; no interactive app needed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    maxb = max(r['draws'] for r in rows)
    ordered = [(name,slot) for name in R.datasets.names('core') for slot in
               sorted({r['slot'] for r in rows if r['system']==name})]
    fig,axes = plt.subplots(1,2,figsize=(10,7),sharey=True)
    for ax,method in zip(axes,('gcv','loo')):
        for offset,score,color in [(-.16,'ordinary','#9a9a9a'),(.16,'cross','#1679a8')]:
            values = [sum(bool(r[score+'_valid_front']) for r in rows if r['system']==name and
                          r['slot']==slot and r['draws']==maxb and r['bandwidth']==method) for name,slot in ordered]
            ax.barh(np.arange(len(ordered))+offset,values,height=.3,label=score,color=color)
        ax.set_title(method.upper()+' bandwidth')
        ax.set_xlim(0,3.2);ax.set_xticks([0,1,2,3]);ax.set_xlabel('Seeds retaining a registered law / 3')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(ordered)),[f'{n} [{s}]' for n,s in ordered])
    axes[0].invert_yaxis();axes[1].legend(loc='lower right')
    fig.suptitle('Core 15: finite-panel structural survival at 5% noise\nShared dyad axis; independent residual cross-product vs matched energy')
    fig.tight_layout();fig.savefig(output/'structural_survival.png',dpi=180)
    fig.savefig(output/'structural_survival.svg');plt.close(fig)
    return summary


def main():
    if R.os.name=='nt':
        R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=R.ROOT/'results')
    parser.add_argument('--output',type=Path,default=R.ROOT/'audited')
    parser.add_argument('--draws',type=int,default=32)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in sorted(args.input.glob(f'*_b{args.draws}.json')):
        x=audit(path);results.append(x);R.dump(args.output/path.name,x)
    rows=summarize(results)
    print(json.dumps(report(results,rows,args.output),indent=2))
    print('Audited cases:',len(results))


if __name__=='__main__':main()
