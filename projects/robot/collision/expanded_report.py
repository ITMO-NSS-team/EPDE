"""Pool coefficients for the common five-term law and plot recording-level transfer scores."""
import json

from expanded_data import DEST, fingerprint
from expanded_terms import IDS, V, WARMUP, columns, history
from physics import LAG, arena_for, fit_self, recordings
from discovery import write_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pooled_law(terms, front=None, *, directory=DEST, table_filename='merged_pareto_equations.csv', paired=None):
    observed, full = ({},recordings()) if paired is None else paired
    tracks = [tr.part(0,len(tr.pos)*3//4) for tr in full.values()]
    arenas = arena_for(tracks)
    lookup, common = fit_self(tracks,arenas)
    gs = []
    union = list(terms)
    if front is not None:
        union += sorted({t for s in front.terms for t in s.split('+')} - set(terms))
    ix = [IDS.index(term) for term in union]
    for tr in tracks:
        rows = np.arange(WARMUP,len(tr.pos)-LAG,LAG)
        measured = observed[tr.tag].part(0,len(tr.pos)) if tr.tag in observed else None
        mem = history(tr,rows,measured)
        g = np.zeros((len(ix)+1,len(ix)+1))
        count = 0
        for start in range(0,len(rows),32):
            slots = np.arange(start,min(start+32,len(rows)))
            x,base,y,ok,*_ = columns(tr,rows[slots],slots,mem,(lookup,common,arenas),measured)
            a = np.concatenate((x[...,ix],(y-base)[...,None]),axis=-1)[ok].reshape(-1,len(ix)+1)
            g += a.T@a
            count += len(a)
        gs.append(g/count)
    g = np.mean(gs,axis=0)
    def solve(names):
        ids = [union.index(n) for n in names]
        scale = np.sqrt(np.diag(g)[ids])
        b = np.linalg.solve(g[np.ix_(ids,ids)]/scale[:,None]/scale[None,:],g[ids,-1]/scale)/scale
        return dict(zip(names,map(float,b)))
    result = dict(terms=solve(terms),V_px_per_sec=V,dt_seconds=.4,
            model='base + V * sum of dimensionless weighted vectors',split='first 75% of all five recordings',
            recordings=[tr.tag for tr in tracks],base_persistence_wall=common.tolist(),
            lookup={f'{k[0]}/{k[1]}':v.tolist() for k,v in lookup.items()},
            arenas={str(k):v.tolist() for k,v in arenas.items()})
    write_json(directory/'pooled_common_law.json',result)
    if front is not None:
        front.assign(pooled_coefficients=[json.dumps(solve(s.split('+'))) for s in front.terms]).to_csv(
            directory/table_filename,index=False)
    return result


def run():
    data = pd.read_csv(DEST/'evaluated.csv.gz')
    w = pd.read_csv(DEST/'validation_selected.csv').query('budget==10')
    base = data[data.baseline]
    additions = pd.read_csv(DEST/'one_addition.csv')
    shared = additions.loc[additions.groupby('held').validation.idxmin()]
    assert shared.addition.nunique()==1 and len(shared)==5
    terms = shared.terms.iloc[0].split('+')
    merged = pd.read_csv(DEST/'merged_pareto.csv')
    pooled = pooled_law(terms,merged)
    common = pd.read_csv(DEST/'common_structures.csv')
    native = pd.read_csv(DEST/'nsga2_evaluated.csv.gz')
    tags = sorted(data.held.unique())
    sets = [('Four-term benchmark',base),('Common five-term law',shared),
            ('Catalog contact terms',w[w.scope=='contact_catalog']),
            ('Expanded contact terms',w[w.scope=='contact']),
            ('Contact and robot controls',w[w.scope=='joint'])]
    fig,(ax,front) = plt.subplots(1,2,figsize=(11,4.8),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
    for (label,table),marker in zip([sets[i] for i in [0,1,3,4]],['o','s','^','D']):
        ax.plot(np.arange(5),table.set_index('held').loc[tags].r2,marker=marker,label=label,linewidth=1.4)
    ax.set_xticks(np.arange(5),[t.replace('_','\n') for t in tags])
    ax.set_ylabel('Held-out velocity R²')
    ax.legend(frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.2)
    front.scatter(common.dyad.clip(lower=1e-8),common.wape,s=9,c='#9da8b6',alpha=.32)
    line = merged.sort_values('dyad')
    front.plot(line.dyad,line.wape,'o-',color='#99340d',markersize=4,label='Nondominated common supports')
    front.set_xscale('log')
    front.set_xlim(line.dyad.min()*.75,line.dyad.max()*1.5)
    front.set_ylim(line.wape.min()-.008,line.wape.max()+.012)
    front.set_xlabel('Mean EPDE dyad instability (log scale)')
    front.set_ylabel('Mean training-screen WAPE')
    front.legend(frameon=False,fontsize=8)
    fig.savefig(DEST/'expanded_front.png',dpi=190)
    plt.close(fig)

    rows = []
    for label,table in sets+[(f'Focused grammar, {i} factor cap',w[w.scope==f'native_grammar_{i}']) for i in (1,2,3)]:
        rows.append(dict(model=label,mean_r2=table.r2.mean(),worst_r2=table.r2.min(),rmse=table.rmse.mean(),
                         normal_rmse=table.normal.mean(),tangent_rmse=table.tangent.mean()))
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False)
    for held in tags:
        candidates = data[(data.held==held)&data.eligible]
        f = candidates[candidates.epde_front]
        for _,r in f.iterrows():
            assert not (((candidates.wape<=r.wape)&(candidates.dyad<=r.dyad))&
                        ((candidates.wape<r.wape)|(candidates.dyad<r.dyad))).any()
    assert data.groupby('held')['size'].max().min()==10
    assert native.groupby(['library','held']).terms.apply(lambda terms:any('Gwall*Gspin' in s for s in terms)).loc['focused'].all()
    assert set(data[data['size']==1].terms)==set(IDS)
    summary = dict(recordings=tags,terms=len(IDS),evaluated_fold_supports=len(data),
                   common_eligible=len(common),merged_front=len(merged),
                   native_fold_supports=len(native),native_three_factor_fold_supports=int(native.terms.str.contains('Gwall*Gspin',regex=False).sum()),
                   best_single_addition=shared.addition.iloc[0],coefficients=pooled['terms'],input_digest=fingerprint())
    write_json(DEST/'summary.json',summary)
    lines = ['# Expanded robot collision search','',
        'Five recording-level holdouts predict the mean two-dimensional velocity over the next 0.4 s. The four training recordings supply separate 75% calibration and 25% validation segments. Each segment rebuilds history from its own start. The first 50 frames provide history, and unresolved initial contact histories are masked. Each recording receives equal weight.',
        '', '## Search and errors','',
        'The 185 translation candidates in the workbook, 81 three-scalar pair products, and 33 robot-modulated contact vectors give 299 columns. The generated pair products multiply distance, a closing/release/drive factor and a signed or geometric factor before summation. The two scalar robot modifiers are smooth wall proximity and absolute focal rim speed. The 14 yaw candidates have a different target. Eight further candidates need instrumentation.',
        '', 'The native EPDE NSGA-II jobs used one seed (923701), 16 individuals, 12 generations, up to three native factors and ten RHS terms. Each holdout ran both a broad pool and a focused 11-vector pool. Each physical RHS term contains one contact or control vector. The two optional modifiers are available for the 11 designated contact vectors. The directional equations have zero scalar intercept. Native receipts archive screen fits and final populations.',
        '', 'The deterministic companion search ranks one-term extensions by training squared error with a beam of eight and up to ten extensions per parent. It includes every single term, the four-term benchmark with every single addition, and focused searches on restricted libraries. EPDE scores candidate structures on 512 stratified component rows per training recording using WAPE and dyad coefficient instability. Full training segments supply the coefficients used for the validation and transfer tables. Native screen-fit errors are also available separately. Validation selects each fold’s model before its held-out score is used.',
        '',f'The search evaluated {len(data):,} fold–structure combinations. The merged Pareto front minimizes the two mean EPDE training objectives over the {len(common):,} eligible structures independently evaluated in all five folds. It contains {len(merged)} structures. The finite search covers a subset of possible equations.',
        '', '| Model | Mean R² | Worst R² | Velocity RMSE | Pair-normal RMSE | Pair-tangent RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {model} | {mean_r2:.3f} | {worst_r2:.3f} | {rmse:.2f} | {normal_rmse:.2f} | {tangent_rmse:.2f} |'.format(**r))
    lines.extend(['','## Held-out velocity R² by recording','',
                  '| Omitted recording | Four terms | Common five terms | Expanded contact | With robot controls |',
                  '|---|---:|---:|---:|---:|'])
    for tag in tags:
        values = [sets[i][1].set_index('held').loc[tag,'r2'] for i in (0,1,3,4)]
        lines.append('| '+tag+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines.extend(['','## Merged Pareto front','',
        'Each row gives one common equation structure. WAPE and dyad are minimized. Held-out R² and velocity RMSE describe transfer. The term dictionary supplies the vector and weight for every ID. The full evaluation table includes fitted coefficients for each recording holdout.',
        '', '| Terms | WAPE | Dyad | Mean held-out R² | Worst held-out R² | RMSE (px/s) |',
        '|---|---:|---:|---:|---:|---:|'])
    for r in merged.sort_values('wape').itertuples():
        lines.append(f'| `{r.terms}` | {r.wape:.4f} | {r.dyad:.3g} | {r.r2:.3f} | {r.worst_r2:.3f} | {r.rmse:.2f} |')
    lines.extend(['','RMSE values are in camera pixels per second. Velocity RMSE and R² use stacked x and y components. Pair errors project relative velocity on the current contact basis for neighbors inside 253 px. The compact law won the single-addition validation comparison in all five folds. Its cross-validation scores use fold-specific training coefficients. The pooled fit for deployment uses the first 75% of all five recordings.',
                  '', '## Common law','',r'\[\widehat{\mathbf v}_i=\mathbf v_{i,\mathrm{base}}+V\sum_{j\,:\,0<r_{ij}<253}\left('+
                  f"{pooled['terms']['C002']:.5f}q_{{ij}}+{pooled['terms']['C020']:.5f}c_{{ij}}{pooled['terms']['C081']:+.5f}h_i+{pooled['terms']['C101']:.5f}D_j+{pooled['terms']['C102']:.5f}D_i"+
                  r'\right)\mathbf n_{ij}.\]','',
                  'The normal points from neighbor j to focal robot i. The dimensionless quantities are cutoff closeness q, approach speed c, inward focal heading h, and the components of each calibrated free propulsion vector directed into the contact, D_i and D_j. V is the cap diameter divided by 0.4 s. The additive base contains calibrated free propulsion, velocity persistence and the wall term.',
                  '', 'Past-hit terms use completed 0.12 s pre/post onset velocity differences with 0.4 s decay and a 1.2 s cutoff. These are kinematic proxies for influences at a contact onset. The fixed dimensionless scales are q0=c0=s0=z0=0.1, x0=1 and eps=0.01. Setup-level generalization is measured across these five recordings, several of which share robot IDs.',
                  '', '## Reproduction','', '```powershell','python projects/robot/collision/run.py --stage expanded','```',''])
    (DEST/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=1),flush=True)


if __name__=='__main__':
    run()
