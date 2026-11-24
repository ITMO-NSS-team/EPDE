"""Check identical held-out observations and summarize the hard-cap discovery experiment."""
import json

from expanded_data import DEST, PROJECTED, fingerprint
from expanded_native import RHS, SEED
from expanded_report import pooled_law
from expanded_search import error, fitted, metrics, stored
from expanded_terms import IDS, TERMS, WARMUP, history
from physics import CUTOFF, DT, LAG, OUT, geometry, rotate
from projected_data import projected_tracks
from discovery import write_json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANCHOR = ('C012','C108','C109','C111','C180','C183','C025','C156')
MEASURES = ['r2','rmse','normal','tangent']


def family_extensions(held_tags):
    """Allow one catalog term to extend the previous shared four-term contact law."""
    base = [IDS.index(k) for k in ANCHOR[:4]]
    families = ['Spin direction and relative spin','Earlier hits from other robots',
                'Simultaneous neighbors','Same-pair contact history']
    rows = []
    for held in held_tags:
        train = np.mean([z['g'] for z in stored(held,'train')],axis=0)
        valid = np.mean([z['g'] for z in stored(held,'valid')],axis=0)
        test, = stored(held,'test')
        refit = fitted(train,base)
        baseline = metrics(test,base,refit)
        for family in families:
            candidates = [(error(valid,base,refit),'none',base,refit)]
            for j,term in enumerate(TERMS):
                if term[1] == family:
                    ix = [*base,j]
                    b = fitted(train,ix)
                    if b is not None:
                        candidates.append((error(valid,ix,b),term[0],ix,b))
            validation,term,ix,b = min(candidates,key=lambda entry:entry[:2])
            score = metrics(test,ix,b)
            rows.append(dict(held=held,family=family,addition=term,validation=validation,**score,
                delta_r2=score['r2']-baseline['r2'],delta_normal_rmse=score['normal']-baseline['normal']))
    return pd.DataFrame(rows)


def check_observations():
    previous = OUT/'preprocessed'/'heldout_raw_target.csv'
    reference_scores = (pd.read_csv(previous).query("training=='projected' and test_mask=='all'").set_index('held')
                        if previous.exists() else None)
    ix = [IDS.index(k) for k in ANCHOR]
    verified = []
    for path in sorted((DEST/'statistics').glob('*__test.npz')):
        held = path.name.split('__')[0]
        train = np.mean([z['g'] for z in stored(held,'train')],axis=0)
        test, = stored(held,'test')
        b = fitted(train,ix)
        scores = metrics(test,ix,b)
        with np.load(OUT/'expanded_factors'/'statistics'/f'{held}__{held}__test.npz') as raw:
            assert test['count'] == raw['count'] and test['pairs'] == raw['pairs']
            np.testing.assert_allclose(test['variance'],raw['variance'],rtol=1e-12)
        if reference_scores is not None:
            reference = reference_scores.loc[held]
            assert test['count']==2*reference.robot_origins and test['pairs']==reference.pair_origins
            np.testing.assert_allclose([scores[k] for k in MEASURES],
                reference[['r2','rmse','normal_rmse','tangent_rmse']].to_numpy(float),rtol=1e-7)
            np.testing.assert_allclose(b,[json.loads(reference.coefficients)[k] for k in ANCHOR],rtol=1e-6)
        verified.append(dict(held=held,robot_origins=int(test['count'])//2,pair_origins=int(test['pairs']),
                             **scores))
    assert len(verified)==5
    return pd.DataFrame(verified)


def pair_variances(observations, verified):
    variances = {}
    counts = verified.set_index('held').pair_origins
    for tag,tr in observations.items():
        rows = np.arange(WARMUP,len(tr.pos)-LAG,LAG)
        ready = history(tr,rows)[-1]
        d,axes = geometry(tr.pos[rows])
        f,i,j = np.nonzero(np.triu((d>0)&(d<CUTOFF),k=1))
        keep = ready[f,i]&ready[f,j]
        f,i,j = f[keep],i[keep],j[keep]
        assert len(f)==counts[tag]
        y = (tr.pos[rows+LAG]-tr.pos[rows])/DT
        relative = y[f,i]-y[f,j]
        n = axes[f,i,j]
        variances[tag] = [np.var((relative*a).sum(1)) for a in (n,rotate(n))]
    return pd.DataFrame.from_dict(variances,orient='index',columns=['normal','tangent'])


def fold_comparison(verified, common_terms, variances):
    rows = []
    old = OUT/'expanded_factors'
    anchor_name = '+'.join(ANCHOR)
    shared_ix = [IDS.index(k) for k in common_terms]
    anchor_ix = [IDS.index(k) for k in ANCHOR]
    for name, directory in [('raw',old),('projected',DEST)]:
        selected = pd.read_csv(directory/'validation_selected.csv')
        selected = selected[(selected.budget==RHS)&selected.scope.isin(['contact','joint'])]
        rows.append(selected[['held','scope','terms',*MEASURES]].rename(columns={'scope':'selection'}).assign(geometry=name))
        for held in verified.held:
            files = sorted((directory/'statistics').glob(f'{held}__*__train.npz'))
            assert len(files)==4
            matrices = []
            for path in files:
                with np.load(path) as z:
                    matrices.append(z['g'])
            train = np.mean(matrices,axis=0)
            with np.load(directory/'statistics'/f'{held}__{held}__test.npz') as test:
                scores = metrics(test,shared_ix,fitted(train,shared_ix))
                anchor = metrics(test,anchor_ix,fitted(train,anchor_ix))
                base = metrics(test,[],[])
            rows.append(pd.DataFrame([dict(held=held,**scores,terms='+'.join(common_terms),
                                     selection='min_wape_shared',geometry=name),
                                      dict(held=held,**base,terms='',selection='base',geometry=name),
                                      dict(held=held,**anchor,terms=anchor_name,selection='fixed_eight',geometry=name)]))
    folds = pd.concat(rows,ignore_index=True)
    for axis in ('normal','tangent'):
        folds[axis+'_r2'] = 1-folds[axis]**2/folds.held.map(variances[axis])
    folds.to_csv(DEST/'comparison_by_recording.csv',index=False)
    summary = folds.groupby(['selection','geometry'],as_index=False).agg(
        mean_r2=('r2','mean'),worst_r2=('r2','min'),rmse=('rmse','mean'),
        normal_rmse=('normal','mean'),tangent_rmse=('tangent','mean'),
        normal_r2=('normal_r2','mean'),tangent_r2=('tangent_r2','mean'))
    summary.to_csv(DEST/'comparison.csv',index=False)
    return summary, folds


def write_front_report(front, common, summary, candidate_count):
    equations = pd.read_csv(DEST/'cross_recording'/'projected_pareto_equations.csv')
    lines = [
        '# Collision laws across five recordings', '',
        'Each equation predicts mean velocity over the next 0.4 s. Its coefficient vector was refitted on four recordings for each held-out score. The displayed coefficients are a separate fit to the training prefixes of all five recordings.', '',
        r'For each row, $\widehat{\mathbf v}_i = \mathbf b_i + V\,F_i$. The base $\mathbf b_i$ combines calibrated propulsion, velocity persistence and wall motion. $V$ is one cap diameter per forecast interval. Each $\Phi_k$ is a dimensionless vector from the term dictionary. A pair term sums its weighted direction over neighbors within 253 camera pixels.', '',
        f'Both geometries were scored on the same {candidate_count:,} archived structures and reference equations. The {len(front)} rows below minimize mean training-screen WAPE and dyad coefficient instability among {len(common):,} structures eligible in all five projected-data folds. The velocity errors use raw measured displacements. Each recording has equal weight. The structure pool uses evidence from all five recordings. The fold-selected contact and joint comparisons use proposals generated inside each training fold.', '',
        '| Correction $F_i$ | WAPE | Dyad | Mean held-out R² | Worst held-out R² | RMSE (px/s) |',
        '|---|---:|---:|---:|---:|---:|']
    used = set()
    for row in equations.itertuples():
        summands = []
        for term,b in json.loads(row.pooled_coefficients).items():
            vector,*gates = term.split('*')
            used.add(vector)
            sub = vector[1:] if vector.startswith('C') else vector
            notation = ''.join({'Gspin':r'\lvert w_i\rvert','Gwall':r'g_{\rm wall}'}[g] for g in gates)
            summands.append(f'{b:+.4g}'+notation+r'\Phi_{'+sub+'}')
        expr = ' '.join(summands).lstrip('+')
        lines.append(f'| ${expr}$ | {row.wape:.4f} | {row.dyad:.4f} | {row.r2:.4f} | {row.worst_r2:.4f} | {row.rmse:.2f} |')
    lines.extend(['', '## Terms on this front', '',
        'The weight is evaluated before summing neighbors. The normal points from the neighbor to the focal robot. The tangent is its 90° counterclockwise rotation. Here rho is projected center distance in cap diameters, rhoc is the cutoff in cap diameters, and q is cutoff minus distance divided by 230 pixels. D_i and D_j are the positive parts of the robots’ propulsion components toward one another. T_i is the focal robot’s signed tangential propulsion component. The propulsion components are divided by V.', '',
        '| Term | Family | Weight | Direction |','|---|---|---|---|'])
    for term in sorted(used):
        _,family,_,weight,direction = TERMS[IDS.index(term)]
        lines.append(f'| {term} | {family} | `{weight}` | {direction} |')
    lines.extend(['','## Matched transfer errors','',
        'Contact and joint models are selected separately within each fold by validation RMSE. The shared minimum-WAPE row uses the same equation structure in every fold. Pair errors project the relative velocity of observed neighbors onto their observed normal or tangent at the forecast origin.', '',
        '| Selection | Geometry | Robot R² | Pair-normal R² | Pair-tangent R² | Normal RMSE (px/s) |',
        '|---|---|---:|---:|---:|---:|'])
    for row in summary.itertuples():
        lines.append(f'| {row.selection} | {row.geometry} | {row.mean_r2:.4f} | {row.normal_r2:.4f} | {row.tangent_r2:.4f} | {row.normal_rmse:.2f} |')
    lines.extend(['','The native search used one seed, two libraries, 16 individuals and 12 generations per recording holdout. The companion beam search used the same 299-column dictionary. Two geometric columns have zero variation after the hard-cap correction. All original forecast rows passing the history warmup and availability checks were retained. Raw measured cap crossings determine contact state. The observed gap also supplies apparent-overlap terms.'])
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    if not PROJECTED:
        raise ValueError('Run with --stage projected.')
    from cross_recording_front import run as refit_all
    count = refit_all(['+'.join(ANCHOR[:6]),'+'.join(ANCHOR)])
    report(count)


def report(candidate_count):
    directory = DEST/'cross_recording'
    common = pd.read_csv(directory/'projected_common.csv.gz')
    front = pd.read_csv(directory/'projected_pareto.csv').sort_values('wape')
    # Direct dominance check on every claimed front point against the entire eligible set.
    objectives = common[['wape','dyad']].to_numpy()
    for point in front[['wape','dyad']].to_numpy():
        assert not np.any((objectives<=point).all(1)&(objectives<point).any(1))
    assert common.folds.eq(5).all() and not common[['wape','dyad',*MEASURES]].isna().any().any()
    assert set(front.terms)==set(common.loc[common.merged_front,'terms'])
    terms = front.iloc[0].terms.split('+')
    paired = projected_tracks()
    local_front = pd.read_csv(DEST/'merged_pareto.csv')
    pooled_law(local_front.iloc[0].terms.split('+'),local_front,paired=paired)
    pooled = pooled_law(terms,front,directory=directory,table_filename='projected_pareto_equations.csv',paired=paired)
    verified = check_observations()
    summary, folds = fold_comparison(verified,terms,pair_variances(paired[0],verified))
    family_extensions(verified.held).to_csv(DEST/'family_extensions.csv',index=False)
    write_front_report(front,common,summary,candidate_count)
    explored = pd.read_csv(DEST/'evaluated.csv.gz')
    native = pd.read_csv(DEST/'nsga2_evaluated.csv.gz')
    test_counts = verified[['held','robot_origins','pair_origins']].set_index('held').to_dict('index')
    write_json(DEST/'summary.json',dict(input_digest=fingerprint(),seed=SEED,terms=len(IDS),
        max_factors=3,max_rhs=RHS,evaluated_fold_structures=len(explored),unique_structures=explored.terms.nunique(),
        native_fold_library_structures=len(native),cross_recording_candidates=candidate_count,
        independent_proposal_front=len(local_front),common_eligible=len(common),merged_front=len(front),
        matched_counts=test_counts,minimum_wape_terms=terms,pooled_coefficients=pooled['terms'],
        training='Projected next-window component velocities. Raw gap sets contact onset, clearance and apparent overlap.',
        test='Original observed next-window component velocities. Pair errors use raw neighbors and raw pair axes.',
        objectives='Minimize mean training-screen WAPE and mean dyad coefficient instability. Equal recording weights.',
        selection='Fold recommendations minimize validation RMSE on training recordings. Shared front is exploratory across five folds.'))
    fig,(ax,transfer) = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    ax.scatter(common.wape,common.dyad,s=8,alpha=.26,color='0.52',label='Eligible in all five folds')
    ax.plot(front.wape,front.dyad,marker='o',lw=1.4,ms=4,color='#146a75',label='Nondominated')
    span = front.wape.max()-front.wape.min()
    ax.set(xlabel='Mean training WAPE',ylabel='Mean dyad instability',title='Shared equations near the Pareto front',
           xlim=(front.wape.min()-.05*span,front.wape.max()+.08*span),
           ylim=(front.dyad.min()*.8,front.dyad.max()*1.3))
    ax.set_yscale('symlog',linthresh=.1)
    ax.legend(fontsize=8)
    for label,marker in [('raw','o'),('projected','x')]:
        s = folds[(folds.geometry==label)&(folds.selection=='min_wape_shared')].sort_values('held')
        transfer.plot(range(len(s)),s.r2,ls='none',marker=marker,ms=9,mfc='none',label=label.capitalize())
    transfer.set_xticks(range(len(s)),[t.replace('_','\n') for t in s.held],fontsize=8)
    transfer.set(ylabel='Held-out velocity R²',title='Same minimum-WAPE structure')
    transfer.legend(fontsize=8)
    fig.savefig(DEST/'pareto_transfer.png',dpi=160)
    plt.close(fig)
    print(summary.round(6).to_string(index=False),flush=True)
