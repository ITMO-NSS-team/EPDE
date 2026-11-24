"""Apply the scalar forecasts to a common robot velocity field and report matched transfer errors."""
import json
import re

from discovery import write_json
from pair_normal_data import PREVIOUS, STUDY
from pair_normal_correction import OUTPUT
from pair_normal_search import metric, read, tags
from physics import FPS, rotate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reconcile(ids,normals,residual,ridge=.1):
    """Minimum-change velocity update for all close pairs in one frame, with a ridge on robot updates."""
    robots,mapping = np.unique(ids.reshape(-1),return_inverse=True)
    i,j = mapping.reshape(-1,2).T
    a = np.zeros((len(ids),len(robots),2))
    at = np.arange(len(ids))
    a[at,i],a[at,j] = normals,-normals
    a = a.reshape(len(ids),-1)
    dual = np.linalg.solve(a@a.T+ridge*np.eye(len(ids)),residual)
    delta = (a.T@dual).reshape(-1,2)
    return delta,i,j


def apply_updates(d,desired,reference,count):
    origin = d['origins']
    n,ei,ej = np.split(d['vector_checks'][:,:6],[2,4],axis=1)
    ty = d['vector_checks'][:,6]
    tangent = rotate(n)
    tp = ty+np.sum((ei-ej)*tangent,axis=1)
    normal = d['previous'].copy()
    delta_sse = 0.
    for idx in np.split(np.arange(len(origin)),1+np.flatnonzero(np.diff(origin[:,0]))):
        dv,i,j = reconcile(origin[idx,1:3],n[idx],desired[idx]-normal[idx])
        relative = dv[i]-dv[j]
        normal[idx] += np.sum(relative*n[idx],axis=1)
        tp[idx] += np.sum(relative*tangent[idx],axis=1)
        robot_error = np.empty_like(dv)
        robot_error[i],robot_error[j] = ei[idx],ej[idx]
        delta_sse += (2*robot_error*dv + dv**2).sum()
    new_mse = reference.rmse**2+delta_sse/(2*count)
    robot_variance = reference.rmse**2/(1-reference.r2)
    return normal,dict(tangent_rmse=metric(ty,tp)['rmse'],tangent_r2=metric(ty,tp)['r2'],
                       robot_rmse=np.sqrt(new_mse),robot_r2=1-new_mse/robot_variance)


def block_uncertainty(blocks):
    # Resample whole ten-second time blocks within each recording. All pairs at an origin stay together.
    rng = np.random.default_rng(25481)
    replicates = []
    for z in blocks:
        draw = rng.integers(len(z),size=(2000,len(z)))
        n,sy,sy2,old,new = z[draw].sum(axis=1).T
        var = sy2/n-(sy/n)**2
        replicates.append(np.column_stack(((old-new)/n/var,np.sqrt(old/n)-np.sqrt(new/n))))
    bounds = np.quantile(np.mean(replicates,axis=0),[.025,.975],axis=0)
    return dict(r2_gain_95=bounds[:,0].tolist(),rmse_reduction_95=bounds[:,1].tolist(),
                block_seconds=10,replicates=2000)


def term_definitions(names):
    direct = {
        'bias':'Constant 1','gapSpeed':'Projected clearance g / forecast interval T','rawIndent':'Positive observed overlap / T',
        'originShift':'Projected minus measured pair separation vector dotted with the observed normal, divided by T',
        'near2':'(D/T) Near²','otherCount':'Number of other neighbors of either member, counted per robot',
        'otherPush':'Difference of the two other-neighbor loads projected on the pair normal. Each load sums proximity times positive six-frame closing speed times neighbor normal.',
        'driveAlong':'Difference of heading unit vectors projected on the normal','facingIn':'Sum of the positive inward heading projections',
        'headingAlign':'Dot product of the two heading unit vectors','spinMagnitude':'Mean absolute ten-frame angular speed',
        'oppositeClosing':'Six-frame closing speed when the signed angular speeds are opposite, zero otherwise',
        'Near':'Projected (cutoff − distance) / (cutoff − D), clipped to [0,1]',
        'Crowded':'Indicator that either member has another observed neighbor inside the cutoff',
        'active_age':'Contact age in forecast intervals','fresh':'Exponential recency of the ongoing contact',
        'c_entry':'Inward speed just before contact entry in D/T units','H_cap':'Fraction of recent frames measured inside the cap',
        'H_c':'Recent average of dimensionless closing speed inside the measured cap',
        'H_spin':'Recent average of the pair’s summed dimensionless spin inside the measured cap',
        'repeat_count':'Number of contact entries during the last ten frames',
        'prior':'Previous Cartesian velocity law, differenced between the pair and dotted with the observed normal',
        'priorOpening':'Positive part of prior','priorClosing':'Positive part of −prior',
        'capCorrection':'Positive part of −g/T − prior',
        'capCurvedCorrection':'Positive part of −g/T − priorBend − prior',
        'priorBend':'Prior relative tangential speed squared times T/(2r)',
        'freeNormal':'Calibrated free-motion base velocity difference projected on the observed normal',
        'freeOpening':'Positive part of freeNormal','freeClosing':'Positive part of −freeNormal',
        'driveNormal':'Difference of calibrated propulsion velocities projected on the observed normal',
        'driveOpening':'Positive part of driveNormal','driveClosing':'Positive part of −driveNormal',
        'driveCapCorrection':'Positive part of −g/T − driveBend − driveNormal',
        'driveBend':'Relative tangential propulsion speed squared times T/(2r)',
        'freeCapCorrection':'Positive part of −g/T − freeNormal minus the free tangential curvature term',
        'individualIn':'Sum of the positive inward projections of the individual calibrated propulsion vectors',
        'turnedDrive':'driveNormal with each propulsion vector rotated by (current minus calibrated mean spin) times T/2',
        'turnedDriveCap':'Positive part of −g/T minus turnedDrive and its tangential curvature term'}
    windowed = {'u':'Signed projected relative velocity','opening':'Positive part of signed normal speed',
        'closing':'Positive part of negative normal speed','limit':'Maximum of u and −g/T',
        'bend':'Relative tangential speed squared times T/(2r)','limitBend':'Maximum of u and −g/T − bend',
        'rawU':'Signed normal speed from measured center displacements',
        'turn':'Normal projection after rotating each past velocity forward at its recent angular speed and applying the circular-arc mean-speed correction',
        'turnLimit':'Maximum of turn and −g/T minus its tangential curvature term'}
    definitions = []
    for name in names:
        if name in direct:
            desc = direct[name]
        elif '*' in name:
            desc = 'Product of '+', '.join(name.split('*'))
        else:
            family,h = re.fullmatch(r'([a-zA-Z]+)(\d+)',name).groups()
            desc = f'{windowed[family]}, using the past {h} frames'
        definitions.append(dict(term=name,definition=desc))
    return pd.DataFrame(definitions)


def write_report(summary,vectors,groups,interval):
    law = json.loads((OUTPUT/'common_law.json').read_text())
    term_definitions(law['candidate_terms']).to_csv(STUDY/'term_dictionary.csv',index=False)
    b = law['compact']
    mean_v = vectors.groupby('model')[['tangent_r2','robot_r2']].mean()
    primary = vectors[vectors.model=='residual_sparse'].set_index('held')
    previous = vectors[vectors.model=='previous'].set_index('held')
    lines = ['# Close-pair normal velocity over 0.4 seconds','',
        r'The target is $u^+_{ij}=[(\Delta\mathbf x_i-\Delta\mathbf x_j)/\Delta t]\cdot\mathbf n_{ij}(t)$, positive for separation. Both displacements and the origin normal use the measured centers. Projected centers supply kinematics and clearance, and the original measured gap supplies contact evidence. The 253 px neighbor cutoff, forecast origins and availability masks match the previous vector-law experiment. '+f'All models are tested on the same {int(primary.pairs.sum()):,} unordered pair observations.','',
        'Each holdout uses the first 75% of the other four recordings to fit coefficients and their final 25% to choose the equation or tree capacity. Each recording has equal weight. The merged common equation uses the union of proposals across the five training folds and is exploratory. Its coefficients for deployment use all five training prefixes. Rows marked coherent use one velocity per robot, including at simultaneous contacts.','',
        '| Prediction | Mean normal R² | Worst recording R² | Normal RMSE (px/s) |','|---|---:|---:|---:|']
    labels = {'previous':'Previous vector law','compact':'Direct three-term law','sparse':'Direct sparse equation',
              'boosted':'Direct boosted pair model','residual_compact':'Three-term correction',
              'coherent_residual_sparse':'Fold-selected sparse correction, coherent',
              'coherent_residual_shared':'Common sparse correction, coherent',
              'residual_boosted':'Boosted correction, pair prediction','coherent_residual_boosted':'Boosted correction, coherent'}
    for key,label in labels.items():
        row = summary.loc[key]
        lines.append(f'| {label} | {row.mean_r2:.4f} | {row.worst_r2:.4f} | {row.normal_rmse:.2f} |')
    lines += ['','## A compact common correction','',
        r'Let $u_0$ be the previous vector law projected onto the observed line of centers, $g=\max(r_{\rm projected}-D,0)$ be the clearance in pixels, and $[x]_+=\max(x,0)$. The pooled three-term correction gives','',
        f'$$\\hat u = {b["bias"]:.3f} {1+b["prior"]:+.4f}u_0 {b["capCorrection"]:+.4f}[-g/\\Delta t-u_0]_+.$$','',
        'The hinge activates when the predicted inward displacement exceeds the available radial clearance over the forecast interval. The intercept has units of px/s. Spin, lateral motion, neighbor effects and measurement noise can change observed motion along the origin axis over this interval.','',
        '## Sparse correction by recording','',
        '| Held-out recording | Pairs | Previous normal R² | Updated normal R² | Previous RMSE | Updated RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,r in primary.iterrows():
        lines.append(f'| {tag} | {int(r.pairs):,} | {previous.loc[tag,"r2"]:.3f} | {r.r2:.3f} | {previous.loc[tag,"rmse"]:.2f} | {r.rmse:.2f} |')
    lines += ['',
        r'To obtain one velocity for each robot in a frame, the update minimizes $\|A\delta v-(\hat u-A v_0)\|^2+0.1\|\delta v\|^2$. Each row of $A$ contains the two opposite unit normals for a pair. This preserves the sum of velocity updates within each connected contact group. Tangential and whole-robot errors are recomputed after this update.','',
        f'Mean tangential R² changes from {mean_v.loc["previous","tangent_r2"]:.3f} to {mean_v.loc["residual_sparse","tangent_r2"]:.3f}. Mean whole-robot Cartesian R² changes from {mean_v.loc["previous","robot_r2"]:.3f} to {mean_v.loc["residual_sparse","robot_r2"]:.3f}.',
        f'The paired 95% interval for the sparse update’s mean R² gain is {interval["r2_gain_95"][0]:.3f} to {interval["r2_gain_95"][1]:.3f}. It resamples ten-second time blocks within these five recordings.','',
        '| State at the forecast origin | Pairs | Previous normal RMSE | Updated normal RMSE |','|---|---:|---:|---:|']
    by_group = groups.groupby(['group','model']).agg(rmse=('rmse','mean'),pairs=('pairs','sum'))
    for state in ['approaching','separating_or_stationary','isolated_pair','multiple_neighbors','observed_inside_cap','observed_outside_cap']:
        old = by_group.loc[(state,'previous')]
        new = by_group.loc[(state,'coherent_residual_sparse')]
        lines.append(f'| {state.replace("_"," ")} | {int(old.pairs):,} | {old.rmse:.2f} | {new.rmse:.2f} |')
    lines += ['','## Search and outputs','',
        f'Two native EPDE NSGA-II searches ran per holdout, one on the pair speed and one on the previous law’s residual. Each used seed 923701, 16 individuals, 12 generations, up to ten RHS terms and up to three factors per term. Deterministic beam searches also proposed supports. For the correction, {law["structures"]:,} distinct proposals yielded {law["common_eligible"]:,} common eligible structures and {law["epde_front"]} nondominated structures on mean residual WAPE and dyad instability. The validation-error versus size front is also flagged. Rank-deficient fits have no unique coefficients.','',
        'The main comparison is the sparse correction selected independently within each fold. The nonlinear reference uses histogram gradient boosting and selects its leaf count on the same validation tails. Forecasts use past pose windows of 2, 3, 6, 10 and 20 frames. All forecasts are conditional on a close pair at the origin. Force, impulse and event-time restitution are separate quantities.','',
        '[Full common equations and pooled coefficients](correction/pareto_equations.csv) · [Term dictionary](term_dictionary.csv) · [Prior vector law and calibration](../search/cross_recording/pooled_common_law.json) · [Every candidate’s fold errors](correction/candidate_errors.csv.gz) · [Vector scores](vector_scores.csv) · [Matched score plot](normal_transfer.png)']
    (STUDY/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def report():
    scores = pd.concat([pd.read_csv(STUDY/'heldout_scores.csv'),pd.read_csv(OUTPUT/'heldout_scores.csv')])
    previous = pd.read_csv(PREVIOUS/'comparison_by_recording.csv').query(
        "geometry=='projected' and selection=='min_wape_shared'").set_index('held')
    counts = json.loads((PREVIOUS/'summary.json').read_text())['matched_counts']
    law = json.loads((STUDY/'common_law.json').read_text())
    keys = '+'.join(law['shared'])
    candidates = pd.read_csv(STUDY/'candidate_errors.csv.gz',usecols=['held','terms','refit'])
    weights = candidates[candidates.terms==keys].set_index('held').refit
    groups,coherent,blocks = [],[],[]
    for tag in tags():
        d = read(tag,'test')
        names = list(d['names'])
        with np.load(STUDY/f'{tag}_predictions.npz') as z:
            pred = {k:z[k] for k in z.files if k not in ('origins','y')}
            np.testing.assert_array_equal(z['origins'],d['origins'])
            np.testing.assert_array_equal(z['y'],d['y'])
        with np.load(OUTPUT/f'{tag}_predictions.npz') as z:
            pred.update(dict(z))
        y = d['y']
        b = json.loads(weights[tag])
        pred['shared'] = d['x'][:,[names.index(k) for k in b]]@list(b.values())
        ref = previous.loc[tag]
        for model in ('previous','sparse','shared','boosted','residual_compact','residual_sparse','residual_shared','residual_boosted'):
            p,extra = apply_updates(d,pred[model],ref,counts[tag]['robot_origins'])
            coherent.append(dict(held=tag,model=model,pairs=len(y),**metric(y,p),**extra))
            if model=='previous':
                np.testing.assert_allclose([extra['tangent_rmse'],metric(y,p)['rmse']],[ref.tangent,ref.normal],rtol=1e-8)
            else:
                pred['coherent_'+model] = p
        mask = d['x'][:,names.index('u10')]<0
        multi = d['x'][:,names.index('Crowded')]>0
        inside = d['x'][:,names.index('rawIndent')]>0
        for group,keep in [('approaching',mask),('separating_or_stationary',~mask),('isolated_pair',~multi),
                           ('multiple_neighbors',multi),('observed_inside_cap',inside),('observed_outside_cap',~inside)]:
            for model in ('previous','compact','sparse','coherent_residual_sparse','residual_boosted'):
                groups.append(dict(held=tag,group=group,model=model,pairs=int(keep.sum()),**metric(y[keep],pred[model][keep])))
        # Test displacements and time blocks enter only aggregate reporting.
        ts = d['origins'][:,0]//int(10*FPS)
        values = np.column_stack((np.ones(len(y)),y,y*y,(y-pred['previous'])**2,(y-pred['coherent_residual_sparse'])**2))
        blocks.append(np.add.reduceat(values,np.r_[0,1+np.flatnonzero(np.diff(ts))],axis=0))
        np.savez_compressed(STUDY/f'{tag}_predictions.npz',origins=d['origins'],y=y,**pred)
    vectors = pd.DataFrame(coherent)
    vectors.to_csv(STUDY/'vector_scores.csv',index=False)
    groups = pd.DataFrame(groups)
    groups.to_csv(STUDY/'by_contact_state.csv',index=False)
    interval = block_uncertainty(blocks)
    write_json(STUDY/'paired_uncertainty.json',interval)
    comparison = pd.concat([scores[['held','model','rmse','r2','mae']],
              vectors[vectors.model!='previous'].assign(model=lambda t:'coherent_'+t.model)[['held','model','rmse','r2','mae']]])
    summary = comparison.groupby('model').agg(mean_r2=('r2','mean'),worst_r2=('r2','min'),
              normal_rmse=('rmse','mean'),normal_mae=('mae','mean'))
    summary.to_csv(STUDY/'comparison.csv')
    write_report(summary,vectors,groups,interval)
    fig,(ax,bx) = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for name,label,marker in [('previous','Previous vector law','o'),('residual_compact','Three-term correction','s'),
                              ('coherent_residual_sparse','Sparse vector update','D'),('residual_boosted','Boosted pair forecast','^')]:
        m = comparison[comparison.model==name].sort_values('held')
        ax.plot(range(len(m)),m.r2,marker=marker,label=label)
    ax.set_xticks(range(len(m)),[t.replace('_','\n') for t in m.held],fontsize=8)
    ax.set(ylabel='Held-out pair-normal R²')
    ax.legend(fontsize=8)
    group = groups.groupby(['group','model']).rmse.mean().unstack()
    labels = ['approaching','separating_or_stationary','isolated_pair','multiple_neighbors']
    yy = np.arange(len(labels))
    bx.barh(yy-.18,group.loc[labels,'previous'],height=.35,label='Previous')
    bx.barh(yy+.18,group.loc[labels,'coherent_residual_sparse'],height=.35,label='Normal-target update')
    bx.set_yticks(yy,['Approaching','Separating or stationary','Isolated pair','Multiple neighbors'])
    bx.set(xlabel='Mean recording RMSE (px/s)')
    bx.legend(fontsize=8)
    fig.savefig(STUDY/'normal_transfer.png',dpi=150)
    plt.close(fig)
    print(summary.round(4).to_string(),flush=True)
