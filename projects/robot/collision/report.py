"""Generate the transfer report and machine-readable comparison tables from completed folds."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from discovery import gram, solve, weak_design, window, write_json
from impacts import VARIABLES
from physics import (CAP, DT, EDGES, LAG, NAMES, OUT, ROBOT, SEEDS, SIGMA, arena_for, design,
                     fit_self, recordings, swarm_files, swarm_meta)

BASES=('self','local_pair','constant_drive','banded_drive')
TITLES={'self':'Self motion','local_pair':'Local pair','constant_drive':'Constant push + drive',
        'banded_drive':'Banded push + drive','strong':'Forward EPDE','weak':'Weak EPDE'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equation(coef):
    if not coef:
        return '0'
    return ' '.join(f'{b:+.3g} {k}' for k,b in coef.items()).lstrip('+')


def rebound_section():
    path = OUT / 'contact_rebound.csv'
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    lines = ['', '## Does extra rebound improve the contact forecast?', '',
             'At each inward crossing of the 48.7 mm cap diameter, we forecast how far the two centers move apart '
             'along the line joining them at entry. The baseline projects the frozen drift forecast onto the disk and arena constraints. '
             'The alternative adds a fraction between zero and one of incoming closing speed times elapsed time. '
             'At 0.2 s we advance the same 0.4 s drift at constant velocity over the shorter interval.', '',
             'The table uses closing speeds measured over three pre-entry frames. We screen for walls and third robots '
             'through entry and include every eligible encounter with ten subsequent frames. A release means the centers '
             'reached 1.03 cap diameters at least once by the scoring time. The fit weights each represented training recording equally. '
             'Validation selects the positive correction only if it lowers mean recording-level squared error. The table pools held-out errors by encounter.', '',
             '| Minimum third-robot distance | Time | Encounters | Still awaiting first release | Baseline RMSE | Selected rebound model RMSE |',
             '|---|---:|---:|---:|---:|---:|']
    for isolation in ('neighborhood', 'physical_clearance'):
        for h in ('0.2', '0.4'):
            group = [r for r in rows if r['isolation'] == isolation and r['horizon_s'] == h
                     and r['pre_frames'] == '3' and int(r['test_events']) > 0]
            if not group:
                continue
            weights = [int(r['test_events']) for r in group]
            n = sum(weights)
            pending = n - sum(int(r['released_by_horizon']) for r in group)
            errors = [np.sqrt(np.average([float(r[k])**2 for r in group], weights=weights))
                      for k in ('baseline_rmse_mm', 'selected_rmse_mm')]
            clearance = float(group[0]['third_clearance_px'])
            lines.append(f'| {clearance:.2f} px | {h} s | {n:,} | {pending} | {errors[0]:.2f} mm | {errors[1]:.2f} mm |')
    cells = [r for r in rows if r['isolation'] == 'physical_clearance' and r['horizon_s'] == '0.4'
             and r['pre_frames'] == '3' and r['train_multiplier'] and int(r['test_events']) > 0]
    zeros = sum(float(r['train_multiplier']) == 0 for r in cells)
    lines += ['', f'With the broader sample, the training fit placed the additional rebound fraction at zero in {zeros} of '
              f'{len(cells)} scorable folds at 0.4 s. The forecast already includes a learned response to closing speed. '
              'These endpoint measurements leave the instantaneous material restitution coefficient unresolved.', '',
              'Run `python projects/robot/collision/run.py --stage rebound` to repeat the test. The full table includes '
              'five-frame velocity estimates, a constant outward-offset control, and errors grouped by release. '
              '[Results by recording](contact_rebound.csv)']
    return lines


def early_contact_section():
    path = OUT / 'early_contact.csv'
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    lines = ['', '## The first three camera frames after contact', '',
             'At 25 frames/s the first observation after a detected crossing is 40 ms later. '
             'We move the entry distance two pixels inward or outward (about 0.44 mm), then detect encounters again. '
             'All variants use the same physical disk diameter for projection, the same release threshold and the same ten-frame follow-up requirement. '
             'Moving the threshold can change both the entry frame and which encounters qualify.', '',
             'The table uses three pre-entry frames to estimate closing speed and the 226.14 px third-robot clearance. '
             '“Outward >1 px” counts observed displacement along the entry line of centers. '
             'The cutoff describes displacement in image units. The baseline advances the frozen drift over the shorter time and enforces disk and wall constraints. '
             'To test the information in closing speed, we fit a signed constant offset on training recordings, then compare it with a model that also fits '
             'a nonnegative speed coefficient bounded by one. Validation chooses between those two models. Each held-out encounter contributes equally to the reported RMSE.', '',
             '| Entry offset | Time | Encounters | Outward >1 px | Baseline RMSE | With constant offset | With offset and optional speed term |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    primary = [r for r in rows if r['isolation'] == 'physical_clearance' and int(r['pre_frames']) == 3 and int(r['test_events'])]
    pooled_changes = {}
    for offset in sorted({int(r['entry_offset_px']) for r in primary}):
        for h in sorted({float(r['horizon_s']) for r in primary}):
            group = [r for r in primary if int(r['entry_offset_px']) == offset and float(r['horizon_s']) == h]
            if not group:
                continue
            weights = [int(r['test_events']) for r in group]
            n = sum(weights)
            outward = sum(int(r['outward_over_one_px']) for r in group) / n
            errors = [np.sqrt(np.average([float(r[k])**2 for r in group], weights=weights))
                      for k in ('baseline_rmse_mm', 'calibrated_offset_rmse_mm', 'selected_offset_speed_rmse_mm')]
            pooled_changes[offset, h] = errors[2] - errors[1]
            lines.append(f'| {offset:+d} px | {1000*h:.0f} ms | {n:,} | {outward:.1%} | '
                         f'{errors[0]:.3f} mm | {errors[1]:.3f} mm | {errors[2]:.3f} mm |')
    nominal = [change for (offset, h), change in pooled_changes.items() if offset == 0]
    if nominal:
        lines += ['', f'At the nominal entry distance, adding the optional speed term changed pooled RMSE by '
                  f'{min(nominal):+.4f} to {max(nominal):+.4f} mm relative to the constant offset. '
                  f'Across all three entry distances the changes ranged from {min(pooled_changes.values()):+.4f} to '
                  f'{max(pooled_changes.values()):+.4f} mm. Negative changes indicate lower error.']
    lines += ['', 'These horizons describe motion after a sampled inward crossing. An inward tracking fluctuation can trigger entry, '
              'and the physical impact can occur between camera frames. '
              'Run `python projects/robot/collision/run.py --stage early` to repeat the analysis. The per-recording table also includes '
              'five-frame incoming speeds, the stricter third-robot clearance, signed mean displacement and the speed-only comparison. '
              '[Per-recording early contact results](early_contact.csv)']
    return lines


def pooled(folds):
    # The same structure must be available in every training fold. Rank using internal validation only.
    tables={}
    for tag, f in folds.items():
        vals={}
        for family in ('strong','weak'):
            for model in read(OUT/tag/f'{family}_refits.json'):
                vals[family,tuple(sorted(model['coef']))]=model['validation_rmse']**2
        for name in BASES:
            m=f['models'][name]
            vals[name,tuple(sorted(m['coef']))]=m['validation_rmse']**2
        tables[tag]=vals
    common=set.intersection(*(set(v) for v in tables.values()))
    key=min(common,key=lambda k:(np.mean([v[k] for v in tables.values()]),len(k[1]),k))
    family,support=key
    tracks=list(recordings().values())
    arenas=arena_for(tracks)
    lookup,self_coef=fit_self(tracks,arenas)
    grams=[]
    for tr in tracks:
        step=1 if family=='weak' else LAG
        rows,x,base,y,n=design(tr,lookup,self_coef,arenas,step)
        if family=='weak':
            x,y=weak_design(tr.pos[rows],x,base,window(tr.pos[rows]))
        else:
            y=y-base
            if family=='local_pair':
                x,y=x[n==1],y[n==1]
        grams.append(gram(x,y))
    beta=solve(np.mean(grams,axis=0),support)
    output=dict(family=family,coefficients={n:float(beta[NAMES.index(n)]) for n in support},
                common_self=dict(persistence=self_coef[0],wall=self_coef[1]),
                robot_parameters={f'{k[0]}/{k[1]}':v.tolist() for k,v in lookup.items()},
                arenas={k:v.tolist() for k,v in arenas.items()},eligible_structures=len(common))
    write_json(OUT/'pooled_law.json',output)
    return output


def plot(folds):
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
    colors=plt.colormaps['tab10'].colors
    distances=np.linspace(.85,1.12,450)
    def response(coef):
        b=np.digitize(distances,EDGES)-1
        return (distances<1.1)*(coef.get('push',0)+coef.get('overlap',0)*np.maximum(1.1-distances,0)
                                 +np.array([coef.get('band'+str(i),0) for i in b]))
    for col,(tag,f) in zip(colors,folds.items()):
        for ax,form in zip(axes,['strong','weak']):
            ax.plot(distances,response(f['models'][form]['coef']),color=col,lw=1.5,label=tag)
    for ax,form in zip(axes,['Forward increments','Weak form']):
        ax.axvline(CAP/SIGMA,color='.4',ls=':',lw=1)
        ax.set(title=form,xlabel='Center distance / 230 px',ylabel='Outward pair velocity (px/s)')
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'distance_response.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True,constrained_layout=True)
    for ax,(tag,fold) in zip(axes.flat,folds.items()):
        models=fold['models']
        obs=models['strong']['heldout']['bond_profile']['observed']
        ax.plot(range(5),obs,'o-',color='black',label='Observed')
        for method in ('strong','weak','banded_drive'):
            ax.plot(range(5),models[method]['heldout']['bond_profile']['predicted'],'.--',label=TITLES[method])
        ax.axhline(0,color='.6',lw=.7)
        ax.set_title(tag)
        ax.set_xticks(range(5),['<.93','.93–.97','.97–1','1–1.05','1.05–1.1'],rotation=25,fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.supylabel('Bond-normal relative velocity (px/s)')
    fig.supxlabel('Center distance / 230 px')
    fig.savefig(OUT/'heldout_bond_velocity.png',dpi=180)
    plt.close(fig)


def report():
    tags=[swarm_meta(f)['tag'] for f in swarm_files()]
    folds={tag:read(OUT/tag/'assessment.json') for tag in tags}
    impacts={tag:read(OUT/tag/'impacts.json') for tag in tags}
    legacy=read(OUT/'legacy.json')
    law=pooled(folds)
    coeff=law['coefficients']
    at_cap=coeff.get('push',0.)+(1.1-CAP/SIGMA)*coeff.get('overlap',0.)+coeff.get('band1',0.)
    plot(folds)
    drift=[]
    for tag,f in folds.items():
        for method,model in f['models'].items():
            h=model['heldout']
            row=dict(recording=tag,method=method,validation_rmse=model['validation_rmse'],
                     heldout_rmse=h['rmse'],r2=h['r2'],normal_mae=h['pair_mae'][0],tangential_mae=h['pair_mae'][1],
                     normal_rmse=h['pair_rmse'][0],tangential_rmse=h['pair_rmse'][1])
            for baseline in BASES:
                row['delta_r2_vs_'+baseline]=h['r2']-f['models'][baseline]['heldout']['r2']
            for k,cl in h['contact_classes'].items():
                row['n_contacts_'+k]=cl['n']
                row['rmse_contacts_'+k]=cl['rmse']
            drift.append(row)
    write_csv(OUT/'drift_scores.csv',drift)
    erows=[]
    for tag,data in impacts.items():
        for setting,d in data.items():
            name=d['selected']
            observed=d['test'].get(name)
            rest=d['coefficients'].get('restitution')
            erows.append(dict(recording=tag,setting=setting,approaches=d['counts']['approaches'],
                      resolved=d['counts']['resolved'],wall=d['counts'].get('wall',0),
                      third_robot=d['counts'].get('third_robot',0),prolonged=d['counts'].get('prolonged',0),
                      incomplete=d['counts'].get('incomplete',0),nonapproaching=d['counts'].get('nonapproaching',0),
                      training_events=sum(d['train_counts']),validation_events=sum(d['valid_counts']),selected=name,
                      fitted_restitution=-rest[0][0] if rest else None,normal_mae=observed['mae'][0] if observed else None,
                      tangent_mae=observed['mae'][1] if observed else None,
                      outgoing_positive_fraction=d['positive_outgoing_fraction']))
            row=erows[-1]
            for baseline in ('persistence','restitution'):
                scored=d['test'].get(baseline)
                row[baseline+'_rmse']=np.sqrt(scored['mse']) if scored else None
            for i,var in enumerate(VARIABLES):
                for axis,component in enumerate(('normal','tangent')):
                    row[component+'_'+var]=d['coefficients'][name][i][axis] if name else None
    write_csv(OUT/'event_sensitivity.csv',erows)
    lines=['# Robot–robot collision laws across five recordings','',
           'Five leave-one-recording-out tests cover 30–40 robots, PWM 10–20, and 40–41 cm arenas. Each recording contains 8,250 frames at 25 Hz. Physical robot IDs can recur across recordings. Performance here measures transfer to a new setup with that overlap.','',
           '## The transferable contact term','',
           'The common structure with the lowest mean internal validation squared error yielded the following all-record coefficient estimate. The five held-out tests below assess the entire selection procedure. The all-record estimate uses those measurements for fitting.','',
           f"**Pair contribution in px/s**  `{equation(law['coefficients'])}`  ({TITLES[law['family']]}).",'',
           f"At a center separation of one physical cap diameter, the distance term alone contributes {at_cap:.1f} px/s ({at_cap*4.87/CAP:.2f} cm/s) outward per neighbor. The closing speed, headings and neighbor propulsion determine the remaining contribution.",'',
           'Add the contributions of neighbors within 253 px to the robot’s heading-dependent propulsion, previous velocity and inward wall response. The two propulsion coefficients are specific to robot ID and PWM. Unseen IDs use the training PWM average. The full model includes these coefficients and arena geometry. [Pooled law](pooled_law.json)','',
           'The outward unit vector points from neighbor to robot. `front` and `neighbor_front` weight it by the respective heading toward the other robot. `overlap` weights it by (253 − distance)/230. `band0…4` select intervals with edges 0, 0.93, 0.97, 1, 1.05 and 1.1 in units of 230 px. These coefficients and `push` have units of px/s. `approach` weights the outward direction by the positive closing speed. `slide` uses signed relative tangential velocity. `drive` sums neighboring propulsion and `drive_in` sums its positive inward projection. Their coefficients are dimensionless. `gear` weights the tangent by the robots’ summed angular velocities and has a coefficient in px/rad.','',
           '## Held-out drift','',
           '| Omitted recording | Forward R² | Weak R² | Banded baseline R² | Validation choice | Choice R² | Normal / tangent MAE, px/s |',
           '|---|---:|---:|---:|---|---:|---:|']
    for tag,f in folds.items():
        m=f['models']; h=m[f['selected']]['heldout']
        lines.append(f"| {tag} | {m['strong']['heldout']['r2']:.3f} | {m['weak']['heldout']['r2']:.3f} | {m['banded_drive']['heldout']['r2']:.3f} | {TITLES[f['selected']]} | {h['r2']:.3f} | {h['pair_mae'][0]:.1f} / {h['pair_mae'][1]:.1f} |")
    for method in ('strong','weak'):
        delta=np.array([f['models'][method]['heldout']['r2']-f['models']['banded_drive']['heldout']['r2']
                        for f in folds.values()])
        lines+=['',f"{TITLES[method]} increased held-out R² over banded push plus drive in {np.sum(delta>0)}/5 recordings. The paired changes ranged from {delta.min():+.4f} to {delta.max():+.4f}."]
    lines+=['','R² uses both Cartesian velocity components. The MAEs use touching pairs’ relative velocity in the frame of their bond. The full table includes errors by contact count, paired differences against every baseline, and each recording’s own result. [Drift scores](drift_scores.csv)','',
            '| Omitted recording | Best baseline on test | Best EPDE form on test | EPDE minus baseline R² | Weak history-only / weak contact RMSE, px/s |',
            '|---|---|---|---:|---:|']
    for tag,f in folds.items():
        m=f['models']; base=max(BASES,key=lambda k:m[k]['heldout']['r2'])
        form=max(('strong','weak'),key=lambda k:m[k]['heldout']['r2'])
        delta=m[form]['heldout']['r2']-m[base]['heldout']['r2']
        lines.append(f"| {tag} | {TITLES[base]} | {TITLES[form]} | {delta:+.4f} | {f['memory']['weak']['rmse']:.1f} / {f['memory']['weak_contact_rmse']:.1f} |")
    lines+=['','This second comparison describes the outer test results after scoring. The history control uses velocity windows back to 9.6 s. Both RMSEs in its column use the same valid frames.','',
            '![Radial component of the five transferred laws](distance_response.png)','',
            'Each curve shows the separation-only terms of one training fold. The dotted line marks the calibrated 219.55 px cap diameter. Heading, relative velocity, spin and neighbor propulsion can contribute additional motion.','',
            '![Observed and predicted bond-normal velocities by distance on each omitted recording](heldout_bond_velocity.png)','',
            'Positive bond-normal velocity means the two centers separate over the following 0.4 s. The first band contains every pair closer than 0.93 times 230 px.','',
            '## Individual impacts','',
            '| Omitted recording | Cap crossings | Isolated, 3 frames | Isolated, 5 frames | Training / validation events (3 frames) | Physical-clearance sensitivity, 3 frames |',
            '|---|---:|---:|---:|---:|---:|']
    for tag,d in impacts.items():
        c=d['cap_3']
        lines.append(f"| {tag} | {c['counts']['approaches']} | {c['counts']['resolved']} | {d['cap_5']['counts']['resolved']} | {sum(c['train_counts'])} / {sum(c['valid_counts'])} | {d['physical_clearance_3']['counts']['resolved']} |")
    tested=[tag for tag,d in impacts.items() if d['cap_3']['test'].get(d['cap_3']['selected'])]
    lines+=['',f'The primary isolation rule yielded a validation-selected model with scorable outgoing velocities in **{len(tested)} of five** transfer folds. '
            'Each accepted event begins after an inward cap crossing and ends after a 3%-of-cap clearance. Velocity slopes use three or five frames before entry and after release. A second approach requires a fresh release. Wall encounters, third-robot encounters, episodes lasting over one second, censored windows and non-approaching pre-event velocities are counted. Wall, third-body and prolonged flags can overlap. These measurements condition on observed separation.','',
            'The primary rule excludes any third robot within 253 px during the event or its velocity windows. The exploratory physical-clearance rule uses 226.14 px for third robots, after inspecting event availability. The sensitivity table also varies the entry threshold by ±2 px and repeats detection at the historical neighborhood radius. It includes fitted normal restitution, prediction errors and the fraction of outgoing normal velocities above zero. [Event sensitivity](event_sensitivity.csv)','',
            '| Omitted recording | Physical-clearance model | Resolved | Normal / tangent MAE, px/s | Persistence MAE, px/s |',
            '|---|---|---:|---:|---:|']
    def pair(s):
        return '—' if not s else ' / '.join(f'{v:.1f}' for v in s['mae'])
    for tag,d in impacts.items():
        c=d['physical_clearance_3']
        lines.append(f"| {tag} | {c['selected'] or '—'} | {c['counts']['resolved']} | {pair(c['test'].get(c['selected']))} | {pair(c['test'].get('persistence'))} |")
    slopes={}
    for width in (3,5):
        cells=[d[f'physical_clearance_{width}'] for d in impacts.values()]
        slopes[width]=[c['coefficients'][c['selected']][0][0] for c in cells if c['selected']]
    if slopes[3] and slopes[5]:
        lines+=['',f"Under physical clearance, the coefficient of incoming normal velocity in the outgoing normal equation ranged from {min(slopes[3]):.2f} to {max(slopes[3]):.2f} with three frames and {min(slopes[5]):.2f} to {max(slopes[5]):.2f} with five. The negative sign describes reversal conditional on a measured release. Window length affects its magnitude. The event table gives every selected coefficient for both outgoing components. Headings enter as the relative heading vector projected onto the entry bond. Surface spin is the summed pre-event angular speed times the cap radius in px."]
    lines += rebound_section()
    lines += early_contact_section()
    lines+=['','## Swarm simulation','',
            'Each fold initializes the held-out swarm from ten frames of position and heading history. For an unseen ID at that PWM, the simulator combines its observed initial spin direction with the training PWM mean absolute spin rate. Four matched random seeds propagate frozen laws with training residual noise. The cap condition projects overlapping centers apart. The unprojected condition leaves pair separation to the learned law. Both enforce the arena boundary.','',
            '| Omitted recording | Law | Projection | Nearest-gap TV | Contact-count TV | Overlap fraction | Observed overlap |',
            '|---|---|---|---:|---:|---:|---:|']
    for tag,f in folds.items():
        for method,tests in f['simulations'].items():
            for condition,runs in tests.items():
                avg=lambda k:np.mean([r[k] for r in runs])
                lines.append(f"| {tag} | {TITLES[method]} | {condition} | {avg('gap_hist_tv'):.3f} | {avg('contacts_tv'):.3f} | {avg('overlap_fraction'):.3f} | {f['observed']['overlap_fraction']:.3f} |")
    free_overlap=[np.mean([r['overlap_fraction'] for r in f['simulations'][f['selected_search']]['unprojected']])
                  for f in folds.values()]
    actual_overlap=[f['observed']['overlap_fraction'] for f in folds.values()]
    excess=sum(v>a for v,a in zip(free_overlap,actual_overlap))
    lines+=['',f"With pair projection disabled, the selected EPDE model produced more overlap than observed in {excess} of five recordings. Its overlap fraction was {min(free_overlap):.1%}–{max(free_overlap):.1%}, compared with {min(actual_overlap):.1%}–{max(actual_overlap):.1%} in the recordings."]
    profiles=[f['models']['strong']['heldout']['bond_profile'] for f in folds.values()]
    deepest=[p['predicted'][0]-p['observed'][0] for p in profiles if p['count'][0]]
    lines+=['',f"The forward law overpredicted mean separation speed in the closest distance band in {sum(x>0 for x in deepest)}/5 recordings. Its signed errors there ranged from {min(deepest):+.1f} to {max(deepest):+.1f} px/s."]
    lines+=['','Overlap counts nearest centers more than one camera pixel inside the cap diameter, so floating-point roundoff in hard-core projection cannot count as a physical overlap. TV is total variation distance between histograms. Per-seed files also contain the strict-cap count as a fraction and the 0.1st percentile of nearest-neighbor distance in px.','',
            '## Reproduction','',
            'Run `python projects/robot/collision/run.py` from the repository root. It resumes completed folds and seeds. Set `EPDE_COLLISION_OUTPUT` to a new directory for a fresh campaign. [Historical baseline replay](legacy.json) and [source and measurement hashes](provenance.json).','',
            'Each outer fold uses the first 75% of four raw recordings for training and their last 25% for internal selection. Preprocessing restarts at each cut. The discovery screen uses 512 time-stratified component rows per training recording, identical across seeds within each form. The forward target spans 10 raw frames on a nonoverlapping 0.4 s clock. Weak rows integrate positions by parts and average physical vector columns on contiguous 25 Hz tracks with EPDE’s half-power window selector. Both matrices enter the same native plain-mode NSGA-II fitter as precomputed target rows. Each run uses automatic population sizing, 50 full epochs, WAPE and dyadic coefficient instability. Three seeds per form yield 30 searches.','',
            'Each distinct identifiable support is refitted on all eligible training rows of its form, weighting recordings equally. A shared coefficient multiplies both vector components. The physical pair law has zero intercept. Selection minimizes mean recording-level validation squared error. The local-pair control uses single-contact training rows and the two historical terms. The five-band and constant-drive controls use the common self-motion calibration. The historical replay separately checks the original fitting procedure and time split.','',
            'Each fold directory contains training windows and calibration, native fronts and candidate archives, refitted candidates, event tables, and held-out scores.']
    spans={tag:2*m/25 for f in folds for tag,(m,_,_) in read(OUT/f/'training.json')['windows'].items()}
    lines+=['',f'The native weak-window spans were {min(spans.values()):.2f}–{max(spans.values()):.2f} seconds across recording prefixes. This time averaging is substantially longer than the 0.4 s prediction horizon.']
    coverage=[read(OUT/tag/f'{form}_coverage.json') for tag in folds for form in ('strong','weak')]
    gaps=[c['archived_validation_best']['validation_rmse']-c['validation_best_over_subsets']['validation_rmse']
          for c in coverage]
    lines+=['',f"An independent enumeration fitted every full-rank subset of up to six physical vector terms on the full training matrices. EPDE's archives reached the best validation RMSE within 0.000001 px/s in {sum(g<1e-6 for g in gaps)} of ten fold/form comparisons. The largest gap was {max(gaps):.4f} px/s. Only EPDE-proposed supports competed for the reported search result."]
    gains=[v['contact']-v['drive'] for v in legacy['heldout'].values()]
    lines+=['',f"The historical replay recovered the 30-robot pair term `{equation(legacy['local_pair'])}` in its original token names (px/s). Adding distance bands to constant push plus neighbor drive improved the original within-recording R² by {min(gains):.4f}–{max(gains):.4f} across all five recordings."]
    lines+=['','For robot ID/PWM combinations present in training with zero contact-free rows, the minimum-norm calibration assigns zero propulsion. A sensitivity fit substitutes the mean among supported robots at that PWM and refits the same selected supports. It uses the original held-out frames.','',
            '| Omitted recording | Unsupported training ID/PWM pairs | Affected test IDs | Selected law R², primary / sensitivity |',
            '|---|---:|---|---:|']
    for tag,f in folds.items():
        a=f['propulsion_sensitivity']; k=f['selected']
        ids=', '.join(map(str,a['heldout_affected'])) or '—'
        lines.append(f"| {tag} | {len(a['unobserved'])} | {ids} | {f['models'][k]['heldout']['r2']:.3f} / {a['scores'][k]:.3f} |")
    if (OUT/'pareto'/'report.md').exists():
        lines += ['', '## Pareto fronts for the full contact library', '',
                  'Every subset of the 14 collision terms was evaluated with the native discrepancy and instability objectives. '
                  'Each equation has validation and held-out errors, including error along the line between contacting robots. '
                  '[Fronts, equations and term-by-term errors](pareto/report.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    root=ROBOT.parents[1]
    files=sorted((ROBOT/'collision').glob('*.py'))+[Path(p) for p in swarm_files()]
    sys_path=ROBOT/'epde_u'
    import sys
    sys.path.insert(0,str(sys_path))
    from epde_digest import source_hash
    write_json(OUT/'provenance.json',dict(epde_source_at_report=source_hash(),seeds=SEEDS,
               sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print(OUT/'report.md')
