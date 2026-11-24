"""Validate the independent archives and report the frozen paired comparison."""
import json

from discover import DONOR_CONFIRMATION, ROOT
from compare import previous, read_archive
from epoch_settings import OPTIONS, live


def build():
    reference = read_archive(ROOT/'results'/'24112401')
    shared_keys = ('system', 'train_hashes', 'input_shape', 'variables', 'plan', 'noise_percent', 'epochs_completed')
    reference_inputs = {key:reference[key] for key in shared_keys}
    reference_settings = reference['settings']
    del reference
    rows = []
    missed_donors = []
    forms, frequencies = previous.registry('ns', live.load_data('ns', **OPTIONS['ns']))
    for seed in DONOR_CONFIRMATION:
        dest = ROOT/'results'/str(seed)
        a = read_archive(dest)
        assert a['seed'] == seed and all(a[k] == value for k,value in reference_inputs.items())
        settings = dict(a['settings'])
        assert settings['system_rps_params']['variable_order_seed'] == seed
        settings['system_rps_params'] = dict(settings['system_rps_params'],
                                             variable_order_seed=reference_settings['system_rps_params']['variable_order_seed'])
        assert settings == reference_settings
        op = a['operator']
        assert op['fill_settings']['fill_beam_width'] == 1000
        assert op['epoch_clock'][99]['epoch'] == 100 and op['epoch_clock'][-1]['epoch'] == 150
        assert op['full_cost'][-1] == a['calls'] + op['extra_fits'] == op['epoch_clock'][-1]['cost']
        assert len(op['fit_milliseconds']) == len(op['full_cost']) == a['calls']
        assert all(x<=y for x,y in zip(op['fit_milliseconds'], op['fit_milliseconds'][1:]))
        r = json.loads((dest/'donor_history.json').read_text())
        for arm in r.values():
            assert arm['total_fits'] == arm['native_fits'] + arm['new_fits'] == op['full_cost'][-1]
            assert arm['native_fits'] >= op['epoch_clock'][99]['cost']
            assert arm['complete_generated'] >= arm['unbeaten_by_native'] >= arm['complete_nondominated'] >= arm['surviving_all_native']
        if all(not x['complete_generated'] for x in r.values()):
            labels = [previous.label_key(t, frequencies) for t in a['labels']]
            supports = [frozenset(labels[j] for j,_ in p[2]) for p in a['points']]
            for k,name in enumerate(('X momentum','Y momentum','Continuity')):
                allowed = {form[k] for form in forms}
                ids = {i for i,s in enumerate(supports) if s in allowed}
                first_call = min((a['points'][i][5] for i in ids), default=None)
                epoch = next((c['epoch'] for c in op['epoch_clock'] if first_call and c['calls'] >= first_call), None)
                present = [h['epoch'] for h in a['epoch_fronts'] if any(i in ids for row in h['members'] for i in row)]
                missed_donors.append((seed,name,epoch,present))
        rows.append((seed,r))
    lines = ['# Independent confirmation of donor retention on clean Navier–Stokes', '',
             'Three previously unused search seeds share the original full 200 × 50 × 100 measurement record. '
             'For each seed the two assembly rules use the same native trajectory, weak rows, seven-term cap, '
             'beam width of 1,000 and total fit budget from 150 native epochs. Each permits cross-parent mixing, '
             'one field/axis exchange and joint choice of equation slots and derivative targets. The only '
             'selection change is which fronts supply equations at epoch 100.', '',
             '| Donor pool | Seeds with complete system | Seeds with complete system on global first front |',
             '|---|---:|---:|']
    names = [('current','Epoch 100'), ('historical','Epochs 1–100')]
    for arm, label in names:
        made = sum(r[arm]['complete_generated']>0 for _,r in rows)
        front = sum(r[arm]['complete_nondominated']>0 for _,r in rows)
        lines.append(f'| {label} | {made}/3 | {front}/3 |')
    gain = [seed for seed,r in rows if r['historical']['complete_nondominated'] and not r['current']['complete_nondominated']]
    loss = [seed for seed,r in rows if r['current']['complete_nondominated'] and not r['historical']['complete_nondominated']]
    lines.extend(['', f'Historical retention gained {len(gain)} and lost {len(loss)} paired first-front recoveries. '
                  f'The other {len(rows)-len(gain)-len(loss)} pairs agreed.', '',
                  '| Seed | Donor pool | Donor structures | Assembly refits | Total fits | Complete candidates | On global first front | Still unbeaten against full native run |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|'])
    for seed, r in rows:
        for arm, label in names:
            x = r[arm]
            lines.append(f"| [{seed}](results/{seed}/donor_history.json) | {label} | {x['donor_supports']:,} | "
                         f"{x['new_fits']:,} | {x['total_fits']:,} | {x['complete_generated']} | "
                         f"{x['complete_nondominated']} | {x['surviving_all_native']} |")
    lines.extend(['', 'A complete candidate contains both momentum equations and continuity in one system, '
                  'including registered equivalent forms. The global front compares WAPE and dyadic instability '
                  'per slot against the admitted native and assembled candidates. The last column also tests '
                  'survivors against native candidates beyond each arm’s fit cutoff.', ''])
    if missed_donors:
        lines.extend(['## Availability on seeds where both rules missed assembly', '',
                      '| Seed | Law | First native fit (epoch) | First epoch-end front | Front at epoch 100 | Front at epoch 125 |',
                      '|---:|---|---:|---:|---|---|'])
        for seed,name,epoch,present in missed_donors:
            lines.append(f"| {seed} | {name} | {epoch or '—'} | {present[0] if present else '—'} | "
                         f"{'Yes' if 100 in present else 'No'} | {'Yes' if 125 in present else 'No'} |")
        lines.append('')
    lines.extend(['## Elapsed time', '',
                  '| Seed | Donor pool | Full native run (min) | Native through allotted fit (min) | Assembly and assessment (min) | Sum of allotted native and assembly time (min) |',
                  '|---:|---|---:|---:|---:|---:|'])
    for seed,r in rows:
        for arm,label in names:
            x = r[arm]
            sim, assembly = x['native_cutoff_seconds']/60, x['seconds']/60
            lines.append(f"| {seed} | {label} | {x['native_full_seconds']/60:.1f} | {sim:.1f} | {assembly:.1f} | {sim+assembly:.1f} |")
    lines.extend(['', 'Elapsed time uses a monotonic clock. The two arms have separate weak-column caches. '
                  'Each sum includes native execution through the last charged fit and the timed assembly and '
                  'assessment. Shared archive I/O and scorer setup are excluded. The three seed workers ran concurrently.', '',
                  '[Protocol](protocol.md) · [Earlier seeds](findings.md) · [Pressure-slot dominance on seed 24112401](dominance_24112401.md)'])
    (ROOT/'retention_confirmation.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    build()
