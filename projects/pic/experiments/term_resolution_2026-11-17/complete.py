import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,digest


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    confirmation=read('forcing_audit.json')
    fit=read('forcing_profile.json')
    assert confirmation['status']=='PASS' and len(confirmation['records'])==12
    assert len(fit['rows'])==36
    assert read('audit.json')['certificates_sha256']==digest(ROOT/'certificates.json')
    assert read('projected_audit.json')['status']=='PASS'
    streams=[read('fresh_replay.json')['rows'],read('projected_replay.json')['rows'],
             read('forcing_profile/replay.json')['rows']]
    assert list(map(len,streams))==[6,6,12]
    draws=sum(r.get('draws',r.get('field_replays',0)) for group in streams for r in group)
    assert draws==3648
    fresh=read('fresh_analysis.json')['records']; earlier=read('projected_audit.json')['records']
    all_hashes=[x for s in streams[:2] for r in s for x in r['raw'].values()]
    for r in confirmation['records']:
        data=read(f"forcing/{r['profile']}_{r['seed']}.json")
        all_hashes.extend(data['raw_hash'].values())
    assert len(all_hashes)==len(set(all_hashes))==24
    assert sum(len(r['warnings']) for r in fresh+earlier+confirmation['records'])==0
    summary=read('summary.json')
    summary.update(fresh_measurement_fields=24,bootstrap_noise_refits=draws,
        forcing_confirmation=confirmation['total'],forcing_by_profile=confirmation['by_profile'],
        unrestricted_confirmation=fit['totals'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

    arms=('whole','pilot','forcing')
    result=[confirmation['total'][k]['accepted'] for k in arms]
    fig,ax=plt.subplots(figsize=(8.4,3.6),layout='constrained')
    y=np.arange(3)
    for j,(label,color) in enumerate([('Physical equation retained','#237268'),
                                     ('Dispersion-only equation rejected','#a04f2e')]):
        vals=[r[0] if j==0 else 12-r[1] for r in result]
        bars=ax.barh(y+(j-.5)*.30,vals,.28,color=color,label=label)
        ax.bar_label(bars,padding=4)
    ax.set(yticks=y,yticklabels=['Whole residual','Pilot-based projection','Known forcing projection'],
           xlabel='Fresh measurements, out of 12',xlim=(0,13.2),xticks=range(0,13,2))
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
    ax.set_title('Discriminating two fixed forced-KdV equations on the same trajectory')
    fig.savefig(ROOT/'forcing_confirmation.png',dpi=190);fig.savefig(ROOT/'forcing_confirmation.pdf');plt.close(fig)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'
                 and '__pycache__' not in p.parts)
    manifest={str(p.relative_to(ROOT)):digest(p) for p in files}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    assert all(digest(ROOT/name)==h for name,h in manifest.items())
    print('Verified',len(files),'artifacts,',len(all_hashes),'distinct fresh noise fields,',draws,'bootstrap refits.')


if __name__=='__main__':
    main()
