"""Controlled geometry, velocity-window, weak integration, and historical reproduction checks."""
import json
import csv
import numpy as np

from discovery import weak_design, wf, write_json
from impacts import collect, episodes, slope
from physics import DT, FPS, LAG, OUT, ROBOT, SEEDS, Track, features, heading, partition, rotate, swarm_files, swarm_meta


def verify():
    p=np.array([[0.,0.],[220.,20.],[-30.,300.]])
    e=heading(np.array([0.,1.,2.]))
    v=np.array([[12.,3.],[-8.,4.],[2.,1.]])
    w=np.array([2.,-3.,1.])
    x,n=features(p,e,v,w,e*34)
    xr,nr=features(rotate(p),rotate(e),rotate(v),w,rotate(e)*34)
    np.testing.assert_allclose(xr,np.moveaxis(rotate(np.moveaxis(x,-2,-1)),-1,-2),atol=1e-12)
    np.testing.assert_array_equal(n,nr)
    swap=[1,0,2]
    xp,_=features(p[swap],e[swap],v[swap],w[swap],e[swap]*34)
    np.testing.assert_allclose(xp,x[swap])
    np.testing.assert_allclose(x[:,:,-5:].sum(-1),x[:,:,0],atol=1e-12)
    np.testing.assert_array_equal(x[2],0.)
    t=np.arange(80)/FPS
    pos=(11*t+5)[:,None,None]
    base=np.full_like(pos,3.)
    wx,wy=weak_design(pos,np.ones(pos.shape+(1,)),base,(4,4,3))
    np.testing.assert_allclose(wy,8.,atol=1e-10)
    np.testing.assert_allclose(wx,1.,atol=1e-10)
    u=np.sin(3*t)+t**2
    feature=np.cos(2*t)
    fields=wf.Fields({'u':u},[t],opaque={'contact':feature})
    terms=[wf.WeakTerm([('u',(1,))],(0,)),wf.WeakTerm([],(0,),('contact',))]
    reference,_=wf.weak_matrix([term.monomials() for term in terms],[term.opaque for term in terms],
                               fields,[0],[8],[4],[3])
    wx,wy=weak_design(u[:,None,None],feature[:,None,None,None],np.zeros((len(u),1,1)),(8,4,3))
    np.testing.assert_allclose(np.column_stack((wy.ravel(),wx.ravel())),reference,atol=1e-12)
    np.testing.assert_allclose((pos[LAG:]-pos[:-LAG])/DT,11.,atol=1e-10)
    np.testing.assert_allclose(slope(pos,np.array([3,9]),5),11.,atol=1e-10)
    start,end=episodes(np.array([220,217,228,224,219,218,221,219,227,215]),220,226)
    np.testing.assert_array_equal(start,[4,9])
    np.testing.assert_array_equal(end,[8,10])
    mock={key:Track(key,10,np.array([1]),pos,pos[:,:,0]) for key in ['a','b']}
    tr,va,te=partition(mock,'a')
    assert tr[0].tag==va[0].tag=='b' and te.tag=='a'
    assert tr[0].pos[-1,0,0] < va[0].pos[0,0,0]
    separation=np.r_[np.full(8,270.),np.arange(270.,204.,-5.),np.arange(210.,271.,5.),np.full(8,270.)]
    pair_pos=np.zeros((len(separation),2,2))
    pair_pos[:,1,0]=separation
    phi=np.tile([.2,1.1],(len(separation),1))
    pair_track=Track('pair',10,np.array([1,2]),pair_pos,phi)
    def events(tr):
        return collect(tr,np.array([0.,0.,1000.]))['cap_3'][1]
    original=events(pair_track)
    assert len(original)==1 and original[0,0]<0<original[0,5]
    spun=Track('rotated',10,pair_track.ids,rotate(pair_pos),phi+np.pi/2)
    swapped=Track('swapped',10,pair_track.ids[::-1],pair_pos[:,::-1],phi[:,::-1])
    for variant in (spun,swapped):
        np.testing.assert_allclose(events(variant)[:,:9],original[:,:9],atol=1e-10)
    cut=int(original[0,7])+1
    assert not len(events(pair_track.part(0,cut))) and not len(events(pair_track.part(cut,None)))
    from contact_rebound import Outcomes, early_scores, entries
    arena = np.array([0., 0., 1000.])
    separated = entries(pair_track, arena)[3]
    held_pos = pair_pos.copy()
    held_pos[cut:] = held_pos[cut - 1]
    sustained = entries(Track('held', 10, pair_track.ids, held_pos, phi), arena)[3]
    assert len(sustained) == len(separated) == 1
    np.testing.assert_array_equal(sustained['frame'], separated['frame'])
    np.testing.assert_allclose(sustained['closing'], separated['closing'])
    assert separated['release'][0] < sustained['release'][0] == len(held_pos)
    from physics import CAP
    gap = np.r_[[12.]*10, [5., 3., 1., -1., -3., -4., -5., 1., 3., 8.], [12.]*15]
    crossing_pos = np.zeros((len(gap), 2, 2))
    crossing_pos[:, 1, 0] = CAP + gap
    crossing = Track('crossing', 10, pair_track.ids, crossing_pos, np.zeros((len(gap), 2)))
    for offset, frame in ((2, 12), (0, 13), (-2, 14)):
        found = entries(crossing, arena, offset=offset)[3]
        np.testing.assert_array_equal(found['frame'], [frame])
        np.testing.assert_array_equal(found['release'], [19])
    def response(x, slope):
        x = np.array(x)
        y = -.7 + slope * x
        return Outcomes(x, y, np.zeros(len(x), bool), y)
    train = [response([1., 2.], .3), response([3., 4., 5., 6.], .3)]
    test = response([2., 3., 4.], .3)
    score = early_scores(train, [test], test)
    np.testing.assert_allclose(score['training_mean_incoming_travel_mm'], 3.)
    np.testing.assert_allclose(score['selected_offset_speed_multiplier'], .3)
    np.testing.assert_allclose(score['selected_offset_speed_rmse_mm'], 0., atol=1e-12)
    assert early_scores(train, [response([2., 3., 4.], -.3)], test)['selected_offset_speed_multiplier'] == 0.
    print('Controlled checks passed')


def legacy():
    import local_law as local
    import swarm_contact_law as old
    import swarm_sim as sim
    prior=json.loads((ROBOT/'results'/'swarm_contact_law.json').read_text())
    baseline={}
    for f in swarm_files():
        tag=swarm_meta(f)['tag']
        R=sim.recording(f)
        h=old.heldout(R,lambda rows:sim.fit_joint(R,True,False,rows)[0],sim.recording_band_push(R))
        original=prior[tag]['heldout']['own_prop']
        np.testing.assert_allclose(list(h.values()),[original[k] for k in h],atol=1e-7)
        baseline[tag]=h
    d=local.build(swarm_files()[0])
    m=d['n']==0
    a,y,g=local.stack(d,local.SELF,m)
    own=local.cv_select(a,y,g,local.SELF)[0]
    m=d['n']==1
    a,y,g=local.stack(d,local.PAIR,m)
    pair=local.cv_select(a,y-local.predict(d,own,m)[0],g,local.PAIR)[0]
    pair.pop('const',None)
    archived=json.loads((ROBOT/'results'/f"local_law_{d['tag']}.json").read_text())
    for name,terms in [('self',own),('pair',pair)]:
        assert set(terms)==set(archived[name])
        np.testing.assert_allclose(list(terms.values()),[archived[name][k] for k in terms],rtol=1e-6)
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'legacy.json',dict(local_pair=pair,local_self=own,heldout=baseline))
    print('Historical coefficients and all five held-out baseline scores reproduced')


def verify_outputs():
    """Audit completed archives, forecast receipts and the published tables on disk."""
    tags={swarm_meta(f)['tag'] for f in swarm_files()}
    runs=0
    for tag in tags:
        dest=OUT/tag
        load=lambda name:json.loads((dest/name).read_text())
        training=load('training.json')
        assert set(training['train'])==tags-{tag}
        for seed in SEEDS:
            for form in ('strong','weak'):
                item=load(f'{form}_{seed}.json')
                assert (item['form'],item['seed'],item['epochs'])==(form,seed,50)
                assert item['evaluated'] and item['native_front'] and item['population']>0
        assessment=load('assessment.json')
        assert assessment['held']==tag
        for model in assessment['models'].values():
            assert np.isfinite(model['heldout']['r2'])
        for method in assessment['simulations'].values():
            assert set(method)=={'cap','unprojected'}
            for reps in method.values():
                assert len(reps)==4
                for r in reps:
                    np.testing.assert_allclose([sum(r['contacts']),sum(r['gap_hist'])],1.)
                    assert 0<=r['overlap_fraction']<=1
                    runs+=1
        impact=load('impacts.json')
        with (dest/'heldout_events.csv').open(newline='') as fh:
            assert sum(1 for _ in csv.DictReader(fh))==impact['cap_3']['counts']['resolved']
        for cell in impact.values():
            assert cell['counts']['resolved']<=cell['counts']['approaches']
    assert runs==len(tags)*16
    for name,expected in [('drift_scores.csv',len(tags)*6),('event_sensitivity.csv',len(tags)*10)]:
        with (OUT/name).open(newline='') as fh:
            assert sum(1 for _ in csv.DictReader(fh))==expected
    if (OUT/'contact_rebound.csv').exists():
        from contact_rebound import CLEARANCES, HORIZONS, WIDTHS
        with (OUT/'contact_rebound.csv').open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        keys = {(r['recording'], r['isolation'], int(r['pre_frames']), float(r['horizon_s'])) for r in rows}
        assert len(keys) == len(rows)
        assert keys == {(tag, label, w, h/FPS) for tag in tags for label in CLEARANCES
                        for w in WIDTHS for h in HORIZONS}
        for r in rows:
            assert 0 <= int(r['released_by_horizon']) <= int(r['test_events'])
            assert 0 <= float(r['selected_multiplier']) <= 1
            if float(r['selected_multiplier']) == 0:
                assert r['baseline_rmse_mm'] == r['selected_rmse_mm']
    if (OUT/'early_contact.csv').exists():
        from contact_rebound import CLEARANCES, EARLY_HORIZONS, ENTRY_OFFSETS_PX, WIDTHS
        with (OUT/'early_contact.csv').open(newline='') as fh:
            early = list(csv.DictReader(fh))
        cohorts = {}
        keys = set()
        for r in early:
            cohort = (r['recording'], r['isolation'], int(r['pre_frames']), int(r['entry_offset_px']))
            keys.add((*cohort, float(r['horizon_s'])))
            n = int(r['test_events'])
            assert 0 <= int(r['outward_over_one_px']) <= int(r['outward_events']) <= n
            assert cohorts.setdefault(cohort, n) == n
            assert 0 <= float(r['selected_offset_speed_multiplier']) <= 1
            if float(r['selected_offset_speed_multiplier']) == 0:
                assert r['calibrated_offset_rmse_mm'] == r['selected_offset_speed_rmse_mm']
        assert len(keys) == len(early)
        assert keys == {(tag, label, w, offset, h/FPS) for tag in tags for label in CLEARANCES
                        for w in WIDTHS for offset in ENTRY_OFFSETS_PX for h in EARLY_HORIZONS}
        if (OUT/'contact_rebound.csv').exists():
            for r in rows:
                nominal = (r['recording'], r['isolation'], int(r['pre_frames']), 0)
                assert cohorts[nominal] == int(r['test_events'])
    for name in ('pooled_law.json','report.md','distance_response.png','heldout_bond_velocity.png','provenance.json'):
        assert (OUT/name).stat().st_size>0
    print(f'Verified {len(tags)} transfer folds, {len(tags)*len(SEEDS)*2} search archives and {runs} rollouts')


if __name__=='__main__':
    verify()
    legacy()
