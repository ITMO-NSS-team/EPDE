from study_context import *
from functools import lru_cache
from datetime import datetime,timezone
from scipy.spatial import ConvexHull
from collections import Counter


@lru_cache(None)
def digest(path):return sha(path)


def main():
    for name in ('point_process','envelope_process','evaluation_process','diagnostic_process'):assert read(ROOT/(name+'.json'))['status']=='COMPLETE'
    for name in ('native_arithmetic_checks','native_fit_checks','envelope_checks','audit'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    results=sorted((ROOT/'results').glob('*.json'));assert len(results)==12;components=0;truth_checks=0
    for path in results:
        r=read(path);key=r['key'];u=read(ROOT/'union_diagnostic'/(key+'.json'));pts=read(ROOT/'nuisance_points'/(key+'.json'))['points'];assert len(u['points'])==129
        allc=u['primary_components']+[v['certificate'] for v in u['points']];assert len(allc)==131;components+=len(allc)
        assert dict(Counter(v['geometry'] for v in allc))==u['component_counts'];assert all(c['geometry']=='bounded_outer' for c in allc)
        np.testing.assert_array_equal(np.min([c['outer_region']['lower'] for c in allc],axis=0),u['outer_box']['lower']);np.testing.assert_array_equal(np.max([c['outer_region']['upper'] for c in allc],axis=0),u['outer_box']['upper'])
        theta=np.array([v['transformed'] for v in pts]);known=np.array([np.log(r['model']['u'][0]/r['sigma_truth_ratio']),np.arctanh(r['model']['u'][1][0]-r['rho_error'])]);hull=ConvexHull(theta);inside=bool(np.all(hull.equations[:,:2]@known+hull.equations[:,2]<=1e-10));assert inside==r['nuisance_truth_in_basic_convex_hull'];truth_checks+=1
    links=jsons=0
    def visit(obj):
        nonlocal links
        if isinstance(obj,list):
            for v in obj:visit(v)
        elif isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('source_sha256','input_sha256'):
                    for p,h in v.items():assert digest(p)==h,(k,p);links+=1
                else:visit(v)
    for p in ROOT.rglob('*.json'):
        if p.name in ('completion_audit.json','artifact_manifest.json'):continue
        visit(read(p));jsons+=1
    unchanged={}
    for prev in (PREVIOUS,C.ROOT,C.PREVIOUS):
        d=read(prev/'artifact_manifest.json')
        for rel,h in d['sha256'].items():assert sha(prev/rel)==h,(prev,rel)
        unchanged[prev.name]=len(d['sha256'])
    helpers={str(C.ROOT/'audit_new.py'):sha(C.ROOT/'audit_new.py'),str(C.PREVIOUS/'audit_verified.py'):sha(C.PREVIOUS/'audit_verified.py')}
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,hash_links=links,distinct_linked_files=digest.cache_info().currsize,union_components_checked=components,independent_convex_hulls=truth_checks,previous_artifacts_unchanged=unchanged,source_sha256={**sources('complete.py'),**helpers}))
    s=read(ROOT/'summary.json');log=ROOT.parent/'solver_free_research_2026-11-16.txt';marker='2026-11-17: Guarded full bootstrap and finite nuisance sensitivity'
    entry='\n\n'+marker+'\nsafeguard_envelope_2026-11-17/findings.txt\n'
    entry+='12 ODE datasets,128 guarded bootstrap refits each. Three old positive\nobserved fields rerun; nine fresh fields under iid,+.4,-.4. Historical3/3\nunbounded reference regions become bounded; all9 fresh regions bounded.\n'
    entry+='Fresh full-library error coverage8/9 in both normalizations. Negative/60\nmisses both banks but reference-equation projection remains covered.\n1548 selected fits all ok, no near-unit values;7 guard triggers(5 observed).\nPositive observed amplitude remains about40% high on the fresh fields.\n'
    entry+='Finite129-point noise sensitivity,128 conditional fields perpoint. Scalar\nenvelope covers12/12 but makespositive/57 unbounded. Exploratory separate\nscenario union retains bounded reference regions12/12(all131 components\nbounded each). Median union outer-width ratio1.418 versus original\nstudentized outer enclosure. This is a finite sensitivity diagnostic,\nNOT a calibrated nuisance confidence set or adaptive-search guarantee.\n'
    entry+='True-noise covariance reducesnegative/60 broad score9.598->7.427;\nchanging correction alone barely matters. With both known,7.447 remains\nabove its separate known-noise conditional cutoff7.195. One case cannot\ndistinguish a chance tail from systematic undercoverage.\n'
    entry+='104 native arithmetic checks,54 complete fits and1548 original-Python\nfit replays passed. All396288 conditional generalized scores and3096\nLoewner bounds audited. Previous692+357+773 artifact files unchanged.\nNo solver, production edit, clipping or dropped scenarios. Constants and\nvalid identities retained. KdV weakness remains unresolved; dyad provisional.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('COMPLETE',jsons,'JSON',links,'hash links',unchanged,flush=True)


if __name__=='__main__':main()
