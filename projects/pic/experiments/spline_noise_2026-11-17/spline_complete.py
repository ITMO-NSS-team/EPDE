from spline_context import *
from functools import lru_cache
from datetime import datetime,timezone

@lru_cache(None)
def digest(path):return sha(path)


def main():
    for name in ('point_process','bootstrap_process','propagation_process','summary'):assert read(ROOT/(name+'.json'))['status']=='COMPLETE'
    for name in ('checks','point_audit','bootstrap_audit','propagation_audit','leakage_diagnostic','union_review'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    expected=dict(point_frozen=264,point_results=264,point_audits=264,bootstrap_anchors=24,bootstrap_chunks=384,bootstrap_frozen=24,bootstrap_results=24,bootstrap_audits=24,propagation_primary=24,propagation_chunks=528,propagation_frozen=24,propagation_results=24,propagation_audits=24,union_reviews=24)
    for folder,n in expected.items():assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    for p in (ROOT/'bootstrap_anchors').glob('*_spline.json'):
        a=read(p);b=read(p.with_name(p.name.replace('_spline','_guarded')));assert a['raw_sha256']==b['raw_sha256'];assert a['seed']==b['seed'] and a['profile']==b['profile']
    links=jsons=arrays=0
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
        d=read(p);visit(d);jsons+=1
        if 'array_sha256' in d:assert digest(str(p.with_suffix('.npz')))==d['array_sha256'];arrays+=1
    unchanged={}
    for name in ('nuisance_calibration_2026-11-17','safeguard_envelope_2026-11-17','nuisance_direction_2026-11-17','identifiability_design_2026-11-17','full_pipeline_uncertainty_2026-11-17'):
        folder=ROOT.parent/name;manifest=read(folder/'artifact_manifest.json')
        for rel,h in manifest['sha256'].items():assert digest(str(folder/rel))==h,(name,rel)
        unchanged[name]=len(manifest['sha256'])
    pa=read(ROOT/'propagation_audit.json');ba=read(ROOT/'bootstrap_audit.json');aa=read(ROOT/'point_audit.json');summary=read(ROOT/'summary.json')
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,array_hashes=arrays,hash_links=links,distinct_linked_files=digest.cache_info().currsize,cohort_counts=expected,fit_audit_totals={k:sum(r.get(k,0) for d in (aa,ba) for r in d['rows']) for k in ('original_fit_replays','spline_selection_replays','dense_selected_checks','field_hashes')},propagation_totals={k:sum(r[k] for r in pa['rows']) for k in ('generalized_scores','certificates','loewner_bounds')},previous_artifacts_unchanged=unchanged,source_sha256=sources('spline_complete.py')))
    marker='2026-11-17: Deterministic spline means for noise estimation';log=ROOT.parent/'solver_free_research_2026-11-16.txt';entry='\n\n'+marker+'\nspline_noise_2026-11-17/findings.txt\n'
    entry+='264 fresh paired fields(ODE192,VDP48,Duffing24); full AR1 REML with\ncubic spline mean; dimension selected by held-out prediction, no truth.\nPositive ODE median sigma/truth1.4354 ->0.9987; rho error+.2948 ->-.0086.\nFive positive fields selected p24 and retained signal in noise; worst+27.8%.\nAll264 point cases independently audited; no selected-boundary warning.\n'
    entry+='Fresh12 paired ODE fields x128 refits x2 methods=3072 refits, spline\nselection repeated in every replicate, same correlated-GCV pilot rule.\n'
    entry+='Noise coverage: '+str(summary['noise_coverage'])+'\nWeak-error coverage: '+str(summary['weak_error_coverage'])+'\n'
    entry+='Paired spline/guarded outer-width ratios: '+str(summary['paired_outer_width_ratio_spline_over_guarded'])+'\n'
    entry+='Reference geometry: '+str(summary['reference_geometry'])+'\n'
    entry+='66 scenarios per method/field,128 known-noise conditional fields each.\nFinite union retains both primary regions. No continuous-region or95%\ncoverage guarantee from this small cohort. Outer widths are enclosures.\nAll numerical/provenance audits pass; prior manifests unchanged. No solver,\nproduction edit, discarded draw or removal of constants/valid identities.\nThis is ode, not ode_simple. KdV remains unresolved; dyad provisional.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}));print('COMPLETE',jsons,'JSON',links,'hash links',arrays,'arrays',unchanged,flush=True)


if __name__=='__main__':main()
