from adequacy_context import *
from functools import lru_cache
from datetime import datetime,timezone

@lru_cache(None)
def digest(p):return sha(p)

def main():
    for n in ('process','bootstrap_process','target_process','bootstrap_summary','summary','development','development_adjacent'):assert read(ROOT/(n+'.json'))['status']=='COMPLETE'
    for n in ('checks','audit','failure_analysis','bootstrap_audit','target_audit','target_scan_audit'):assert read(ROOT/(n+'.json'))['status']=='PASS'
    bs=read(ROOT/'bootstrap_process.json');ts=read(ROOT/'target_process.json');expected=dict(frozen=1832,results=1832,audits=1832,bootstrap_anchors=48,bootstrap_chunks=bs['active']*16,bootstrap_frozen=48,bootstrap_results=48,bootstrap_audits=48,target_anchors=256,target_chunks=ts['active']*16,target_frozen=ts['active'],target_results=ts['active'],target_audits=ts['active'])
    for folder,n in expected.items():assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    links=jsons=arrays=0
    def visit(x):
        nonlocal links
        if isinstance(x,list):
            for v in x:visit(v)
        elif isinstance(x,dict):
            for k,v in x.items():
                if k in ('source_sha256','input_sha256'):
                    for p,h in v.items():assert digest(p)==h,(k,p);links+=1
                else:visit(v)
    for p in ROOT.rglob('*.json'):
        if p.name in ('artifact_manifest.json','completion_audit.json'):continue
        d=read(p);visit(d);jsons+=1
        if 'array_sha256' in d:assert digest(str(p.with_suffix('.npz')))==d['array_sha256'];arrays+=1
    unchanged={}
    for name in ('spline_compatibility_2026-11-17','spline_noise_2026-11-17','nuisance_calibration_2026-11-17','safeguard_envelope_2026-11-17','nuisance_direction_2026-11-17','identifiability_design_2026-11-17','full_pipeline_uncertainty_2026-11-17'):
        folder=ROOT.parent/name;m=read(folder/'artifact_manifest.json')
        for rel,h in m['sha256'].items():assert digest(str(folder/rel))==h,(name,rel)
        unchanged[name]=len(m['sha256'])
    audit=read(ROOT/'audit.json');ba=read(ROOT/'bootstrap_audit.json');ta=read(ROOT/'target_audit.json');scan=read(ROOT/'target_scan_audit.json');s=read(ROOT/'summary.json')
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,array_hashes=arrays,hash_links=links,counts=expected,main_audit_totals={k:sum(r[k] for r in audit['rows']) for k in ('selection_replays','dense_f_checks','dense_likelihood_checks','independent_limit_checks','fine_grid_checks')},bootstrap_audit_totals={k:sum(r[k] for a in (ba,ta) for r in a['rows']) for k in ('draws','full_selection_replays','fit_parameter_replays')},target_scan_replays=len(scan['rows']),fine_grid_improvement_records=len(s['fine_grid_improvements']),previous_artifacts_unchanged=unchanged,source_sha256=src('adequacy_complete.py')))
    marker='2026-11-17: Automatic spline adequacy and selection-aware bootstrap';log=ROOT.parent/'solver_free_research_2026-11-16.txt';entry='\n\n'+marker+'\nspline_adequacy_2026-11-17/findings.txt\n'
    entry+='1832 fresh noisy fields:264 benchmarks,1024 positive ODE stress,288\nanalytic signals,256 exact-model controls. Primary: adjacent-noise GLS\nextra-mean screen(alpha.01), positive-boundary LR exclusion, then CV.\nAll1832 selections independently replayed. No dropped cases.\nPositive ODE usable nominal coverage911 ->949/1024; boundary outputs2 ->0.\nRemaining p24 stratum:42 selected,only5 covered; median sigma bias+22.1%.\nNonrejection of extra signal is not evidence of negligible mean bias.\nAll48 highest-frequency analytic cases correctly cause abstention under\nthe rule, but broad validity is not established. No solver or production edit.\n'
    entry+='48 new general ODE fields,6144 full bootstrap refits:45/48 nominal ->\n48/48 calibrated, but10 whole-space regions. All observed dimensionsp32.\n256 additional positive fields scanned, first8 p24 choices frozen before\ntruth scoring;1024 extra refits. Conditional p24 test: all8 calibrated\nregions whole-space. This is not unconditional coverage or useful precision.\n'
    entry+='Audits/source hashes/prior manifests pass. Next: address approximation\nbias in selection, rather than further threshold tuning or merely inflating\ncutoffs. Constants/valid identities retained. No dyad/KdV/full15 replacement.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}));print('COMPLETE',jsons,'JSON',links,'hash links',arrays,'arrays',unchanged,flush=True)

if __name__=='__main__':main()
